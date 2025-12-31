/// Winkler Foundation: Elastic Restoring Force at Bottom Boundary
///
/// Models isostatic support from underlying mantle:
///   F_z = -k * (z - z₀)
///
/// where:
/// - k = foundation stiffness (Pa/m)
/// - z = current elevation
/// - z₀ = reference (initial) elevation
///
/// **Physical Meaning:**
/// - Prevents unrealistic bottom subsidence during extension
/// - Simulates buoyant restoring force from mantle
/// - Common in geodynamic models of lithospheric deformation
///
/// **Typical Values:**
/// - k ~ 10⁷-10⁸ Pa/m for lithosphere-asthenosphere boundary
/// - Stiffer (larger k) = stronger resistance to deflection

use crate::mesh::Mesh;
use crate::fem::DofManager;
use std::collections::HashMap;

/// Winkler foundation parameters
#[derive(Debug, Clone)]
pub struct WinklerFoundation {
    /// Foundation stiffness (Pa/m)
    pub stiffness: f64,
    /// Reference elevations stored separately
    reference_elevations: HashMap<usize, f64>,
}

impl WinklerFoundation {
    /// Create new Winkler foundation
    ///
    /// # Arguments
    /// * `stiffness` - Foundation stiffness in Pa/m
    ///   - Typical: 10⁷-10⁸ Pa/m for mantle support
    ///   - Higher = stronger restoring force
    pub fn new(stiffness: f64) -> Self {
        assert!(stiffness > 0.0, "Stiffness must be positive");
        Self {
            stiffness,
            reference_elevations: HashMap::new(),
        }
    }

    /// Initialize reference elevations from current mesh
    ///
    /// Must be called before first use to store initial bottom elevations
    ///
    /// # Arguments
    /// * `mesh` - Mesh to extract bottom node elevations from
    pub fn initialize_reference(&mut self, mesh: &Mesh) {
        let bottom_nodes = identify_bottom_nodes(mesh);
        self.reference_elevations.clear();

        for &node_id in &bottom_nodes {
            self.reference_elevations.insert(node_id, mesh.geometry.nodes[node_id].z);
        }
    }

    /// Compute Winkler restoring forces for bottom boundary
    ///
    /// Returns force vector f where:
    ///   f[3*node + 2] = -k * (z - z₀) for bottom nodes
    ///   f[...] = 0 for all other DOFs
    ///
    /// # Arguments
    /// * `mesh` - Current mesh configuration
    /// * `dof_mgr` - DOF manager (3 DOF/node)
    ///
    /// # Panics
    /// Panics if reference elevations not initialized
    pub fn compute_forces(&self, mesh: &Mesh, dof_mgr: &DofManager) -> Vec<f64> {
        assert!(!self.reference_elevations.is_empty(),
                "Must call initialize_reference() before compute_forces()");

        let mut forces = vec![0.0; dof_mgr.total_dofs()];

        for (&node_id, &z_ref) in &self.reference_elevations {
            let z_current = mesh.geometry.nodes[node_id].z;
            let deflection = z_current - z_ref;

            // Restoring force: F = -k * Δz (opposes deflection)
            let f_z = -self.stiffness * deflection;

            // Add to global force vector (z-component)
            let dof_z = dof_mgr.global_dof(node_id, 2);
            forces[dof_z] = f_z;
        }

        forces
    }

    /// Get nodes where Winkler force is applied
    pub fn get_active_nodes(&self) -> Vec<usize> {
        self.reference_elevations.keys().copied().collect()
    }
}

/// Identify bottom boundary nodes (z = z_min)
fn identify_bottom_nodes(mesh: &Mesh) -> Vec<usize> {
    let tol = 1e-6;

    // Find min z coordinate
    let z_min = mesh.geometry.nodes.iter()
        .map(|node| node.z)
        .fold(f64::INFINITY, f64::min);

    // Collect nodes at z_min
    mesh.geometry.nodes.iter()
        .enumerate()
        .filter(|(_, node)| (node.z - z_min).abs() < tol)
        .map(|(id, _)| id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_generator_improved::ImprovedMeshGenerator;

    #[test]
    fn test_identify_bottom() {
        let mesh = ImprovedMeshGenerator::generate_cube(4, 4, 4, 10.0, 10.0, 30.0);
        let bottom = identify_bottom_nodes(&mesh);

        assert!(bottom.len() > 0, "Should find bottom nodes");

        // All bottom nodes should have z = 0.0
        for &node_id in &bottom {
            let z = mesh.geometry.nodes[node_id].z;
            assert!(z.abs() < 1e-5, "Bottom node should be at z=0, got z={}", z);
        }
    }

    #[test]
    fn test_winkler_initialization() {
        let mesh = ImprovedMeshGenerator::generate_cube(2, 2, 2, 1.0, 1.0, 1.0);
        let mut winkler = WinklerFoundation::new(1e7);

        winkler.initialize_reference(&mesh);

        let active = winkler.get_active_nodes();
        assert!(active.len() > 0, "Should have active nodes");
    }

    #[test]
    fn test_winkler_restoring_force() {
        let mut mesh = ImprovedMeshGenerator::generate_cube(2, 2, 2, 1.0, 1.0, 1.0);
        let dof_mgr = DofManager::new(mesh.num_nodes(), 3);

        let mut winkler = WinklerFoundation::new(1e7); // Pa/m
        winkler.initialize_reference(&mesh);

        // Deflect bottom by 0.1 m downward
        let bottom = identify_bottom_nodes(&mesh);
        for &node_id in &bottom {
            mesh.geometry.nodes[node_id].z -= 0.1;
        }

        // Compute restoring forces
        let forces = winkler.compute_forces(&mesh, &dof_mgr);

        // Check that forces are upward (positive z) for deflected nodes
        let mut found_upward_force = false;
        for &node_id in &bottom {
            let dof_z = dof_mgr.global_dof(node_id, 2);
            let f_z = forces[dof_z];

            // Deflected down, so force should be upward (positive)
            if f_z > 0.0 {
                found_upward_force = true;
            }

            // Expected: F = -k * (-0.1) = +1e6 N
            let expected = 1e7 * 0.1; // = 1e6 N
            assert!((f_z - expected).abs() < 1.0,
                    "Expected F ≈ {} N, got {} N", expected, f_z);
        }

        assert!(found_upward_force, "Should have upward restoring forces");
    }

    #[test]
    fn test_winkler_zero_for_no_deflection() {
        let mesh = ImprovedMeshGenerator::generate_cube(2, 2, 2, 1.0, 1.0, 1.0);
        let dof_mgr = DofManager::new(mesh.num_nodes(), 3);

        let mut winkler = WinklerFoundation::new(1e7);
        winkler.initialize_reference(&mesh);

        // No deflection
        let forces = winkler.compute_forces(&mesh, &dof_mgr);

        // All forces should be zero
        for &f in &forces {
            assert!(f.abs() < 1e-10, "Force should be zero without deflection");
        }
    }
}
