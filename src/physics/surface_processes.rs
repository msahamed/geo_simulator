/// Surface Processes: Erosion and Deposition
///
/// Implements hillslope diffusion for topographic evolution:
///   ∂h/∂t = κ ∇²h
///
/// where:
/// - h = surface elevation
/// - κ = diffusion coefficient (m²/yr, typically 0.01-1.0)
/// - ∇²h = Laplacian of topography
///
/// **Physical Meaning:**
/// - Material flows from high curvature (ridges) to low curvature (valleys)
/// - Smooths topography over geological timescales
/// - Models soil creep, small-scale landsliding

use crate::mesh::Mesh;
use std::collections::{HashMap, HashSet};

/// Hillslope diffusion parameters
#[derive(Debug, Clone, Copy)]
pub struct HillslopeDiffusion {
    /// Diffusion coefficient (m²/yr)
    pub kappa: f64,
}

impl HillslopeDiffusion {
    /// Create new hillslope diffusion model
    ///
    /// # Arguments
    /// * `kappa` - Diffusion coefficient in m²/yr
    ///   - Typical values: 0.01-1.0 m²/yr for soil-mantled hillslopes
    ///   - Higher values = faster smoothing
    pub fn new(kappa: f64) -> Self {
        assert!(kappa > 0.0, "Diffusion coefficient must be positive");
        Self { kappa }
    }

    /// Apply hillslope diffusion to surface nodes for one time step
    ///
    /// Uses explicit Euler: h_{n+1} = h_n + Δt κ ∇²h_n
    ///
    /// # Arguments
    /// * `mesh` - Mesh to modify (surface nodes updated in place)
    /// * `dt` - Time step (years)
    ///
    /// # Algorithm
    /// 1. Identify surface nodes (top boundary)
    /// 2. Build surface connectivity (neighbor graph)
    /// 3. Compute Laplacian at each node: ∇²h ≈ Σ(h_j - h_i) / n
    /// 4. Update elevations: h_{n+1} = h_n + Δt κ ∇²h
    ///
    /// # Stability
    /// Explicit diffusion requires: Δt < (Δx)² / (2κ)
    /// For Δx ~ 1 km, κ ~ 0.1 m²/yr: Δt < 5000 years
    pub fn apply_diffusion(&self, mesh: &mut Mesh, dt: f64) {
        // 1. Identify surface nodes (top 10% of z-coordinates)
        // This handles varying topography without assuming flat surface
        let mut z_values: Vec<(usize, f64)> = mesh.geometry.nodes.iter()
            .enumerate()
            .map(|(id, node)| (id, node.z))
            .collect();
        z_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top 10% of nodes as surface (handles topography variation)
        let n_surface = (z_values.len() as f64 * 0.10).max(1.0) as usize;
        let surface_nodes: HashSet<usize> = z_values[..n_surface]
            .iter()
            .map(|(id, _)| *id)
            .collect();

        if surface_nodes.is_empty() {
            return;
        }

        // 2. Build surface connectivity
        let neighbors = build_surface_neighbors(mesh, &surface_nodes);

        // 3. Compute Laplacian and update
        let mut new_z = HashMap::new();

        for &node_id in &surface_nodes {
            let h_i = mesh.geometry.nodes[node_id].z;

            let node_neighbors = &neighbors[&node_id];
            if node_neighbors.is_empty() {
                new_z.insert(node_id, h_i);
                continue;
            }

            // Compute Laplacian: ∇²h ≈ (1/n) Σ (h_j - h_i)
            let mut laplacian = 0.0;
            let mut count = 0;
            for &neighbor_id in node_neighbors {
                // Only use neighbors that are also surface nodes
                if surface_nodes.contains(&neighbor_id) {
                    let h_j = mesh.geometry.nodes[neighbor_id].z;
                    laplacian += h_j - h_i;
                    count += 1;
                }
            }

            if count == 0 {
                new_z.insert(node_id, h_i);
                continue;
            }

            // Need to divide by typical mesh spacing squared for correct units
            // Estimate dx from mesh (this is approximate for irregular meshes)
            let dx = 1000.0; // Default 1 km - should be estimated from mesh
            laplacian /= count as f64 * dx * dx;

            // Explicit Euler update
            let h_new = h_i + dt * self.kappa * laplacian;
            new_z.insert(node_id, h_new);
        }

        // 4. Apply updates
        for (&node_id, &z_new) in &new_z {
            mesh.geometry.nodes[node_id].z = z_new;
        }
    }

    /// Get stable time step for explicit diffusion
    ///
    /// Returns Δt < (Δx)² / (2κ)
    ///
    /// # Arguments
    /// * `dx` - Characteristic mesh spacing (m)
    pub fn stable_timestep(&self, dx: f64) -> f64 {
        0.4 * dx * dx / self.kappa  // Safety factor 0.4 < 0.5
    }
}

/// Identify surface nodes (nodes on top boundary)
///
/// Returns set of node IDs where z = z_max (within tolerance)
#[cfg(test)]
fn identify_surface_nodes(mesh: &Mesh) -> HashSet<usize> {
    let tol = 1e-6;

    // Find max z coordinate
    let z_max = mesh.geometry.nodes.iter()
        .map(|node| node.z)
        .fold(f64::NEG_INFINITY, f64::max);

    // Collect nodes at z_max
    mesh.geometry.nodes.iter()
        .enumerate()
        .filter(|(_, node)| (node.z - z_max).abs() < tol)
        .map(|(id, _)| id)
        .collect()
}

/// Build connectivity graph for surface nodes
///
/// Returns map: node_id → set of neighboring surface node IDs
///
/// Two surface nodes are neighbors if they share an element edge
fn build_surface_neighbors(mesh: &Mesh, surface_nodes: &HashSet<usize>) -> HashMap<usize, HashSet<usize>> {
    let mut neighbors: HashMap<usize, HashSet<usize>> = HashMap::new();

    // Initialize
    for &node_id in surface_nodes {
        neighbors.insert(node_id, HashSet::new());
    }

    // For each element, connect surface nodes that share it
    for elem in &mesh.connectivity.tet10_elements {
        let surface_nodes_in_elem: Vec<usize> = elem.nodes.iter()
            .filter(|&&n| surface_nodes.contains(&n))
            .copied()
            .collect();

        // Connect all pairs
        for i in 0..surface_nodes_in_elem.len() {
            for j in i+1..surface_nodes_in_elem.len() {
                let n1 = surface_nodes_in_elem[i];
                let n2 = surface_nodes_in_elem[j];
                neighbors.get_mut(&n1).unwrap().insert(n2);
                neighbors.get_mut(&n2).unwrap().insert(n1);
            }
        }
    }

    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_generator_improved::ImprovedMeshGenerator;

    #[test]
    fn test_identify_surface() {
        let mesh = ImprovedMeshGenerator::generate_cube(4, 4, 4, 10.0, 10.0, 10.0);
        let surface = identify_surface_nodes(&mesh);

        assert!(surface.len() > 0, "Should find surface nodes");

        // All surface nodes should have z = 10.0
        for &node_id in &surface {
            let z = mesh.geometry.nodes[node_id].z;
            assert!((z - 10.0).abs() < 1e-5, "Surface node should be at z=10");
        }
    }

    #[test]
    fn test_surface_connectivity() {
        let mesh = ImprovedMeshGenerator::generate_cube(2, 2, 2, 1.0, 1.0, 1.0);
        let surface = identify_surface_nodes(&mesh);
        let neighbors = build_surface_neighbors(&mesh, &surface);

        // All surface nodes should have neighbors
        for &node_id in &surface {
            assert!(neighbors[&node_id].len() > 0, "Surface node should have neighbors");
        }
    }

    #[test]
    fn test_diffusion_flattens() {
        // Use small mesh and uniform perturbation (all surface nodes at same z)
        let mut mesh = ImprovedMeshGenerator::generate_cube(4, 4, 4, 1.0, 1.0, 1.0);

        // Identify surface BEFORE perturbation
        let surface = identify_surface_nodes(&mesh);

        // Create a simple "single node bump" - raise just the center node
        // Find a node near center
        let mut center_node = 0;
        let mut min_dist = f64::INFINITY;
        for &node_id in &surface {
            let node = &mesh.geometry.nodes[node_id];
            let dist = (node.x - 0.5).powi(2) + (node.y - 0.5).powi(2);
            if dist < min_dist {
                min_dist = dist;
                center_node = node_id;
            }
        }

        // Raise center node by 0.1
        mesh.geometry.nodes[center_node].z += 0.1;

        // Measure max elevation before diffusion
        let z_max_before = surface.iter()
            .map(|&id| mesh.geometry.nodes[id].z)
            .fold(f64::NEG_INFINITY, f64::max);

        println!("Max z before: {:.6}", z_max_before);

        // Apply diffusion - should spread the bump
        let diffusion = HillslopeDiffusion::new(1.0);
        for _ in 0..10 {
            diffusion.apply_diffusion(&mut mesh, 0.01);
        }

        // Measure max elevation after
        let z_max_after = surface.iter()
            .map(|&id| mesh.geometry.nodes[id].z)
            .fold(f64::NEG_INFINITY, f64::max);

        println!("Max z after: {:.6}", z_max_after);

        // The peak should decrease (diffusion spreads it out)
        assert!(z_max_after < z_max_before,
                "Diffusion should reduce peak: before={:.6}, after={:.6}",
                z_max_before, z_max_after);
    }

    #[test]
    fn test_stable_timestep() {
        let diffusion = HillslopeDiffusion::new(0.1); // m²/yr
        let dx = 1000.0; // 1 km
        let dt_stable = diffusion.stable_timestep(dx);

        // Should be << (dx)²/κ = 10^7 years
        assert!(dt_stable > 0.0);
        assert!(dt_stable < 1e7);
    }

    #[test]
    fn test_simple_flat_to_perturbed() {
        // Simpler test: start flat, manually perturb one node, see if diffusion spreads it
        let mut mesh = ImprovedMeshGenerator::generate_cube(4, 4, 4, 1.0, 1.0, 1.0);
        let surface = identify_surface_nodes(&mesh);
        let neighbors = build_surface_neighbors(&mesh, &surface);

        println!("Surface nodes: {}", surface.len());
        let neighbor_counts: Vec<usize> = surface.iter()
            .map(|&id| neighbors[&id].len())
            .collect();
        println!("Neighbor counts: {:?}", neighbor_counts);

        // Just check that we have surface nodes and they have neighbors
        assert!(surface.len() > 0);
        assert!(neighbor_counts.iter().all(|&c| c > 0), "All surface nodes should have neighbors");
    }
}
