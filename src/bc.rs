//! Boundary Condition Module
//!
//! This module handles all boundary condition logic including:
//! - Node identification on domain boundaries
//! - BC type definitions and application
//! - Velocity ramp-up for gradual spin-up

use crate::config::{BoundaryConditions, SimulationConfig};
use crate::fem::DofManager;
use crate::mesh::Mesh;
use nalgebra::Point3;

/// Boundary condition types
///
/// Defines the various types of boundary conditions that can be applied
/// to mesh boundaries. Each type has different physical meanings and
/// is applied differently to the DOF manager.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryType {
    /// Fixed velocity in specified component
    /// - Value: velocity magnitude (m/s)
    /// - Applied to: specific velocity component
    Velocity(f64),

    /// Plane strain condition (normal component fixed to zero, shear components free)
    /// - Applied to: normal velocity component
    /// - Common use: quasi-2D simulations
    PlaneStrain,

    /// Fixed normal component only
    /// - Applied to: normal velocity component = 0
    /// - Shear components: free
    FixedNormal,

    /// Free surface (no constraints)
    /// - No Dirichlet conditions applied
    /// - Commonly used for top surface in geodynamic models
    FreeSurface,

    /// Free slip (normal velocity = 0, tangential stress = 0)
    /// - Applied to: normal velocity component = 0
    /// - Future: Will require special treatment in weak form
    FreeSlip,
}

impl BoundaryType {
    /// Parse boundary type from string (from config file)
    ///
    /// # Arguments
    /// * `s` - String identifier from config
    /// * `value` - Optional velocity value for Velocity type
    ///
    /// # Returns
    /// Result with BoundaryType or error message
    pub fn from_str(s: &str, value: f64) -> Result<Self, String> {
        match s {
            "velocity" => Ok(BoundaryType::Velocity(value)),
            "plane_strain" => Ok(BoundaryType::PlaneStrain),
            "fixed_normal" => Ok(BoundaryType::FixedNormal),
            "free_surface" => Ok(BoundaryType::FreeSurface),
            "free_slip" => Ok(BoundaryType::FreeSlip),
            _ => Err(format!("Unknown boundary condition type: '{}'", s)),
        }
    }

    /// Get list of all valid BC type strings (for validation and GUI)
    pub fn valid_types() -> &'static [&'static str] {
        &["velocity", "plane_strain", "fixed_normal", "free_surface", "free_slip"]
    }
}

/// Boundary node sets for a cubic domain
///
/// Stores identified nodes on each face of the domain for efficient
/// BC application. Nodes are identified once and reused.
#[derive(Debug, Clone)]
pub struct BoundaryNodes {
    pub left: Vec<usize>,    // x = 0
    pub right: Vec<usize>,   // x = lx
    pub back: Vec<usize>,    // y = 0
    pub front: Vec<usize>,   // y = ly
    pub bottom: Vec<usize>,  // z = lz
    pub top: Vec<usize>,     // z = 0
}

impl BoundaryNodes {
    /// Identify all boundary nodes for a cubic domain
    ///
    /// # Arguments
    /// * `mesh` - Computational mesh
    /// * `lx, ly, lz` - Domain dimensions
    ///
    /// # Returns
    /// BoundaryNodes structure with all boundary node sets
    pub fn identify(mesh: &Mesh, lx: f64, ly: f64, lz: f64) -> Self {
        let tol = 1.0; // Tolerance for boundary identification

        BoundaryNodes {
            left: find_boundary_nodes(mesh, |p| p.x < tol),
            right: find_boundary_nodes(mesh, |p| (p.x - lx).abs() < tol),
            back: find_boundary_nodes(mesh, |p| p.y < tol),
            front: find_boundary_nodes(mesh, |p| (p.y - ly).abs() < tol),
            bottom: find_boundary_nodes(mesh, |p| (p.z - lz).abs() < tol),
            top: find_boundary_nodes(mesh, |p| p.z < tol),
        }
    }
}

/// Setup boundary conditions from configuration
///
/// Applies boundary conditions based on config file specification.
/// Supports velocity ramp-up for gradual spin-up.
///
/// # Arguments
/// * `config` - Simulation configuration
/// * `mesh` - Computational mesh
/// * `dof_mgr` - DOF manager (modified in place)
/// * `current_time_years` - Current simulation time for ramp-up
///
/// # BC Types Supported
/// - "velocity": Fixed velocity (with ramp-up)
/// - "plane_strain": Normal component fixed, shear free
/// - "fixed_normal": Normal velocity = 0
/// - "free_surface": No constraints
///
/// # Example
/// ```ignore
/// setup_boundary_conditions(&config, &mesh, &mut dof_mgr, 1000.0);
/// ```
pub fn setup_boundary_conditions(
    config: &SimulationConfig,
    mesh: &Mesh,
    dof_mgr: &mut DofManager,
    current_time_years: f64,
) {
    let (lx, ly, lz) = (config.domain.lx, config.domain.ly, config.domain.lz);

    // Compute current velocity (with ramp-up)
    let v_max = config.boundary_conditions.extension_rate_m_per_s;
    let ramp_duration = config.boundary_conditions.ramp_duration_years;
    let ramp_fraction = if current_time_years < ramp_duration {
        current_time_years / ramp_duration
    } else {
        1.0
    };
    let v_current = v_max * ramp_fraction;

    // Identify boundary nodes
    let nodes = BoundaryNodes::identify(mesh, lx, ly, lz);

    // X-boundaries: Extension (vbc_x = 1)
    if config.boundary_conditions.bc_x0 == "velocity" {
        apply_velocity_bc(dof_mgr, &nodes.left, 0, -v_current);
    }
    if config.boundary_conditions.bc_x1 == "velocity" {
        apply_velocity_bc(dof_mgr, &nodes.right, 0, v_current);
    }

    // Y-boundaries: Plane strain (vbc_y = 1)
    if config.boundary_conditions.bc_y0 == "plane_strain" {
        apply_velocity_bc(dof_mgr, &nodes.back, 1, 0.0);
    }
    if config.boundary_conditions.bc_y1 == "plane_strain" {
        apply_velocity_bc(dof_mgr, &nodes.front, 1, 0.0);
    }

    // Z-boundaries
    if config.boundary_conditions.bc_z1 == "fixed_normal" {
        apply_velocity_bc(dof_mgr, &nodes.bottom, 2, 0.0);
    }
    // Z-top is free surface (no BC)
}

/// Apply velocity boundary condition to a set of nodes
///
/// Helper function to apply Dirichlet BC for velocity component.
///
/// # Arguments
/// * `dof_mgr` - DOF manager
/// * `nodes` - List of node IDs
/// * `component` - Velocity component (0=x, 1=y, 2=z)
/// * `value` - Velocity value (m/s)
fn apply_velocity_bc(
    dof_mgr: &mut DofManager,
    nodes: &[usize],
    component: usize,
    value: f64,
) {
    for &node_id in nodes {
        dof_mgr.set_dirichlet(dof_mgr.velocity_dof(node_id, component), value);
    }
}

/// Find boundary nodes matching a predicate
///
/// Generic function for identifying nodes on boundaries based on
/// spatial criteria.
///
/// # Arguments
/// * `mesh` - Computational mesh
/// * `predicate` - Function that returns true for nodes on desired boundary
///
/// # Returns
/// Vector of node IDs satisfying the predicate
///
/// # Example
/// ```ignore
/// let left_nodes = find_boundary_nodes(mesh, |p| p.x < 1.0);
/// ```
pub fn find_boundary_nodes<F>(mesh: &Mesh, predicate: F) -> Vec<usize>
where
    F: Fn(&Point3<f64>) -> bool,
{
    (0..mesh.geometry.nodes.len())
        .filter(|&i| predicate(&mesh.geometry.nodes[i]))
        .collect()
}

/// Compute velocity ramp fraction
///
/// Returns a fraction [0, 1] for gradual velocity ramp-up during spin-up.
///
/// # Arguments
/// * `bc_config` - Boundary condition configuration
/// * `current_time_years` - Current simulation time (years)
///
/// # Returns
/// Ramp fraction (0 at t=0, 1 at t>=ramp_duration)
pub fn compute_ramp_fraction(bc_config: &BoundaryConditions, current_time_years: f64) -> f64 {
    if current_time_years < bc_config.ramp_duration_years {
        current_time_years / bc_config.ramp_duration_years
    } else {
        1.0
    }
}

/// Print BC summary
///
/// Displays boundary condition information for diagnostics.
pub fn print_bc_summary(_config: &SimulationConfig) {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Boundary Conditions (DES3D-style: vbc flag = 1)");
    println!("  Flag 1: Normal component fixed, shear components free");
    println!("═══════════════════════════════════════════════════════════════");
    println!("X-boundaries: Extension (pull apart)");
    println!("  vbc_x0 = 1: Left (x=0): vx = -v_ext (FIXED), vy & vz FREE");
    println!("  vbc_x1 = 1: Right (x=Lx): vx = +v_ext (FIXED), vy & vz FREE");
    println!("\nY-boundaries: Plane strain (quasi-2D)");
    println!("  vbc_y0 = 1: Back (y=0): vy = 0 (FIXED), vx & vz FREE");
    println!("  vbc_y1 = 1: Front (y=Ly): vy = 0 (FIXED), vx & vz FREE");
    println!("\nZ-boundaries:");
    println!("  Bottom (z=Lz): vz = 0 (FIXED), vx & vy FREE");
    println!("  Top (z=0): FREE SURFACE (no BCs)");
    println!("═══════════════════════════════════════════════════════════════\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_type_parsing() {
        // Valid types
        let vel = BoundaryType::from_str("velocity", 1e-9).unwrap();
        assert_eq!(vel, BoundaryType::Velocity(1e-9));

        let plane = BoundaryType::from_str("plane_strain", 0.0).unwrap();
        assert_eq!(plane, BoundaryType::PlaneStrain);

        let fixed = BoundaryType::from_str("fixed_normal", 0.0).unwrap();
        assert_eq!(fixed, BoundaryType::FixedNormal);

        let free = BoundaryType::from_str("free_surface", 0.0).unwrap();
        assert_eq!(free, BoundaryType::FreeSurface);

        // Invalid type
        let invalid = BoundaryType::from_str("invalid_type", 0.0);
        assert!(invalid.is_err());
    }

    #[test]
    fn test_valid_types_list() {
        let types = BoundaryType::valid_types();
        assert!(types.contains(&"velocity"));
        assert!(types.contains(&"plane_strain"));
        assert!(types.contains(&"free_surface"));
        assert_eq!(types.len(), 5);
    }

    #[test]
    fn test_ramp_fraction() {
        let bc_config = BoundaryConditions {
            extension_rate_cm_per_year: 3.15,
            extension_rate_m_per_s: 1e-9,
            ramp_duration_years: 50_000.0,
            bc_x0: "velocity".to_string(),
            bc_x1: "velocity".to_string(),
            bc_y0: "plane_strain".to_string(),
            bc_y1: "plane_strain".to_string(),
            bc_z0: "free_surface".to_string(),
            bc_z1: "fixed_normal".to_string(),
        };

        // At start
        assert_eq!(compute_ramp_fraction(&bc_config, 0.0), 0.0);

        // Halfway
        assert!((compute_ramp_fraction(&bc_config, 25_000.0) - 0.5).abs() < 1e-10);

        // After ramp
        assert_eq!(compute_ramp_fraction(&bc_config, 100_000.0), 1.0);
    }
}
