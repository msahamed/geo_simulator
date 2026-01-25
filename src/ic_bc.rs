//! Initial and Boundary Condition Setup
//!
//! Provides reusable functions for setting up initial conditions (IC) and
//! boundary conditions (BC) from configuration files.

use crate::config::SimulationConfig;
use crate::fem::DofManager;
use crate::mesh::{Mesh, TracerSwarm};
use nalgebra::Point3;

/// Setup boundary conditions from configuration
///
/// Applies DES3D-style boundary conditions:
/// - vbc flag = 1: normal component fixed, shear components free
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
    let left_nodes = find_boundary_nodes(mesh, |p| p.x < 1.0);
    let right_nodes = find_boundary_nodes(mesh, |p| (p.x - lx).abs() < 1.0);
    let back_nodes = find_boundary_nodes(mesh, |p| p.y < 1.0);
    let front_nodes = find_boundary_nodes(mesh, |p| (p.y - ly).abs() < 1.0);
    let bottom_nodes = find_boundary_nodes(mesh, |p| (p.z - lz).abs() < 1.0);

    // X-boundaries: Extension (vbc_x = 1)
    if config.boundary_conditions.bc_x0 == "velocity" {
        for &node_id in &left_nodes {
            dof_mgr.set_dirichlet(dof_mgr.velocity_dof(node_id, 0), -v_current);
        }
    }
    if config.boundary_conditions.bc_x1 == "velocity" {
        for &node_id in &right_nodes {
            dof_mgr.set_dirichlet(dof_mgr.velocity_dof(node_id, 0), v_current);
        }
    }

    // Y-boundaries: Plane strain (vbc_y = 1)
    if config.boundary_conditions.bc_y0 == "plane_strain" {
        for &node_id in &back_nodes {
            dof_mgr.set_dirichlet(dof_mgr.velocity_dof(node_id, 1), 0.0);
        }
    }
    if config.boundary_conditions.bc_y1 == "plane_strain" {
        for &node_id in &front_nodes {
            dof_mgr.set_dirichlet(dof_mgr.velocity_dof(node_id, 1), 0.0);
        }
    }

    // Z-boundaries
    if config.boundary_conditions.bc_z1 == "fixed_normal" {
        for &node_id in &bottom_nodes {
            dof_mgr.set_dirichlet(dof_mgr.velocity_dof(node_id, 2), 0.0);
        }
    }
    // Z-top is free surface (no BC)
}

/// Find boundary nodes matching a predicate
fn find_boundary_nodes<F>(mesh: &Mesh, predicate: F) -> Vec<usize>
where
    F: Fn(&Point3<f64>) -> bool,
{
    (0..mesh.geometry.nodes.len())
        .filter(|&i| predicate(&mesh.geometry.nodes[i]))
        .collect()
}

/// Setup initial tracer properties (material IDs and plastic strain)
pub fn setup_initial_tracers(
    config: &SimulationConfig,
    _mesh: &Mesh,
    swarm: &mut TracerSwarm,
) {
    let _lx = config.domain.lx;
    let _ly = config.domain.ly;
    let _lz = config.domain.lz;

    // Material layer depths
    let upper_crust_depth = config.materials.upper_crust.depth_bottom;

    // Weak zone parameters (if enabled)
    let weak_zone_enabled = config.materials.weak_zone.enabled;
    let wz_x_center = config.materials.weak_zone.x_center;
    let wz_y_center = config.materials.weak_zone.y_center;
    let wz_depth_min = config.materials.weak_zone.depth_min;
    let wz_depth_max = config.materials.weak_zone.depth_max;
    let wz_dip_rad = config.materials.weak_zone.dip_angle.to_radians();
    let wz_half_width = config.materials.weak_zone.half_width;
    let wz_plastic_strain = config.materials.weak_zone.initial_plastic_strain;

    // Weak zone plane normal (simplified: dip in x-z plane)
    let nx = -wz_dip_rad.sin();
    let ny = 0.0;
    let nz = wz_dip_rad.cos();

    // Assign material IDs to tracers
    for i in 0..swarm.num_tracers() {
        let x = swarm.x[i];
        let y = swarm.y[i];
        let z = swarm.z[i];

        // Default: assign based on depth
        let depth = z;
        let mut mat_id = if depth < upper_crust_depth {
            0 // Upper crust
        } else {
            1 // Lower crust
        };

        // Check if tracer is in weak zone
        if weak_zone_enabled {
            let dx = x - wz_x_center;
            let dy = y - wz_y_center;
            let dz = z;

            // Distance to weak zone plane
            let dist = (dx * nx + dy * ny + dz * nz).abs();

            // Check if within weak zone bounds
            let in_depth_range = z >= wz_depth_min && z <= wz_depth_max;
            let in_width = dist < wz_half_width;

            if in_depth_range && in_width {
                mat_id = 2; // Weak zone
                swarm.plastic_strain[i] = wz_plastic_strain;
            }
        }

        swarm.material_id[i] = mat_id;
    }

    println!("  Tracers initialized:");
    let n_upper = swarm.material_id.iter().filter(|&&id| id == 0).count();
    let n_lower = swarm.material_id.iter().filter(|&&id| id == 1).count();
    let n_weak = swarm.material_id.iter().filter(|&&id| id == 2).count();
    println!("    Upper crust: {}", n_upper);
    println!("    Lower crust: {}", n_lower);
    println!("    Weak zone: {}", n_weak);
}

/// Get material properties for a given material ID
pub fn get_material_properties(
    config: &SimulationConfig,
    mat_id: usize,
) -> MaterialProps {
    match mat_id {
        0 => MaterialProps {
            viscosity: config.materials.upper_crust.viscosity,
            density: config.materials.upper_crust.density,
            cohesion: config.materials.upper_crust.cohesion_mpa * 1e6,
            cohesion_min: config.materials.upper_crust.cohesion_min_mpa * 1e6,
            friction_angle: config.materials.upper_crust.friction_angle.to_radians(),
            shear_modulus: config.materials.upper_crust.shear_modulus_pa,
            min_viscosity: config.materials.upper_crust.min_viscosity,
            max_viscosity: config.materials.upper_crust.max_viscosity,
        },
        1 => MaterialProps {
            viscosity: config.materials.lower_crust.viscosity,
            density: config.materials.lower_crust.density,
            cohesion: config.materials.lower_crust.cohesion_mpa * 1e6,
            cohesion_min: config.materials.lower_crust.cohesion_min_mpa * 1e6,
            friction_angle: config.materials.lower_crust.friction_angle.to_radians(),
            shear_modulus: config.materials.lower_crust.shear_modulus_pa,
            min_viscosity: config.materials.lower_crust.min_viscosity,
            max_viscosity: config.materials.lower_crust.max_viscosity,
        },
        2 => MaterialProps {
            viscosity: config.materials.weak_zone.viscosity,
            density: config.materials.lower_crust.density, // Use lower crust density
            cohesion: config.materials.weak_zone.cohesion_mpa * 1e6,
            cohesion_min: config.materials.weak_zone.cohesion_min_mpa * 1e6,
            friction_angle: config.materials.lower_crust.friction_angle.to_radians(),
            shear_modulus: config.materials.lower_crust.shear_modulus_pa,
            min_viscosity: config.materials.weak_zone.min_viscosity,
            max_viscosity: config.materials.weak_zone.max_viscosity,
        },
        _ => {
            eprintln!("WARNING: Unknown material ID {}, using lower crust", mat_id);
            get_material_properties(config, 1)
        }
    }
}

/// Material properties bundle
#[derive(Debug, Clone, Copy)]
pub struct MaterialProps {
    pub viscosity: f64,
    pub density: f64,
    pub cohesion: f64,
    pub cohesion_min: f64,
    pub friction_angle: f64,
    pub shear_modulus: f64,
    pub min_viscosity: f64,
    pub max_viscosity: f64,
}

/// Print BC summary
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
