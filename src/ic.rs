//! Initial Conditions Module
//!
//! This module handles all initialization logic for geodynamic simulations:
//! - Tracer distribution and material ID assignment
//! - Velocity field initialization
//! - Pressure field initialization (lithostatic)
//! - Temperature field initialization (geotherm)
//! - Material property lookup

use crate::config::{InitialConditions, SimulationConfig};
use crate::fem::DofManager;
use crate::mesh::{Mesh, TracerSwarm};
use crate::utils::units::{mpa_to_pa, deg_to_rad};
use nalgebra::Vector3;

/// Material properties bundle
///
/// Container for all material properties needed for constitutive modeling.
/// Extracted from configuration and ready for use in simulation.
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

/// Setup initial tracer properties (material IDs and plastic strain)
///
/// Assigns material IDs to tracers based on depth layering and weak zone geometry.
/// Also initializes plastic strain for pre-damaged weak zones.
///
/// # Arguments
/// * `config` - Simulation configuration
/// * `_mesh` - Computational mesh (for future use)
/// * `swarm` - Tracer swarm (modified in place)
///
/// # Material ID Assignment
/// - 0: Upper crust (z < upper_crust.depth_bottom)
/// - 1: Lower crust (z >= upper_crust.depth_bottom)
/// - 2: Weak zone (if enabled and within geometry)
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
    let wz_dip_rad = deg_to_rad(config.materials.weak_zone.dip_angle);
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
///
/// Looks up material properties from configuration and converts to
/// SI units with proper type conversions.
///
/// # Arguments
/// * `config` - Simulation configuration
/// * `mat_id` - Material ID (0=upper crust, 1=lower crust, 2=weak zone)
///
/// # Returns
/// MaterialProps structure with all properties in SI units
pub fn get_material_properties(
    config: &SimulationConfig,
    mat_id: usize,
) -> MaterialProps {
    match mat_id {
        0 => MaterialProps {
            viscosity: config.materials.upper_crust.viscosity,
            density: config.materials.upper_crust.density,
            cohesion: mpa_to_pa(config.materials.upper_crust.cohesion_mpa),
            cohesion_min: mpa_to_pa(config.materials.upper_crust.cohesion_min_mpa),
            friction_angle: deg_to_rad(config.materials.upper_crust.friction_angle),
            shear_modulus: config.materials.upper_crust.shear_modulus_pa,
            min_viscosity: config.materials.upper_crust.min_viscosity,
            max_viscosity: config.materials.upper_crust.max_viscosity,
        },
        1 => MaterialProps {
            viscosity: config.materials.lower_crust.viscosity,
            density: config.materials.lower_crust.density,
            cohesion: mpa_to_pa(config.materials.lower_crust.cohesion_mpa),
            cohesion_min: mpa_to_pa(config.materials.lower_crust.cohesion_min_mpa),
            friction_angle: deg_to_rad(config.materials.lower_crust.friction_angle),
            shear_modulus: config.materials.lower_crust.shear_modulus_pa,
            min_viscosity: config.materials.lower_crust.min_viscosity,
            max_viscosity: config.materials.lower_crust.max_viscosity,
        },
        2 => MaterialProps {
            viscosity: config.materials.weak_zone.viscosity,
            density: config.materials.lower_crust.density, // Use lower crust density
            cohesion: mpa_to_pa(config.materials.weak_zone.cohesion_mpa),
            cohesion_min: mpa_to_pa(config.materials.weak_zone.cohesion_min_mpa),
            friction_angle: deg_to_rad(config.materials.lower_crust.friction_angle),
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

/// Initialize lithostatic pressure field
///
/// Computes initial pressure based on depth and gravity:
/// P(z) = ρ * g * depth
///
/// where depth = (domain_height - z)
///
/// # Arguments
/// * `config` - Initial conditions configuration
/// * `mesh` - Computational mesh
/// * `density` - Average density (kg/m³)
/// * `domain_height` - Total domain height (m)
///
/// # Returns
/// Vector of element pressures (Pa)
///
/// # Example
/// ```ignore
/// let pressures = initialize_lithostatic_pressure(
///     &config.initial_conditions,
///     &mesh,
///     2700.0,  // crustal density
///     10_000.0 // 10 km domain
/// );
/// ```
pub fn initialize_lithostatic_pressure(
    config: &InitialConditions,
    mesh: &Mesh,
    density: f64,
    domain_height: f64,
) -> Vec<f64> {
    if !config.use_lithostatic_pressure {
        // Return zero pressure if not enabled
        return vec![0.0; mesh.num_elements()];
    }

    let g = config.gravity;

    mesh.connectivity.tet10_elements
        .iter()
        .map(|elem| {
            // Compute element center depth
            let z_avg: f64 = elem.nodes.iter()
                .map(|&n| mesh.geometry.nodes[n].z)
                .sum::<f64>() / elem.nodes.len() as f64;

            let depth = (domain_height - z_avg).max(0.0);
            density * g * depth
        })
        .collect()
}

/// Initialize extension velocity field
///
/// Creates a linear velocity profile for extension scenarios:
/// v_x(x) = v_ext * (2 * x/lx - 1)
///
/// This gives:
/// - v_x = -v_ext at x = 0 (left boundary)
/// - v_x = 0 at x = lx/2 (center)
/// - v_x = +v_ext at x = lx (right boundary)
///
/// Provides a better initial guess for nonlinear solvers than zero velocity.
///
/// # Arguments
/// * `mesh` - Computational mesh
/// * `dof_mgr` - DOF manager
/// * `extension_rate` - Extension velocity (m/s)
/// * `domain_width` - Domain width in x-direction (m)
///
/// # Returns
/// Initial solution vector with velocity field
///
/// # Example
/// ```ignore
/// let initial_sol = initialize_extension_velocity(
///     &mesh,
///     &dof_mgr,
///     1e-9,      // 1 cm/yr extension
///     100_000.0  // 100 km domain
/// );
/// ```
pub fn initialize_extension_velocity(
    mesh: &Mesh,
    dof_mgr: &DofManager,
    extension_rate: f64,
    domain_width: f64,
) -> Vec<f64> {
    let mut sol = vec![0.0; dof_mgr.total_dofs()];

    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        let x_normalized = node.x / domain_width;
        // Linear profile: -v_ext at left, +v_ext at right
        sol[dof_mgr.velocity_dof(node_id, 0)] = extension_rate * (2.0 * x_normalized - 1.0);
        // vy and vz remain zero
    }

    sol
}

/// Initialize geotherm (temperature field)
///
/// Computes initial temperature distribution based on linear geotherm:
/// T(z) = T_surface + (T_moho - T_surface) * z / moho_depth
///
/// # Arguments
/// * `config` - Initial conditions configuration
/// * `mesh` - Computational mesh
/// * `moho_depth` - Depth to Moho discontinuity (m)
///
/// # Returns
/// Vector of nodal temperatures (K)
///
/// # Notes
/// - Returns uniform surface temperature if `use_geotherm` is false
/// - Assumes linear temperature gradient from surface to Moho
/// - Can be extended for more complex geotherms (exponential, piecewise, etc.)
///
/// # Example
/// ```ignore
/// let temperatures = initialize_geotherm(
///     &config.initial_conditions,
///     &mesh,
///     30_000.0  // 30 km to Moho
/// );
/// ```
pub fn initialize_geotherm(
    config: &InitialConditions,
    mesh: &Mesh,
    moho_depth: f64,
) -> Vec<f64> {
    if !config.use_geotherm {
        // Return uniform surface temperature if geotherm disabled
        return vec![config.surface_temp_kelvin; mesh.num_nodes()];
    }

    let t_surface = config.surface_temp_kelvin;
    let t_moho = config.moho_temp_kelvin;

    mesh.geometry.nodes
        .iter()
        .map(|node| {
            let depth = node.z;
            let depth_fraction = (depth / moho_depth).min(1.0); // Clamp to [0, 1]

            // Linear interpolation
            t_surface + (t_moho - t_surface) * depth_fraction
        })
        .collect()
}

/// Initialize uniform gravity vector
///
/// Helper function to create gravity vector pointing downward (negative z).
///
/// # Arguments
/// * `config` - Initial conditions configuration
///
/// # Returns
/// Gravity vector (m/s²)
pub fn get_gravity_vector(config: &InitialConditions) -> Vector3<f64> {
    Vector3::new(0.0, 0.0, -config.gravity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImprovedMeshGenerator, DofManager};

    #[test]
    fn test_material_props_unit_conversion() {
        // Create minimal config for testing
        let config_str = r#"
            [domain]
            lx = 100000.0
            ly = 10000.0
            lz = 10000.0
            dx = 5000.0
            dy = 5000.0
            dz = 2500.0

            [simulation]
            total_time_years = 1e6
            output_interval_years = 1e5
            quality_check_interval_years = 1e4

            [boundary_conditions]
            extension_rate_cm_per_year = 3.15
            extension_rate_m_per_s = 1e-9
            ramp_duration_years = 5e4
            bc_x0 = "velocity"
            bc_x1 = "velocity"
            bc_y0 = "plane_strain"
            bc_y1 = "plane_strain"
            bc_z0 = "free_surface"
            bc_z1 = "fixed_normal"

            [initial_conditions]
            surface_temp_kelvin = 273.0
            moho_temp_kelvin = 873.0
            use_geotherm = true
            use_lithostatic_pressure = true
            gravity = 9.81

            [materials.upper_crust]
            depth_top = 0.0
            depth_bottom = 30000.0
            density = 2700.0
            viscosity = 1e21
            cohesion_mpa = 44.0
            cohesion_min_mpa = 4.0
            friction_angle = 30.0
            shear_modulus_pa = 3e10
            min_viscosity = 1e18
            max_viscosity = 1e24

            [materials.lower_crust]
            depth_top = 30000.0
            depth_bottom = 30000.0
            density = 2700.0
            viscosity = 1e21
            cohesion_mpa = 44.0
            cohesion_min_mpa = 4.0
            friction_angle = 30.0
            shear_modulus_pa = 3e10
            min_viscosity = 1e18
            max_viscosity = 1e24

            [materials.weak_zone]
            enabled = true
            geometry = "dipping_plane"
            x_center = 50000.0
            y_center = 5000.0
            depth_min = 5000.0
            depth_max = 10000.0
            dip_angle = -60.0
            azimuth = 15.0
            half_width = 1200.0
            viscosity = 1e21
            cohesion_mpa = 44.0
            cohesion_min_mpa = 4.0
            friction_angle = 30.0
            initial_plastic_strain = 0.5

            [time_stepping]
            use_adaptive = true
            dt_min_years = 100.0
            dt_max_years = 3000.0
            dt_initial_years = 500.0
            cfl_target = 0.3
            use_cfl_constraint = true
            use_maxwell_constraint = false
            use_mesh_quality_constraint = true

            [solver]
            nonlinear_solver = "Picard"
            max_nonlinear_iterations = 50
            nonlinear_tolerance = 1e-3
            nonlinear_abs_tolerance = 1e-6
            linear_solver = "GMRES"
            gmres_restart = 200
            gmres_max_iterations = 8000
            gmres_tolerance = 5e-3
            gmres_abs_tolerance = 1e-8
            use_picard_warmstart = false
            picard_warmstart_iterations = 0
            use_nondimensionalization = true

            [tracers]
            tracers_per_element = 4
            total_tracers = 1920

            [output]
            format = "VTK"
            output_dir = "./output/test"
            save_velocity = true
            save_pressure = true
            save_stress = true
            save_strain = true
            save_material_id = true
            save_tracers = true

            [physics]
            gravity_enabled = true
            surface_processes_enabled = false
            thermal_diffusion_enabled = false
            winkler_foundation_enabled = false
            surface_diffusion_kappa = 1e-7
        "#;

        let config: SimulationConfig = toml::from_str(config_str).unwrap();
        let props = get_material_properties(&config, 0);

        // Check unit conversions
        assert_eq!(props.cohesion, 44.0e6); // MPa to Pa
        assert_eq!(props.cohesion_min, 4.0e6);
        assert!((props.friction_angle - 30.0_f64.to_radians()).abs() < 1e-10); // deg to rad
    }

    #[test]
    fn test_lithostatic_pressure() {
        let config = InitialConditions {
            surface_temp_kelvin: 273.0,
            moho_temp_kelvin: 873.0,
            use_geotherm: true,
            use_lithostatic_pressure: true,
            gravity: 9.81,
        };

        let mesh = ImprovedMeshGenerator::generate_cube(2, 2, 2, 10.0, 10.0, 10.0);
        let density = 2700.0;
        let domain_height = 10.0;

        let pressures = initialize_lithostatic_pressure(&config, &mesh, density, domain_height);

        // Check that pressure increases with depth
        assert_eq!(pressures.len(), mesh.num_elements());
        assert!(pressures.iter().all(|&p| p >= 0.0));

        // Pressure at top should be lower than at bottom
        // (This is a rough check; exact values depend on element centers)
        let max_p = pressures.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_p = pressures.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(max_p > min_p);
    }

    #[test]
    fn test_extension_velocity_profile() {
        let mesh = ImprovedMeshGenerator::generate_cube(4, 2, 2, 100.0, 10.0, 10.0);
        let dof_mgr = DofManager::new_mixed(mesh.num_nodes(), &mesh.connectivity.corner_nodes());
        let extension_rate = 1e-9;
        let domain_width = 100.0;

        let sol = initialize_extension_velocity(&mesh, &dof_mgr, extension_rate, domain_width);

        // Find nodes at left, center, and right
        let left_node = mesh.geometry.nodes.iter()
            .position(|n| n.x < 1.0)
            .unwrap();
        let right_node = mesh.geometry.nodes.iter()
            .position(|n| (n.x - domain_width).abs() < 1.0)
            .unwrap();

        // Check velocity profile
        let vx_left = sol[dof_mgr.velocity_dof(left_node, 0)];
        let vx_right = sol[dof_mgr.velocity_dof(right_node, 0)];

        assert!((vx_left - (-extension_rate)).abs() < 1e-12);
        assert!((vx_right - extension_rate).abs() < 1e-12);
    }

    #[test]
    fn test_geotherm() {
        let config = InitialConditions {
            surface_temp_kelvin: 273.0,
            moho_temp_kelvin: 873.0,
            use_geotherm: true,
            use_lithostatic_pressure: false,
            gravity: 9.81,
        };

        let mesh = ImprovedMeshGenerator::generate_cube(2, 2, 2, 10.0, 10.0, 30_000.0);
        let moho_depth = 30_000.0;

        let temps = initialize_geotherm(&config, &mesh, moho_depth);

        // Check temperature range
        assert_eq!(temps.len(), mesh.num_nodes());
        let min_t = temps.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_t = temps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        assert!(min_t >= config.surface_temp_kelvin);
        assert!(max_t <= config.moho_temp_kelvin);
    }

    #[test]
    fn test_gravity_vector() {
        let config = InitialConditions {
            surface_temp_kelvin: 273.0,
            moho_temp_kelvin: 873.0,
            use_geotherm: false,
            use_lithostatic_pressure: false,
            gravity: 9.81,
        };

        let g_vec = get_gravity_vector(&config);
        assert_eq!(g_vec.x, 0.0);
        assert_eq!(g_vec.y, 0.0);
        assert_eq!(g_vec.z, -9.81);
    }
}
