//! Configuration management for geodynamic simulations
//!
//! Reads TOML configuration files and provides structured data for setting up
//! initial conditions, boundary conditions, material properties, and solver parameters.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Main simulation configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SimulationConfig {
    pub domain: DomainConfig,
    pub simulation: SimulationParams,
    pub boundary_conditions: BoundaryConditions,
    pub initial_conditions: InitialConditions,
    pub materials: MaterialsConfig,
    pub time_stepping: TimeSteppingConfig,
    pub solver: SolverConfig,
    pub tracers: TracerConfig,
    pub output: OutputConfig,
    pub physics: PhysicsConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DomainConfig {
    /// Domain length in x (m)
    pub lx: f64,
    /// Domain length in y (m)
    pub ly: f64,
    /// Domain length in z (m)
    pub lz: f64,
    /// Target mesh spacing in x (m)
    pub dx: f64,
    /// Target mesh spacing in y (m)
    pub dy: f64,
    /// Target mesh spacing in z (m)
    pub dz: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SimulationParams {
    /// Total simulation time (years)
    pub total_time_years: f64,
    /// Output interval (years)
    pub output_interval_years: f64,
    /// Mesh quality check interval (years)
    pub quality_check_interval_years: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BoundaryConditions {
    /// Extension rate (cm/yr)
    pub extension_rate_cm_per_year: f64,
    /// Extension rate (m/s)
    pub extension_rate_m_per_s: f64,
    /// Velocity ramp-up duration (years)
    pub ramp_duration_years: f64,
    /// Boundary condition types
    pub bc_x0: String,
    pub bc_x1: String,
    pub bc_y0: String,
    pub bc_y1: String,
    pub bc_z0: String,
    pub bc_z1: String,

    /// Explicit boundary values (optional override)
    pub bc_val_x0: Option<f64>,
    pub bc_val_x1: Option<f64>,
    pub bc_val_y0: Option<f64>,
    pub bc_val_y1: Option<f64>,
    pub bc_val_z0: Option<f64>,
    pub bc_val_z1: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InitialConditions {
    pub surface_temp_kelvin: f64,
    pub moho_temp_kelvin: f64,
    pub use_geotherm: bool,
    pub use_lithostatic_pressure: bool,
    pub gravity: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MaterialsConfig {
    pub upper_crust: MaterialProperties,
    pub lower_crust: MaterialProperties,
    pub weak_zone: WeakZoneConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MaterialProperties {
    pub depth_top: f64,
    pub depth_bottom: f64,
    pub density: f64,
    pub viscosity: f64,
    pub cohesion_mpa: f64,
    pub cohesion_min_mpa: f64,
    pub friction_angle: f64,
    pub shear_modulus_pa: f64,
    #[serde(default = "default_min_viscosity")]
    pub min_viscosity: f64,  // Minimum viscosity cap (Pa·s) - DES3D style
    #[serde(default = "default_max_viscosity")]
    pub max_viscosity: f64,  // Maximum viscosity cap (Pa·s) - for stability
}

fn default_min_viscosity() -> f64 { 1e18 }
fn default_max_viscosity() -> f64 { 1e30 }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WeakZoneConfig {
    pub enabled: bool,
    pub geometry: String,
    pub x_center: f64,
    pub y_center: f64,
    pub depth_min: f64,
    pub depth_max: f64,
    pub dip_angle: f64,
    pub azimuth: f64,
    pub half_width: f64,
    pub viscosity: f64,
    pub cohesion_mpa: f64,
    pub cohesion_min_mpa: f64,
    pub friction_angle: f64,  // Friction angle in degrees (lower for weak detachments)
    #[serde(default = "default_min_viscosity")]
    pub min_viscosity: f64,  // Minimum viscosity cap (Pa·s) - DES3D style
    #[serde(default = "default_max_viscosity")]
    pub max_viscosity: f64,  // Maximum viscosity cap (Pa·s) - for stability
    #[serde(default)]
    pub initial_plastic_strain: f64,  // Pre-damage (0.0-1.0)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TimeSteppingConfig {
    pub use_adaptive: bool,
    pub dt_min_years: f64,
    pub dt_max_years: f64,
    pub dt_initial_years: f64,
    pub cfl_target: f64,
    pub use_cfl_constraint: bool,
    pub use_maxwell_constraint: bool,
    pub use_mesh_quality_constraint: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SolverConfig {
    pub nonlinear_solver: String,
    pub max_nonlinear_iterations: usize,
    pub nonlinear_tolerance: f64,
    pub nonlinear_abs_tolerance: f64,
    pub linear_solver: String,
    pub gmres_restart: usize,
    pub gmres_max_iterations: usize,
    pub gmres_tolerance: f64,
    pub gmres_abs_tolerance: f64,
    pub use_picard_warmstart: bool,
    pub picard_warmstart_iterations: usize,
    pub use_nondimensionalization: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TracerConfig {
    pub tracers_per_element: usize,
    pub total_tracers: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutputConfig {
    pub format: String,
    pub output_dir: String,
    pub save_velocity: bool,
    pub save_pressure: bool,
    pub save_stress: bool,
    pub save_strain: bool,
    pub save_material_id: bool,
    pub save_tracers: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PhysicsConfig {
    pub gravity_enabled: bool,
    pub surface_processes_enabled: bool,
    pub thermal_diffusion_enabled: bool,
    pub winkler_foundation_enabled: bool,
    pub surface_diffusion_kappa: f64,
}

impl SimulationConfig {
    /// Load configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let contents = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;

        let config: SimulationConfig = toml::from_str(&contents)
            .map_err(|e| format!("Failed to parse config file: {}", e))?;

        Ok(config)
    }

    /// Get grid dimensions from domain config
    pub fn grid_dimensions(&self) -> (usize, usize, usize) {
        let nx = (self.domain.lx / self.domain.dx).ceil() as usize;
        let ny = (self.domain.ly / self.domain.dy).ceil() as usize;
        let nz = (self.domain.lz / self.domain.dz).ceil() as usize;
        (nx, ny, nz)
    }

    /// Print configuration summary
    pub fn print_summary(&self) {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  Simulation Configuration");
        println!("═══════════════════════════════════════════════════════════════");
        println!("Domain:");
        println!("  Size: {:.1} × {:.1} × {:.1} km",
            self.domain.lx/1e3, self.domain.ly/1e3, self.domain.lz/1e3);
        println!("  Resolution: {:.1} × {:.1} × {:.1} km",
            self.domain.dx/1e3, self.domain.dy/1e3, self.domain.dz/1e3);

        let (nx, ny, nz) = self.grid_dimensions();
        println!("  Grid: {} × {} × {} cells ({} elements)",
            nx, ny, nz, nx * ny * nz);

        println!("\nSimulation:");
        println!("  Duration: {:.1} Myr", self.simulation.total_time_years / 1e6);
        println!("  Output every: {:.1} kyr", self.simulation.output_interval_years / 1e3);

        println!("\nBoundary Conditions:");
        println!("  Extension rate: {:.2} cm/yr", self.boundary_conditions.extension_rate_cm_per_year);
        println!("  Ramp duration: {:.1} kyr", self.boundary_conditions.ramp_duration_years / 1e3);

        println!("\nMaterials:");
        println!("  Upper crust: μ = {:.1e} Pa·s, c = {:.1} MPa",
            self.materials.upper_crust.viscosity,
            self.materials.upper_crust.cohesion_mpa);
        println!("  Lower crust: μ = {:.1e} Pa·s, c = {:.1} MPa",
            self.materials.lower_crust.viscosity,
            self.materials.lower_crust.cohesion_mpa);
        if self.materials.weak_zone.enabled {
            println!("  Weak zone: μ = {:.1e} Pa·s, dip = {:.1}°",
                self.materials.weak_zone.viscosity,
                self.materials.weak_zone.dip_angle);
        }

        println!("\nSolver:");
        println!("  Nonlinear: {}", self.solver.nonlinear_solver);
        println!("  Linear: {} (restart={}, max_iter={})",
            self.solver.linear_solver,
            self.solver.gmres_restart,
            self.solver.gmres_max_iterations);

        println!("═══════════════════════════════════════════════════════════════\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_config() {
        // This test would load an actual config file
        // For now, just test the structure compiles
        assert_eq!(1, 1);
    }
}
