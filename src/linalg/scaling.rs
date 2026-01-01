/// Non-dimensionalization for geodynamic simulations
///
/// **Problem:** Geodynamic systems have extreme parameter ranges:
/// - Viscosity: 1e18 - 1e25 Pa·s
/// - Domain size: 1e3 - 1e6 m
/// - Velocities: 1e-11 - 1e-8 m/s (mm/yr to cm/yr)
/// - Forces: 1e10 - 1e15 N
///
/// This causes:
/// - Ill-conditioned matrices (condition numbers > 1e20)
/// - Numerical overflow/underflow
/// - Solvers struggle to converge
///
/// **Solution:** Non-dimensionalization
/// Work with O(1) dimensionless variables, then convert back for physics.
///
/// # References
/// - Moresi & Solomatov (1998): "Numerical investigation of 2D convection"
/// - Schubert et al. (2001): "Mantle Convection in the Earth and Planets"
/// - ASPECT manual: "Nondimensionalization" section

use crate::fem::DofManager;

/// Characteristic scales for non-dimensionalization
///
/// All physical quantities are scaled by characteristic values:
/// - Length: L* (domain size)
/// - Velocity: v* (characteristic velocity, e.g., plate speed)
/// - Viscosity: μ* (reference viscosity)
///
/// Derived scales:
/// - Time: t* = L*/v*
/// - Stress/Pressure: τ* = μ*v*/L*
/// - Force: F* = τ*L*²
/// - Strain rate: ė* = v*/L*
#[derive(Debug, Clone, Copy)]
pub struct CharacteristicScales {
    /// Length scale L* [m] (typically domain width)
    pub length: f64,

    /// Velocity scale v* [m/s] (typically plate velocity)
    pub velocity: f64,

    /// Viscosity scale μ* [Pa·s] (reference viscosity)
    pub viscosity: f64,

    /// Density scale ρ* [kg/m³] (reference density)
    pub density: f64,

    /// Gravity magnitude g [m/s²] (acceleration, dimensional)
    pub gravity: f64,

    // Derived scales (computed from above)
    /// Time scale t* = L*/v* [s]
    pub time: f64,

    /// Stress scale τ* = μ*v*/L* [Pa]
    pub stress: f64,

    /// Force scale F* = τ*L*² [N]
    pub force: f64,

    /// Strain rate scale ė* = v*/L* [1/s]
    pub strain_rate: f64,
}

impl CharacteristicScales {
    /// Create characteristic scales from fundamental parameters
    ///
    /// # Arguments
    /// * `length` - Characteristic length L* [m] (e.g., 100 km)
    /// * `velocity` - Characteristic velocity v* [m/s] (e.g., 1 cm/yr)
    /// * `viscosity` - Reference viscosity μ* [Pa·s] (e.g., 1e21)
    /// * `density` - Reference density ρ* [kg/m³] (e.g., 3000)
    /// * `gravity` - Gravity magnitude g [m/s²] (e.g., 9.81)
    ///
    /// # Example
    /// ```
    /// let scales = CharacteristicScales::new(
    ///     100_000.0,                                 // 100 km
    ///     0.01 / (365.25 * 24.0 * 3600.0),          // 1 cm/yr
    ///     1e21,                                      // Pa·s
    ///     3000.0,                                    // kg/m³
    ///     9.81,                                      // m/s²
    /// );
    /// ```
    pub fn new(length: f64, velocity: f64, viscosity: f64, density: f64, gravity: f64) -> Self {
        assert!(length > 0.0, "Length scale must be positive");
        assert!(velocity > 0.0, "Velocity scale must be positive");
        assert!(viscosity > 0.0, "Viscosity scale must be positive");
        assert!(density > 0.0, "Density scale must be positive");
        assert!(gravity > 0.0, "Gravity must be positive");

        // Compute derived scales
        let time = length / velocity;
        let stress = viscosity * velocity / length;
        let force = stress * length * length;
        let strain_rate = velocity / length;

        Self {
            length,
            velocity,
            viscosity,
            density,
            gravity,
            time,
            stress,
            force,
            strain_rate,
        }
    }

    /// Print scaling information for diagnostics
    pub fn print_summary(&self) {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  Non-dimensionalization Scales");
        println!("═══════════════════════════════════════════════════════════════");
        println!("Fundamental Scales:");
        println!("  Length (L*):      {:.2e} m ({:.1} km)", self.length, self.length / 1e3);
        println!("  Velocity (v*):    {:.2e} m/s ({:.2} cm/yr)",
                 self.velocity, self.velocity * 365.25 * 24.0 * 3600.0 * 100.0);
        println!("  Viscosity (μ*):   {:.2e} Pa·s", self.viscosity);
        println!("  Density (ρ*):     {:.2e} kg/m³", self.density);
        println!("  Gravity (g):      {:.2e} m/s²", self.gravity);
        println!("\nDerived Scales:");
        println!("  Time (t*):        {:.2e} s ({:.2} Myr)",
                 self.time, self.time / (365.25 * 24.0 * 3600.0 * 1e6));
        println!("  Stress (τ*):      {:.2e} Pa ({:.2} MPa)", self.stress, self.stress / 1e6);
        println!("  Force (F*):       {:.2e} N", self.force);
        println!("  Strain rate (ė*): {:.2e} 1/s", self.strain_rate);
        println!("═══════════════════════════════════════════════════════════════\n");
    }

    // ========================================================================
    // Nondimensionalization (Physical → Dimensionless)
    // ========================================================================

    /// Non-dimensionalize length: x̂ = x / L*
    pub fn nondim_length(&self, x_phys: f64) -> f64 {
        x_phys / self.length
    }

    /// Non-dimensionalize velocity: v̂ = v / v*
    pub fn nondim_velocity(&self, v_phys: f64) -> f64 {
        v_phys / self.velocity
    }

    /// Non-dimensionalize pressure/stress: p̂ = p / τ*
    pub fn nondim_stress(&self, p_phys: f64) -> f64 {
        p_phys / self.stress
    }

    /// Non-dimensionalize force: F̂ = F / F*
    pub fn nondim_force(&self, f_phys: f64) -> f64 {
        f_phys / self.force
    }

    /// Non-dimensionalize time: t̂ = t / t*
    pub fn nondim_time(&self, t_phys: f64) -> f64 {
        t_phys / self.time
    }

    /// Non-dimensionalize viscosity: μ̂ = μ / μ*
    pub fn nondim_viscosity(&self, mu_phys: f64) -> f64 {
        mu_phys / self.viscosity
    }

    /// Non-dimensionalize density: ρ̂ = ρ / ρ*
    pub fn nondim_density(&self, rho_phys: f64) -> f64 {
        rho_phys / self.density
    }

    // ========================================================================
    // Dimensionalization (Dimensionless → Physical)
    // ========================================================================

    /// Dimensionalize length: x = x̂ * L*
    pub fn dim_length(&self, x_nondim: f64) -> f64 {
        x_nondim * self.length
    }

    /// Dimensionalize velocity: v = v̂ * v*
    pub fn dim_velocity(&self, v_nondim: f64) -> f64 {
        v_nondim * self.velocity
    }

    /// Dimensionalize pressure/stress: p = p̂ * τ*
    pub fn dim_stress(&self, p_nondim: f64) -> f64 {
        p_nondim * self.stress
    }

    /// Dimensionalize force: F = F̂ * F*
    pub fn dim_force(&self, f_nondim: f64) -> f64 {
        f_nondim * self.force
    }

    /// Dimensionalize time: t = t̂ * t*
    pub fn dim_time(&self, t_nondim: f64) -> f64 {
        t_nondim * self.time
    }

    /// Dimensionalize viscosity: μ = μ̂ * μ*
    pub fn dim_viscosity(&self, mu_nondim: f64) -> f64 {
        mu_nondim * self.viscosity
    }

    // ========================================================================
    // Solution Vector Scaling (for JFNK solver)
    // ========================================================================

    /// Non-dimensionalize solution vector [v_x, v_y, v_z, ..., p_1, p_2, ...]
    ///
    /// Velocities scaled by v*, pressures by τ*
    pub fn nondim_solution(&self, sol_phys: &[f64], dof_mgr: &DofManager) -> Vec<f64> {
        let n = sol_phys.len();
        let mut sol_nondim = vec![0.0; n];
        let nv = dof_mgr.total_vel_dofs();

        // Scale velocity DOFs
        for i in 0..nv {
            sol_nondim[i] = self.nondim_velocity(sol_phys[i]);
        }

        // Scale pressure DOFs
        for i in nv..n {
            sol_nondim[i] = self.nondim_stress(sol_phys[i]);
        }

        sol_nondim
    }

    /// Dimensionalize solution vector [v̂_x, v̂_y, v̂_z, ..., p̂_1, p̂_2, ...]
    pub fn dim_solution(&self, sol_nondim: &[f64], dof_mgr: &DofManager) -> Vec<f64> {
        let n = sol_nondim.len();
        let mut sol_phys = vec![0.0; n];
        let nv = dof_mgr.total_vel_dofs();

        // Scale velocity DOFs
        for i in 0..nv {
            sol_phys[i] = self.dim_velocity(sol_nondim[i]);
        }

        // Scale pressure DOFs
        for i in nv..n {
            sol_phys[i] = self.dim_stress(sol_nondim[i]);
        }

        sol_phys
    }

    /// Non-dimensionalize residual vector (force residual)
    ///
    /// Same scaling as solution derivative: [F/F*, F/F*, ..., p/τ*, p/τ*, ...]
    pub fn nondim_residual(&self, res_phys: &[f64], dof_mgr: &DofManager) -> Vec<f64> {
        let n = res_phys.len();
        let mut res_nondim = vec![0.0; n];
        let nv = dof_mgr.total_vel_dofs();

        // Scale momentum residuals (forces)
        for i in 0..nv {
            res_nondim[i] = self.nondim_force(res_phys[i]);
        }

        // Scale continuity residuals (dimensionless already, but apply strain rate scaling)
        // ∇·v has units of 1/s, so scale by ė* = v*/L*
        for i in nv..n {
            res_nondim[i] = res_phys[i] / self.strain_rate;
        }

        res_nondim
    }

    /// Compute non-dimensional Rayleigh number
    ///
    /// Ra = (ρ* g Δρ L*³) / (μ* κ)
    ///
    /// For incompressible Stokes flow without temperature, we use a modified form:
    /// Ra_eff = (ρ* g L*²) / (μ* v*)
    ///
    /// This represents the ratio of buoyancy to viscous forces.
    pub fn rayleigh_number(&self, delta_rho: f64, thermal_diffusivity: f64) -> f64 {
        (self.density * self.gravity * delta_rho * self.length.powi(3))
            / (self.viscosity * thermal_diffusivity)
    }

    /// Compute non-dimensional buoyancy number (for isothermal cases)
    ///
    /// B = (ρ* g L*²) / (μ* v*) = (ρ* g L*) / τ*
    ///
    /// This is the ratio of gravitational to viscous stress
    pub fn buoyancy_number(&self) -> f64 {
        (self.density * self.gravity * self.length) / self.stress
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_characteristic_scales() {
        // Typical mantle convection scales
        let l_star = 1e6;                          // 1000 km
        let v_star = 1e-9;                         // ~3 cm/yr
        let mu_star = 1e21;                        // Pa·s
        let rho_star = 3300.0;                     // kg/m³
        let g = 9.81;                              // m/s²

        let scales = CharacteristicScales::new(l_star, v_star, mu_star, rho_star, g);

        // Check derived scales
        assert_relative_eq!(scales.time, 1e15, epsilon = 1e10);           // ~31.7 Myr
        assert_relative_eq!(scales.stress, 1e12, epsilon = 1e7);          // ~1 GPa
        assert_relative_eq!(scales.force, 1e24, epsilon = 1e19);          // Huge!
        assert_relative_eq!(scales.strain_rate, 1e-15, epsilon = 1e-20);  // Very small
    }

    #[test]
    fn test_nondimensionalization() {
        let scales = CharacteristicScales::new(1e5, 1e-9, 1e21, 3000.0, 9.81);

        // Test length
        let x_phys = 50_000.0; // 50 km
        let x_nd = scales.nondim_length(x_phys);
        assert_relative_eq!(x_nd, 0.5, epsilon = 1e-10);
        assert_relative_eq!(scales.dim_length(x_nd), x_phys, epsilon = 1e-6);

        // Test velocity
        let v_phys = 2e-9; // 2x reference
        let v_nd = scales.nondim_velocity(v_phys);
        assert_relative_eq!(v_nd, 2.0, epsilon = 1e-10);
        assert_relative_eq!(scales.dim_velocity(v_nd), v_phys, epsilon = 1e-15);

        // Test pressure
        let p_phys = 1e8; // 100 MPa
        let p_nd = scales.nondim_stress(p_phys);
        let expected_nd = p_phys / scales.stress;
        assert_relative_eq!(p_nd, expected_nd, epsilon = 1e-10);
    }

    #[test]
    fn test_buoyancy_number() {
        // Earth mantle typical values
        let scales = CharacteristicScales::new(
            2.9e6,      // Mantle depth
            1e-9,       // ~3 cm/yr
            1e21,       // Pa·s
            3300.0,     // kg/m³
            9.81,       // m/s²
        );

        let b = scales.buoyancy_number();
        // B = (ρ g L) / τ* = (ρ g L²) / (μ v)
        // B = (3300 * 9.81 * 2.9e6²) / (1e21 * 1e-9)
        // B ≈ 2.7e20 / 1e12 = 2.7e8
        assert!(b > 1e8 && b < 1e9, "Buoyancy number: {}", b);
    }
}
