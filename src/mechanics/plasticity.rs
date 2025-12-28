/// Plasticity models for geodynamic materials
///
/// Implements yield criteria and viscoplastic regularization for
/// combining viscous flow with plastic yielding.

use nalgebra::SMatrix;

/// Drucker-Prager yield criterion for geomaterials
///
/// **Yield Function:**
/// ```text
/// F = √J₂ - (C cos φ - P sin φ)
/// ```
///
/// where:
/// - J₂ = second invariant of deviatoric stress = (1/2) s:s
/// - C = cohesion (Pa)
/// - φ = internal friction angle (radians)
/// - P = pressure = -tr(σ)/3 (compressive positive)
///
/// **Physical Interpretation:**
/// - C = 0, φ = 0: Von Mises (pressure-independent)
/// - C > 0, φ = 0: Cohesive material
/// - φ > 0: Frictional material (strength increases with pressure)
///
/// **Geodynamic Parameters:**
/// - Dry olivine: C ≈ 10-50 MPa, φ ≈ 15-30°
/// - Wet olivine: C ≈ 1-10 MPa, φ ≈ 5-15°
/// - Crust: C ≈ 20-100 MPa, φ ≈ 20-40° (Byerlee's law)
///
/// # References
/// - Drucker & Prager (1952), "Soil mechanics and plastic analysis"
/// - Moresi et al. (2003), "A Lagrangian integration point FEM"
/// - Glerum et al. (2018), "Nonlinear viscoplasticity in ASPECT"
#[derive(Debug, Clone, Copy)]
pub struct DruckerPrager {
    /// Cohesion C (Pa)
    pub cohesion: f64,
    /// Internal friction angle φ (radians)
    pub friction_angle: f64,
    /// Smoothing parameter (Pa) for yield surface apex
    pub smoothing: f64,
    /// Minimum cohesion after softening (Pa)
    pub cohesion_min: f64,
    /// Minimum friction angle after softening (radians)
    pub friction_min: f64,
    /// Reference plastic strain for softening completion
    pub strain_ref: f64,
}

impl DruckerPrager {
    /// Create new Drucker-Prager yield criterion
    ///
    /// # Arguments
    /// * `cohesion` - Cohesion in Pa (must be > 0)
    /// * `friction_angle` - Internal friction angle in radians (0 to π/2)
    ///
    /// # Example
    /// ```
    /// // Typical mantle lithosphere
    /// let plastic = DruckerPrager::new(20e6, 20.0_f64.to_radians());
    /// ```
    pub fn new(cohesion: f64, friction_angle: f64) -> Self {
        assert!(cohesion >= 0.0, "Cohesion must be non-negative");
        assert!(
            friction_angle >= 0.0 && friction_angle < std::f64::consts::FRAC_PI_2,
            "Friction angle must be in [0, π/2)"
        );

        Self {
            cohesion,
            friction_angle,
            smoothing: 1e4, // Default smoothing of 10 kPa
            cohesion_min: cohesion, // No softening by default
            friction_min: friction_angle,
            strain_ref: 1.0, 
        }
    }

    /// Set softening parameters
    pub fn with_softening(
        mut self,
        cohesion_min: f64,
        friction_min_deg: f64,
        strain_ref: f64,
    ) -> Self {
        self.cohesion_min = cohesion_min;
        self.friction_min = friction_min_deg.to_radians();
        self.strain_ref = strain_ref;
        self
    }

    /// Compute softened properties for a given plastic strain
    pub fn softened_properties(&self, accumulated_strain: f64) -> (f64, f64) {
        if accumulated_strain <= 0.0 {
            return (self.cohesion, self.friction_angle);
        }

        let ratio = (accumulated_strain / self.strain_ref).min(1.0);
        
        let c = self.cohesion - (self.cohesion - self.cohesion_min) * ratio;
        let phi = self.friction_angle - (self.friction_angle - self.friction_min) * ratio;
        
        (c, phi)
    }

    /// Convenience constructor with friction angle in degrees
    pub fn new_degrees(cohesion: f64, friction_angle_deg: f64) -> Self {
        Self::new(cohesion, friction_angle_deg.to_radians())
    }

    /// Evaluate yield function F
    ///
    /// Returns:
    /// - F < 0: Elastic regime (no yielding)
    /// - F = 0: At yield surface
    /// - F > 0: Violates yield criterion (yielding occurs)
    ///
    /// # Arguments
    /// * `stress` - Stress tensor in Voigt notation (6×1)
    ///
    /// # Returns
    /// Yield function value F
    pub fn yield_function(&self, stress: &SMatrix<f64, 6, 1>) -> f64 {
        // Compute pressure (compressive positive)
        let pressure = -(stress[0] + stress[1] + stress[2]) / 3.0;

        // Compute deviatoric stress
        let s = SMatrix::<f64, 6, 1>::from_column_slice(&[
            stress[0] + pressure,
            stress[1] + pressure,
            stress[2] + pressure,
            stress[3],
            stress[4],
            stress[5],
        ]);

        // Second invariant: J₂ = (1/2) s:s
        // For Voigt notation: s:s = s₀² + s₁² + s₂² + 2(s₃² + s₄² + s₅²)
        let j2 = 0.5 * (
            s[0] * s[0] + s[1] * s[1] + s[2] * s[2] +
            2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5])
        );

        let sqrt_j2 = j2.sqrt();

        // Yield function: F = √J₂ - (C cos φ + P sin φ)
        let yield_strength = self.cohesion * self.friction_angle.cos()
                           + pressure * self.friction_angle.sin();

        sqrt_j2 - yield_strength
    }

    /// Compute effective plastic viscosity for viscoplastic regularization
    ///
    /// **Viscoplastic Method:**
    /// Instead of return mapping iteration, we cap the viscosity:
    ///
    /// ```text
    /// μ_plastic = τ_yield / (2 √J₂(ε̇))
    /// ```
    ///
    /// where:
    /// - τ_yield = C cos φ - P sin φ (yield strength)
    /// - √J₂(ε̇) = √[(1/2) ε̇:ε̇] (strain rate magnitude)
    ///
    /// Then combined viscosity:
    /// ```text
    /// μ_eff = min(μ_viscous, μ_plastic)
    /// ```
    ///
    /// **Advantages over return mapping:**
    /// - No iteration required (10x faster)
    /// - Naturally regularizes stress singularities
    /// - Stable for quasi-static problems
    /// - Standard in ASPECT, CitcomS, Underworld2
    ///
    /// # Arguments
    /// * `strain_rate` - Strain rate tensor in Voigt notation (6×1)
    /// * `pressure` - Pressure (compressive positive)
    ///
    /// # Returns
    /// Effective plastic viscosity μ_plastic
    ///
    /// # Notes
    /// - Returns f64::INFINITY if not yielding (√J₂ → 0)
    /// - Minimum viscosity cap prevents division by zero
    pub fn plastic_viscosity(
        &self,
        strain_rate: &SMatrix<f64, 6, 1>,
        pressure: f64,
    ) -> f64 {
        // Compute deviatoric strain rate magnitude: √J₂(ε̇)
        // Since we use engineering shear strain γ = 2ε, the invariant is:
        // J₂ = 1/2 ε:ε = 1/2 (ε_xx² + ε_yy² + ε_zz²) + ε_xy² + ε_yz² + ε_zx²
        //    = 1/2 [ε_xx² + ε_yy² + ε_zz² + 1/2 (γ_xy² + γ_yz² + γ_zx²)]
        let edot = strain_rate;
        let j2_edot = 0.5 * (
            edot[0] * edot[0] + edot[1] * edot[1] + edot[2] * edot[2] +
            0.5 * (edot[3] * edot[3] + edot[4] * edot[4] + edot[5] * edot[5])
        );

        // Hyperbolic smoothing to avoid apex singularity
        // Use a very small epsilon relative to strain rates (typical geodynamic 1e-15)
        let sqrt_j2_edot = (j2_edot + 1e-40).sqrt();

        // Avoid division by zero (no deformation = no plastic viscosity limit)
        if sqrt_j2_edot < 1e-30 {
            return f64::INFINITY;
        }

        // Yield strength: τ_y = C cos φ + P sin φ
        let tau_yield = self.cohesion * self.friction_angle.cos()
                      + pressure * self.friction_angle.sin();

        // Plastic viscosity: μ_p = τ_y / (2 √J₂(ε̇))
        let mu_plastic = tau_yield / (2.0 * sqrt_j2_edot);

        // Apply minimum viscosity cap (prevent unrealistic low values)
        const MIN_VISCOSITY: f64 = 1e16;  // 10^16 Pa·s (realistic lower bound)
        mu_plastic.max(MIN_VISCOSITY)
    }

    /// Compute effective plastic viscosity with strain softening
    pub fn softened_viscosity(
        &self,
        strain_rate: &SMatrix<f64, 6, 1>,
        pressure: f64,
        accumulated_strain: f64,
    ) -> f64 {
        let (c, phi) = self.softened_properties(accumulated_strain);
        
        // Temporarily swap properties to use plastic_viscosity logic
        let mut temp_dp = *self;
        temp_dp.cohesion = c;
        temp_dp.friction_angle = phi;
        
        temp_dp.plastic_viscosity(strain_rate, pressure)
    }
}

/// Elasto-Visco-Plastic material combining Maxwell viscoelasticity with Drucker-Prager plasticity
///
/// **Rheological Model:**
/// - Elastic: Spring (G, K)
/// - Viscous: Dashpot (μ_viscous)
/// - Plastic: Drucker-Prager yield limit (C, φ)
///
/// **Effective Viscosity:**
/// ```text
/// μ_eff = min(μ_viscous, μ_plastic)
/// ```
///
/// **Stress Update:**
/// 1. Elastic predictor: σ^trial = σ_n + D_elastic * Δε
/// 2. Compute μ_eff from current strain rate
/// 3. Relax deviatoric stress: s_{n+1} = s^trial / [1 + Δt·G/μ_eff]
///
/// # Example
/// ```
/// use geo_simulator::*;
///
/// // Mantle lithosphere rheology
/// let E = 100e9;           // 100 GPa
/// let nu = 0.25;
/// let mu = 1e23;           // 10^23 Pa·s (viscous)
/// let C = 20e6;            // 20 MPa cohesion
/// let phi = 20.0_f64.to_radians();  // 20° friction
///
/// let material = ElastoViscoPlastic::new(E, nu, mu, C, phi);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ElastoViscoPlastic {
    /// Young's modulus (Pa)
    pub youngs_modulus: f64,

    /// Poisson's ratio (dimensionless)
    pub poisson_ratio: f64,

    /// Background viscosity (Pa·s)
    pub viscosity: f64,

    /// Drucker-Prager yield criterion
    pub plasticity: DruckerPrager,
}

impl ElastoViscoPlastic {
    /// Create new EVP material
    pub fn new(
        youngs_modulus: f64,
        poisson_ratio: f64,
        viscosity: f64,
        cohesion: f64,
        friction_angle: f64,
    ) -> Self {
        assert!(youngs_modulus > 0.0, "E must be positive");
        assert!(
            poisson_ratio >= -1.0 && poisson_ratio < 0.5,
            "ν must be in [-1, 0.5)"
        );
        assert!(viscosity > 0.0, "μ must be positive");

        Self {
            youngs_modulus,
            poisson_ratio,
            viscosity,
            plasticity: DruckerPrager::new(cohesion, friction_angle),
        }
    }

    /// Shear modulus G = E / (2(1+ν))
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Bulk modulus K = E / (3(1-2ν))
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))
    }

    /// Maxwell relaxation time τ_M = μ / G (without plasticity)
    pub fn relaxation_time(&self) -> f64 {
        self.viscosity / self.shear_modulus()
    }

    /// Compute effective viscosity combining viscous and plastic limits
    ///
    /// # Arguments
    /// * `strain_rate` - Strain rate tensor (6×1 Voigt)
    /// * `pressure` - Current pressure (compressive positive)
    ///
    /// # Returns
    /// Effective viscosity μ_eff = min(μ_viscous, μ_plastic)
    pub fn effective_viscosity(
        &self,
        strain_rate: &SMatrix<f64, 6, 1>,
        pressure: f64,
    ) -> f64 {
        let mu_plastic = self.plasticity.plastic_viscosity(strain_rate, pressure);
        self.viscosity.min(mu_plastic)
    }

    /// Check if material is currently yielding
    ///
    /// # Arguments
    /// * `stress` - Current stress tensor (6×1 Voigt)
    ///
    /// # Returns
    /// true if F ≥ 0 (yielding), false otherwise
    pub fn is_yielding(&self, stress: &SMatrix<f64, 6, 1>) -> bool {
        self.plasticity.yield_function(stress) >= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_drucker_prager_von_mises_limit() {
        // Von Mises = Drucker-Prager with φ = 0
        let dp = DruckerPrager::new(100e6, 0.0);

        // Pure shear stress σ_xy = 50 MPa
        let stress = SMatrix::<f64, 6, 1>::from_column_slice(&[
            0.0, 0.0, 0.0, 50e6, 0.0, 0.0
        ]);

        // √J₂ = |σ_xy| = 50 MPa for pure shear
        // F = 50 - 100 = -50 MPa (elastic)
        let f = dp.yield_function(&stress);
        assert!(f < 0.0, "Should be elastic");
        assert!((f + 50e6).abs() < 1e6, "Should be 50 MPa below yield");
    }

    #[test]
    fn test_drucker_prager_pressure_dependence() {
        // Material with friction (pressure-dependent strength)
        let dp = DruckerPrager::new_degrees(20e6, 30.0);  // C=20 MPa, φ=30°

        // Test at two pressures with same deviatoric stress
        let dev_stress = 30e6;  // 30 MPa deviatoric

        // Higher pressure should increase yield strength (compressive positive)
        let stress_low_p = SMatrix::<f64, 6, 1>::from_column_slice(&[
            -10e6, -20e6, -10e6, 0.0, 0.0, 0.0
        ]);
        let stress_high_p = SMatrix::<f64, 6, 1>::from_column_slice(&[
            -100e6, -110e6, -100e6, 0.0, 0.0, 0.0
        ]);

        let f_low = dp.yield_function(&stress_low_p);
        let f_high = dp.yield_function(&stress_high_p);

        // Higher pressure means higher yield strength -> smaller yield function value
        assert!(f_high < f_low, "Higher pressure should increase yield strength (lower F)");
    }

    #[test]
    fn test_plastic_viscosity_basic() {
        let dp = DruckerPrager::new(20e6, 0.0); // Pure Von-Mises limit (C=20MPa, φ=0)
        
        // Pure shear strain rate: γ_xy = 1e-15, others 0
        let edot = SMatrix::<f64, 6, 1>::from_column_slice(&[
            0.0, 0.0, 0.0, 1e-15, 0.0, 0.0
        ]);
        let pressure = 0.0;

        // τ_y = C = 20e6
        // √J₂(ε̇) = √(1/2 * 0.5 * (1e-15)²) = √(0.25 * 1e-30) = 0.5e-15
        // μ_p = τ_y / (2 * √J₂) = 20e6 / (2 * 0.5e-15) = 20e6 / 1e-15 = 2e22
        let mu_p = dp.plastic_viscosity(&edot, pressure);
        assert_relative_eq!(mu_p, 2e22, epsilon = 1e18);
        assert!(mu_p > 0.0 && mu_p.is_finite());

        // For typical parameters, should be ~10^22-10^24 Pa·s
        assert!(mu_p > 1e20 && mu_p < 1e26);
    }

    #[test]
    fn test_plastic_viscosity_zero_strain_rate() {
        let dp = DruckerPrager::new(20e6, 0.0);
        let edot = SMatrix::<f64, 6, 1>::zeros();
        let pressure = 0.0;
        
        // With zero strain rate, smoothing ensures viscosity is large but finite
        let mu_p = dp.plastic_viscosity(&edot, pressure);
        assert!(mu_p >= 1e16); // MIN_VISCOSITY cap
    }

    #[test]
    fn test_evp_effective_viscosity() {
        let evp = ElastoViscoPlastic::new(
            100e9,      // E = 100 GPa
            0.25,       // ν = 0.25
            1e23,       // μ = 10^23 Pa·s
            50e6,       // C = 50 MPa
            20.0_f64.to_radians(),  // φ = 20°
        );

        // Low strain rate (viscous regime)
        let edot_low = 1e-18;
        let strain_rate_low = SMatrix::<f64, 6, 1>::from_column_slice(&[
            edot_low, 0.0, 0.0, 0.0, 0.0, 0.0
        ]);
        let pressure = 100e6;

        let mu_eff_low = evp.effective_viscosity(&strain_rate_low, pressure);

        // At low strain rate, plastic viscosity >> background viscosity
        // So μ_eff ≈ μ_viscous
        assert!((mu_eff_low - evp.viscosity).abs() / evp.viscosity < 0.01);

        // High strain rate (plastic regime)
        let edot_high = 1e-12;
        let strain_rate_high = SMatrix::<f64, 6, 1>::from_column_slice(&[
            edot_high, 0.0, 0.0, 0.0, 0.0, 0.0
        ]);

        let mu_eff_high = evp.effective_viscosity(&strain_rate_high, pressure);

        // At high strain rate, plastic viscosity << background viscosity
        // So μ_eff ≈ μ_plastic < μ_viscous
        assert!(mu_eff_high < evp.viscosity);
    }

    #[test]
    fn test_evp_is_yielding() {
        let evp = ElastoViscoPlastic::new(
            100e9, 0.25, 1e23, 50e6, 20.0_f64.to_radians()
        );

        // Low deviatoric stress (elastic)
        let stress_elastic = SMatrix::<f64, 6, 1>::from_column_slice(&[
            -100e6 + 10e6, -100e6, -100e6, 0.0, 0.0, 0.0
        ]);
        assert!(!evp.is_yielding(&stress_elastic));

        // High deviatoric stress (plastic)
        let stress_plastic = SMatrix::<f64, 6, 1>::from_column_slice(&[
            -100e6 + 500e6, -100e6, -100e6, 0.0, 0.0, 0.0
        ]);
        assert!(evp.is_yielding(&stress_plastic));
    }

    #[test]
    fn test_softening() {
        let dp = DruckerPrager::new(20e6, 20.0_f64.to_radians())
            .with_softening(10e6, 10.0, 0.1);
        
        // No strain: use initial properties
        let (c0, phi0) = dp.softened_properties(0.0);
        assert_eq!(c0, 20e6);
        assert_relative_eq!(phi0, 20.0_f64.to_radians());

        // Full strain (0.1): use min properties
        let (c1, phi1) = dp.softened_properties(0.1);
        assert_eq!(c1, 10e6);
        assert_relative_eq!(phi1, 10.0_f64.to_radians());

        // Intermediate strain (0.05): half softened
        let (c2, phi2) = dp.softened_properties(0.05);
        assert_eq!(c2, 15e6);
        assert_relative_eq!(phi2, 15.0_f64.to_radians());
    }
}
