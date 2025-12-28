/// Constitutive models for solid mechanics
///
/// Implements material stiffness matrices (D) that relate stress to strain.

use nalgebra::SMatrix;

/// Isotropic linear elastic material
///
/// Characterized by Young's modulus E and Poisson's ratio ν.
/// Valid for small strains and linear elastic behavior.
///
/// # References
/// - Timoshenko & Goodier, "Theory of Elasticity"
/// - Zienkiewicz & Taylor, "The Finite Element Method", Vol. 1
#[derive(Debug, Clone)]
pub struct IsotropicElasticity {
    pub youngs_modulus: f64,  // E (Pa)
    pub poisson_ratio: f64,   // ν (dimensionless)
}

impl IsotropicElasticity {
    /// Create new isotropic elastic material
    ///
    /// # Arguments
    /// * `youngs_modulus` - Young's modulus E (Pa), must be > 0
    /// * `poisson_ratio` - Poisson's ratio ν, must be in (-1, 0.5) for stability
    ///
    /// # Panics
    /// Panics if E ≤ 0 or ν is outside valid range
    pub fn new(youngs_modulus: f64, poisson_ratio: f64) -> Self {
        assert!(youngs_modulus > 0.0, "Young's modulus must be positive");
        assert!(
            poisson_ratio > -1.0 && poisson_ratio < 0.5,
            "Poisson's ratio must be in (-1, 0.5) for physical stability"
        );

        Self {
            youngs_modulus,
            poisson_ratio,
        }
    }

    /// Compute 6×6 constitutive matrix D for 3D elasticity
    ///
    /// Relates stress to strain in Voigt notation: σ = D ε
    ///
    /// Voigt ordering: [σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_zx]^T
    ///
    /// For isotropic material:
    /// ```text
    /// D = (E / ((1+ν)(1-2ν))) ×
    ///     [1-ν    ν    ν    0      0      0   ]
    ///     [ ν   1-ν    ν    0      0      0   ]
    ///     [ ν    ν   1-ν    0      0      0   ]
    ///     [ 0    0    0  (1-2ν)/2  0      0   ]
    ///     [ 0    0    0    0   (1-2ν)/2   0   ]
    ///     [ 0    0    0    0      0   (1-2ν)/2]
    /// ```
    ///
    /// # Returns
    /// 6×6 symmetric positive-definite constitutive matrix
    #[allow(non_snake_case)]
    pub fn constitutive_matrix(&self) -> SMatrix<f64, 6, 6> {
        let E = self.youngs_modulus;
        let nu = self.poisson_ratio;

        // Compute scaling factor
        let factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu));

        // Diagonal terms (normal stresses)
        let diag = 1.0 - nu;

        // Off-diagonal coupling (normal-normal)
        let off_diag = nu;

        // Shear modulus terms
        let shear = (1.0 - 2.0 * nu) / 2.0;

        // Build D matrix
        let mut D = SMatrix::<f64, 6, 6>::zeros();

        // Normal stress block (3×3 upper left)
        D[(0, 0)] = diag;
        D[(0, 1)] = off_diag;
        D[(0, 2)] = off_diag;

        D[(1, 0)] = off_diag;
        D[(1, 1)] = diag;
        D[(1, 2)] = off_diag;

        D[(2, 0)] = off_diag;
        D[(2, 1)] = off_diag;
        D[(2, 2)] = diag;

        // Shear stress block (3×3 lower right diagonal)
        D[(3, 3)] = shear;
        D[(4, 4)] = shear;
        D[(5, 5)] = shear;

        // Scale by factor
        D * factor
    }

    /// Compute Lamé parameters (λ, μ)
    ///
    /// Alternative parameterization of isotropic elasticity.
    ///
    /// # Returns
    /// (λ, μ) where:
    /// - λ = E ν / ((1+ν)(1-2ν)) - First Lamé parameter
    /// - μ = E / (2(1+ν)) - Shear modulus (second Lamé parameter)
    #[allow(non_snake_case)]
    pub fn lame_parameters(&self) -> (f64, f64) {
        let E = self.youngs_modulus;
        let nu = self.poisson_ratio;

        let lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = E / (2.0 * (1.0 + nu));

        (lambda, mu)
    }

    /// Compute bulk modulus K
    ///
    /// K = E / (3(1-2ν))
    ///
    /// Measures resistance to volumetric compression.
    #[allow(non_snake_case)]
    pub fn bulk_modulus(&self) -> f64 {
        let E = self.youngs_modulus;
        let nu = self.poisson_ratio;

        E / (3.0 * (1.0 - 2.0 * nu))
    }
}

// ============================================================================
// Newtonian Viscosity
// ============================================================================

/// Newtonian viscous constitutive model for fluids
///
/// Relates stress to strain-rate:
/// τ = 2μ ε̇_dev
///
/// Where:
/// - τ is the deviatoric stress tensor
/// - μ is the dynamic viscosity (Pa·s)
/// - ε̇_dev is the deviatoric strain-rate tensor
///
/// For compressible flow, the full relation is:
/// σ = -p I + 2μ ε̇_dev + ζ tr(ε̇) I
///
/// For now, we implement the deviatoric part only (ζ = 0).
#[derive(Debug, Clone, Copy)]
pub struct NewtonianViscosity {
    /// Dynamic viscosity μ (Pa·s)
    pub viscosity: f64,
    /// Bulk viscosity ζ (Pa·s) - used as penalty for incompressibility
    pub bulk_viscosity: f64,
}

impl NewtonianViscosity {
    /// Create a new Newtonian viscosity material
    ///
    /// # Arguments
    /// * `viscosity` - Dynamic viscosity μ in Pa·s
    ///
    /// # Example
    /// ```
    /// use geo_simulator::NewtonianViscosity;
    ///
    /// // Honey-like viscosity
    /// let material = NewtonianViscosity::new(1000.0);
    /// ```
    ///
    /// # Panics
    /// Panics if viscosity is not positive
    pub fn new(viscosity: f64) -> Self {
        assert!(viscosity > 0.0, "Viscosity must be positive, got {}", viscosity);

        Self { 
            viscosity,
            bulk_viscosity: 0.0, // Default to no penalty (purely deviatoric)
        }
    }

    /// Set bulk viscosity for incompressibility penalty
    ///
    /// Typically ζ ≈ 10^7 * μ for strong incompressibility without ill-conditioning
    pub fn with_penalty(mut self, penalty: f64) -> Self {
        self.bulk_viscosity = penalty;
        self
    }

    /// Apply standard geodynamic penalty for incompressibility
    ///
    /// Sets ζ = 1e7 * μ
    pub fn with_incompressible(mut self) -> Self {
        self.bulk_viscosity = self.viscosity * 1e7;
        self
    }

    /// Returns the constitutive matrix D relating strain-rate to stress
    ///
    /// For Newtonian fluid in Voigt notation:
    /// τ = D · ε̇
    ///
    /// Where τ = [τ_xx, τ_yy, τ_zz, τ_xy, τ_yz, τ_zx]
    ///       ε̇ = [ε̇_xx, ε̇_yy, ε̇_zz, γ̇_xy, γ̇_yz, γ̇_zx]
    ///
    /// For deviatoric stress (incompressible or low bulk viscosity):
    ///
    /// D = 2μ ×
    ///     [4/3  -2/3  -2/3   0    0    0  ]
    ///     [-2/3  4/3  -2/3   0    0    0  ]
    ///     [-2/3 -2/3   4/3   0    0    0  ]
    ///     [ 0     0     0    1    0    0  ]
    ///     [ 0     0     0    0    1    0  ]
    ///     [ 0     0     0    0    0    1  ]
    ///
    /// The 4/3, -2/3 terms come from the deviatoric projection:
    /// τ_dev = τ - (1/3)tr(τ)I
    ///
    /// # Returns
    /// 6×6 constitutive matrix in Voigt notation
    pub fn constitutive_matrix(&self) -> SMatrix<f64, 6, 6> {
        let mu = self.viscosity;
        let zeta = self.bulk_viscosity;

        // Relation: σ = 2μ ε̇_dev + ζ tr(ε̇) I
        // τ_ii = 2μ(ε̇_ii - 1/3 tr(ε̇)) + ζ tr(ε̇)
        //      = 2μ ε̇_ii + (ζ - 2μ/3) tr(ε̇)

        let factor_2mu = 2.0 * mu;
        let lambda_v = zeta - (2.0 * mu / 3.0);

        let mut D = SMatrix::<f64, 6, 6>::zeros();

        // Normal stress block
        for i in 0..3 {
            for j in 0..3 {
                D[(i, j)] = lambda_v;
            }
            D[(i, i)] += factor_2mu;
        }

        // Shear stress block
        D[(3, 3)] = factor_2mu;
        D[(4, 4)] = factor_2mu;
        D[(5, 5)] = factor_2mu;

        D
    }

    /// Returns the dynamic viscosity μ
    pub fn dynamic_viscosity(&self) -> f64 {
        self.viscosity
    }
}

// ============================================================================
// Maxwell Viscoelasticity
// ============================================================================

/// Maxwell viscoelastic constitutive model
///
/// Combines elastic spring and viscous dashpot in series.
/// Exhibits time-dependent stress relaxation.
///
/// **Physics:**
/// - Short times (t << τ_M): Elastic response (σ ≈ 2G ε)
/// - Long times (t >> τ_M): Viscous flow (σ ≈ 2μ dε/dt)
/// - Relaxation time: τ_M = μ/G
///
/// **Governing Equation (deviatoric):**
/// dσ/dt + (G/μ) σ = 2G dε/dt
///
/// **Backward Euler Discretization:**
/// σ_{n+1} = [σ_n + 2G Δε] / [1 + Δt/τ_M]
///
/// # References
/// - Malvern, "Introduction to the Mechanics of a Continuous Medium"
/// - Turcotte & Schubert, "Geodynamics", Ch. 6
#[derive(Debug, Clone, Copy)]
pub struct MaxwellViscoelasticity {
    /// Young's modulus (Pa)
    pub youngs_modulus: f64,
    /// Poisson's ratio (dimensionless)
    pub poisson_ratio: f64,
    /// Dynamic viscosity (Pa·s)
    pub viscosity: f64,
}

impl MaxwellViscoelasticity {
    /// Create new Maxwell viscoelastic material
    ///
    /// # Arguments
    /// * `youngs_modulus` - Young's modulus E (Pa), must be > 0
    /// * `poisson_ratio` - Poisson's ratio ν, must be in (-1, 0.5)
    /// * `viscosity` - Dynamic viscosity μ (Pa·s), must be > 0
    ///
    /// # Example
    /// ```
    /// use geo_simulator::MaxwellViscoelasticity;
    ///
    /// // Mantle rheology (typical values)
    /// let E = 100e9;      // 100 GPa
    /// let nu = 0.25;      // Poisson's ratio
    /// let mu = 1e19;      // 10^19 Pa·s (mantle viscosity)
    ///
    /// let material = MaxwellViscoelasticity::new(E, nu, mu);
    /// ```
    ///
    /// # Panics
    /// Panics if parameters are outside valid ranges
    pub fn new(youngs_modulus: f64, poisson_ratio: f64, viscosity: f64) -> Self {
        assert!(youngs_modulus > 0.0, "Young's modulus must be positive, got {}", youngs_modulus);
        assert!(
            poisson_ratio > -1.0 && poisson_ratio < 0.5,
            "Poisson's ratio must be in (-1, 0.5), got {}", poisson_ratio
        );
        assert!(viscosity > 0.0, "Viscosity must be positive, got {}", viscosity);

        Self {
            youngs_modulus,
            poisson_ratio,
            viscosity,
        }
    }

    /// Shear modulus G = E / (2(1+ν))
    ///
    /// Controls instantaneous elastic response.
    #[allow(non_snake_case)]
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Maxwell relaxation time τ_M = μ / G
    ///
    /// Characteristic time scale for stress relaxation.
    /// - At t << τ_M: Material behaves elastically
    /// - At t >> τ_M: Material flows viscously
    ///
    /// # Returns
    /// Relaxation time in seconds
    pub fn relaxation_time(&self) -> f64 {
        self.viscosity / self.shear_modulus()
    }

    /// Effective constitutive matrix for backward Euler time integration
    ///
    /// D_eff = D_elastic / [1 + Δt/τ_M]
    ///
    /// Where:
    /// - D_elastic is the elastic constitutive matrix (6×6)
    /// - τ_M = μ/G is the Maxwell relaxation time
    /// - Δt is the time step size
    ///
    /// **Limits:**
    /// - As Δt → 0: D_eff → D_elastic (elastic limit)
    /// - As Δt → ∞: D_eff → 0 (complete relaxation)
    ///
    /// # Arguments
    /// * `dt` - Time step size (seconds)
    ///
    /// # Returns
    /// 6×6 effective constitutive matrix for this time step
    #[allow(non_snake_case)]
    pub fn effective_constitutive_matrix(&self, dt: f64) -> SMatrix<f64, 6, 6> {
        // Get base elastic matrix
        let elastic = IsotropicElasticity::new(self.youngs_modulus, self.poisson_ratio);
        let D_elastic = elastic.constitutive_matrix();

        // Relaxation factor: 1 / [1 + Δt/τ_M]
        let tau_M = self.relaxation_time();
        let relaxation_factor = 1.0 / (1.0 + dt / tau_M);

        // Scale elastic matrix by relaxation factor
        D_elastic * relaxation_factor
    }

    /// Get base elastic constitutive matrix (without time dependence)
    ///
    /// Used for stress computation from strain increments.
    ///
    /// # Returns
    /// 6×6 elastic constitutive matrix
    #[allow(non_snake_case)]
    pub fn elastic_matrix(&self) -> SMatrix<f64, 6, 6> {
        IsotropicElasticity::new(self.youngs_modulus, self.poisson_ratio)
            .constitutive_matrix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_constitutive_matrix_symmetry() {
        let mat = IsotropicElasticity::new(100e9, 0.25);
        let D = mat.constitutive_matrix();

        // Check symmetry: D = D^T
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(D[(i, j)], D[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_special_case_zero_poisson() {
        // ν = 0 should give diagonal coupling = 0
        let mat = IsotropicElasticity::new(100e9, 0.0);
        let D = mat.constitutive_matrix();

        // Off-diagonal normal stress terms should be zero
        assert_relative_eq!(D[(0, 1)], 0.0, epsilon = 1e-6);
        assert_relative_eq!(D[(0, 2)], 0.0, epsilon = 1e-6);
        assert_relative_eq!(D[(1, 2)], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_lame_parameters() {
        let E = 100e9;
        let nu = 0.25;
        let mat = IsotropicElasticity::new(E, nu);

        let (lambda, mu) = mat.lame_parameters();

        // Expected values
        let expected_lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let expected_mu = E / (2.0 * (1.0 + nu));

        assert_relative_eq!(lambda, expected_lambda, epsilon = 1e-3);
        assert_relative_eq!(mu, expected_mu, epsilon = 1e-3);
    }

    #[test]
    fn test_bulk_modulus() {
        let E = 100e9;
        let nu = 0.25;
        let mat = IsotropicElasticity::new(E, nu);

        let K = mat.bulk_modulus();
        let expected_K = E / (3.0 * (1.0 - 2.0 * nu));

        assert_relative_eq!(K, expected_K, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "Young's modulus must be positive")]
    fn test_negative_youngs_modulus() {
        IsotropicElasticity::new(-100e9, 0.25);
    }

    #[test]
    #[should_panic(expected = "Poisson's ratio must be in")]
    fn test_invalid_poisson_ratio() {
        IsotropicElasticity::new(100e9, 0.6);  // Too high
    }

    // ========================================================================
    // Newtonian Viscosity Tests
    // ========================================================================

    #[test]
    fn test_newtonian_viscosity_matrix_symmetry() {
        let mat = NewtonianViscosity::new(1000.0);
        let D = mat.constitutive_matrix();

        // Check symmetry: D = D^T
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(D[(i, j)], D[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_newtonian_viscosity_matrix_structure() {
        let mu = 1000.0;
        let mat = NewtonianViscosity::new(mu);
        let D = mat.constitutive_matrix();

        // Check diagonal elements (normal components)
        let expected_diag = mu * (4.0 / 3.0);
        assert_relative_eq!(D[(0, 0)], expected_diag, epsilon = 1e-10);
        assert_relative_eq!(D[(1, 1)], expected_diag, epsilon = 1e-10);
        assert_relative_eq!(D[(2, 2)], expected_diag, epsilon = 1e-10);

        // Check off-diagonal elements (normal-normal coupling)
        let expected_off = mu * (-2.0 / 3.0);
        assert_relative_eq!(D[(0, 1)], expected_off, epsilon = 1e-10);
        assert_relative_eq!(D[(0, 2)], expected_off, epsilon = 1e-10);
        assert_relative_eq!(D[(1, 2)], expected_off, epsilon = 1e-10);

        // Check shear components
        let expected_shear = 2.0 * mu;
        assert_relative_eq!(D[(3, 3)], expected_shear, epsilon = 1e-10);
        assert_relative_eq!(D[(4, 4)], expected_shear, epsilon = 1e-10);
        assert_relative_eq!(D[(5, 5)], expected_shear, epsilon = 1e-10);

        // Check that normal-shear coupling is zero
        for i in 0..3 {
            for j in 3..6 {
                assert_relative_eq!(D[(i, j)], 0.0, epsilon = 1e-10);
                assert_relative_eq!(D[(j, i)], 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_newtonian_viscosity_scaling() {
        let mu1 = 1000.0;
        let mu2 = 2000.0;

        let mat1 = NewtonianViscosity::new(mu1);
        let mat2 = NewtonianViscosity::new(mu2);

        let D1 = mat1.constitutive_matrix();
        let D2 = mat2.constitutive_matrix();

        // D should scale linearly with viscosity
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(D2[(i, j)], D1[(i, j)] * 2.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_newtonian_viscosity_deviatoric_property() {
        let mu = 1000.0;
        let mat = NewtonianViscosity::new(mu);
        let D = mat.constitutive_matrix();

        // Apply hydrostatic strain-rate: ε̇ = [1, 1, 1, 0, 0, 0]
        // Should give zero deviatoric stress (only pressure contribution)
        let eps_dot = nalgebra::SVector::<f64, 6>::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        let tau = D * eps_dot;

        // Sum of normal stresses should be zero (deviatoric)
        let trace = tau[0] + tau[1] + tau[2];
        assert_relative_eq!(trace, 0.0, epsilon = 1e-10);

        // Shear stresses should be zero (no shear strain-rate)
        assert_relative_eq!(tau[3], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tau[4], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tau[5], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_newtonian_pure_shear() {
        let mu = 1000.0;
        let mat = NewtonianViscosity::new(mu);
        let D = mat.constitutive_matrix();

        // Apply pure shear strain-rate: γ̇_xy = 1.0
        let eps_dot = nalgebra::SVector::<f64, 6>::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let tau = D * eps_dot;

        // Should give τ_xy = 2μ
        assert_relative_eq!(tau[3], 2.0 * mu, epsilon = 1e-10);

        // All other components should be zero
        assert_relative_eq!(tau[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tau[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tau[2], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tau[4], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tau[5], 0.0, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "Viscosity must be positive")]
    fn test_negative_viscosity() {
        NewtonianViscosity::new(-100.0);
    }

    #[test]
    #[should_panic(expected = "Viscosity must be positive")]
    fn test_zero_viscosity() {
        NewtonianViscosity::new(0.0);
    }

    // ========================================================================
    // Maxwell Viscoelasticity Tests
    // ========================================================================

    #[test]
    fn test_maxwell_shear_modulus() {
        let E = 100e9;
        let nu = 0.25;
        let mu = 1e19;

        let material = MaxwellViscoelasticity::new(E, nu, mu);
        let G = material.shear_modulus();

        // G = E / (2(1+ν))
        let expected_G = E / (2.0 * (1.0 + nu));
        assert_relative_eq!(G, expected_G, epsilon = 1e-3);
    }

    #[test]
    fn test_maxwell_relaxation_time() {
        let E = 100e9;
        let nu = 0.25;
        let mu = 1e19;

        let material = MaxwellViscoelasticity::new(E, nu, mu);
        let tau_M = material.relaxation_time();

        // τ_M = μ / G = μ / (E / (2(1+ν)))
        let G = E / (2.0 * (1.0 + nu));
        let expected_tau_M = mu / G;

        assert_relative_eq!(tau_M, expected_tau_M, epsilon = 1e-3);
    }

    #[test]
    fn test_maxwell_effective_matrix_elastic_limit() {
        // When dt → 0, should recover elastic matrix
        let E = 100e9;
        let nu = 0.25;
        let mu = 1e19;

        let maxwell = MaxwellViscoelasticity::new(E, nu, mu);
        let elastic = IsotropicElasticity::new(E, nu);

        let dt_small = 1e-20;  // Very small time step
        let D_eff = maxwell.effective_constitutive_matrix(dt_small);
        let D_elastic = elastic.constitutive_matrix();

        // Should be essentially identical
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(D_eff[(i, j)], D_elastic[(i, j)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_maxwell_effective_matrix_relaxed_limit() {
        // When dt → ∞, matrix should approach zero
        let E = 100e9;
        let nu = 0.25;
        let mu = 1e19;

        let material = MaxwellViscoelasticity::new(E, nu, mu);
        let tau_M = material.relaxation_time();

        // Use dt = 1000 * tau_M
        let dt_large = 1000.0 * tau_M;
        let D_eff = material.effective_constitutive_matrix(dt_large);
        let D_elastic = material.elastic_matrix();

        // D_eff should be ~1/1000 of D_elastic
        let expected_factor = 1.0 / (1.0 + dt_large / tau_M);

        for i in 0..6 {
            for j in 0..6 {
                let expected = D_elastic[(i, j)] * expected_factor;
                assert_relative_eq!(D_eff[(i, j)], expected, epsilon = 1e-6);
            }
        }

        // Should be very small
        assert!(D_eff[(0, 0)] < D_elastic[(0, 0)] * 0.01);
    }

    #[test]
    fn test_maxwell_effective_matrix_symmetry() {
        let material = MaxwellViscoelasticity::new(100e9, 0.25, 1e19);
        let dt = 3.16e7;  // ~1 year

        let D_eff = material.effective_constitutive_matrix(dt);

        // Check symmetry
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(D_eff[(i, j)], D_eff[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_maxwell_effective_matrix_intermediate_time() {
        // Test dt = tau_M case
        let E = 100e9;
        let nu = 0.25;
        let mu = 1e19;

        let material = MaxwellViscoelasticity::new(E, nu, mu);
        let tau_M = material.relaxation_time();

        let D_eff = material.effective_constitutive_matrix(tau_M);
        let D_elastic = material.elastic_matrix();

        // When dt = tau_M, relaxation factor = 1 / (1 + 1) = 0.5
        for i in 0..6 {
            for j in 0..6 {
                let expected = D_elastic[(i, j)] * 0.5;
                assert_relative_eq!(D_eff[(i, j)], expected, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_maxwell_elastic_matrix_consistency() {
        let E = 100e9;
        let nu = 0.25;
        let mu = 1e19;

        let maxwell = MaxwellViscoelasticity::new(E, nu, mu);
        let elastic = IsotropicElasticity::new(E, nu);

        let D_maxwell = maxwell.elastic_matrix();
        let D_elastic = elastic.constitutive_matrix();

        // Should be identical
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(D_maxwell[(i, j)], D_elastic[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Young's modulus must be positive")]
    fn test_maxwell_negative_youngs_modulus() {
        MaxwellViscoelasticity::new(-100e9, 0.25, 1e19);
    }

    #[test]
    #[should_panic(expected = "Poisson's ratio must be in")]
    fn test_maxwell_invalid_poisson_ratio() {
        MaxwellViscoelasticity::new(100e9, 0.6, 1e19);
    }

    #[test]
    #[should_panic(expected = "Viscosity must be positive")]
    fn test_maxwell_negative_viscosity() {
        MaxwellViscoelasticity::new(100e9, 0.25, -1e19);
    }

    #[test]
    #[should_panic(expected = "Viscosity must be positive")]
    fn test_maxwell_zero_viscosity() {
        MaxwellViscoelasticity::new(100e9, 0.25, 0.0);
    }

    #[test]
    fn test_maxwell_realistic_geodynamic_parameters() {
        // Test with realistic mantle rheology
        let E = 100e9;      // 100 GPa
        let nu = 0.25;      // Typical for rocks
        let mu = 1e19;      // 10^19 Pa·s (upper mantle)

        let material = MaxwellViscoelasticity::new(E, nu, mu);

        let G = material.shear_modulus();
        let tau_M = material.relaxation_time();

        // G should be ~40 GPa
        assert!(G > 30e9 && G < 50e9, "G = {} Pa is out of expected range", G);

        // tau_M = μ / G = 1e19 / 40e9 = 2.5e8 seconds ≈ 8 years
        // Expect tau_M in range 1-10 years (3.16e7 to 3.16e8 seconds)
        let one_year = 3.16e7;  // seconds
        let tau_M_years = tau_M / one_year;
        assert!(tau_M_years > 1.0 && tau_M_years < 10.0,
                "tau_M = {} years is out of expected range (1-10 years)", tau_M_years);
    }
}
