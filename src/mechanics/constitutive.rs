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

        Self { viscosity }
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

        // Deviatoric part: 2μ [I - (1/3)11^T]
        // For normal components: 2μ(4/3) on diagonal, 2μ(-2/3) off-diagonal
        // For shear components: 2μ

        let diag_normal = 2.0 * mu * (4.0 / 3.0);  // 2μ(4/3)
        let off_diag = 2.0 * mu * (-2.0 / 3.0);    // 2μ(-2/3)
        let shear = 2.0 * mu;                       // 2μ

        let mut D = SMatrix::<f64, 6, 6>::zeros();

        // Normal stress block (3×3 upper left) - deviatoric projection
        D[(0, 0)] = diag_normal;
        D[(0, 1)] = off_diag;
        D[(0, 2)] = off_diag;

        D[(1, 0)] = off_diag;
        D[(1, 1)] = diag_normal;
        D[(1, 2)] = off_diag;

        D[(2, 0)] = off_diag;
        D[(2, 1)] = off_diag;
        D[(2, 2)] = diag_normal;

        // Shear stress block (3×3 lower right diagonal)
        D[(3, 3)] = shear;  // τ_xy = 2μ ε̇_xy
        D[(4, 4)] = shear;  // τ_yz = 2μ ε̇_yz
        D[(5, 5)] = shear;  // τ_zx = 2μ ε̇_zx

        D
    }

    /// Returns the dynamic viscosity μ
    pub fn dynamic_viscosity(&self) -> f64 {
        self.viscosity
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
        let expected_diag = 2.0 * mu * (4.0 / 3.0);
        assert_relative_eq!(D[(0, 0)], expected_diag, epsilon = 1e-10);
        assert_relative_eq!(D[(1, 1)], expected_diag, epsilon = 1e-10);
        assert_relative_eq!(D[(2, 2)], expected_diag, epsilon = 1e-10);

        // Check off-diagonal elements (normal-normal coupling)
        let expected_off = 2.0 * mu * (-2.0 / 3.0);
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
}
