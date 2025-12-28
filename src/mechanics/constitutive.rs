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
}
