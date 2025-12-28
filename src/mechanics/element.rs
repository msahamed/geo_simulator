/// Element matrices for linear elasticity
///
/// Implements element stiffness and mass matrices for displacement-based FEM.

use nalgebra::{Point3, SMatrix};
use crate::fem::{GaussQuadrature, Tet10Basis};
use super::{IsotropicElasticity, NewtonianViscosity, StrainDisplacement};

/// Element matrix computations for linear elasticity
pub struct ElasticityElement;

impl ElasticityElement {
    /// Compute element stiffness matrix for linear elasticity
    ///
    /// K_e = ∫ B^T D B dV
    ///
    /// where:
    /// - B is the strain-displacement matrix (6×30)
    /// - D is the constitutive matrix (6×6)
    /// - Integration over element volume using Gaussian quadrature
    ///
    /// # Arguments
    /// * `vertices` - Physical coordinates of the 4 element vertices
    /// * `material` - Elastic material properties (E, ν)
    ///
    /// # Returns
    /// 30×30 symmetric element stiffness matrix
    ///
    /// # References
    /// - Zienkiewicz & Taylor, "The Finite Element Method", Vol. 1
    /// - Bathe, "Finite Element Procedures", Ch. 6
    #[allow(non_snake_case)]
    pub fn stiffness_matrix(
        vertices: &[Point3<f64>; 4],
        material: &IsotropicElasticity,
    ) -> SMatrix<f64, 30, 30> {
        let mut K_elem = SMatrix::<f64, 30, 30>::zeros();

        // Get constitutive matrix (constant for linear isotropic elasticity)
        let D = material.constitutive_matrix();

        // Use 4-point Gauss quadrature (degree 2, sufficient for linear strain)
        let quad = GaussQuadrature::tet_4point();

        // Compute Jacobian determinant (constant for Tet10 with linear geometry)
        let J = Tet10Basis::jacobian(vertices);
        let det_J = J.determinant();

        // Numerical integration over element volume
        for (qp, weight) in quad.points.iter().zip(quad.weights.iter()) {
            // Compute B matrix at this quadrature point
            let B = StrainDisplacement::compute_b_at_point(qp, vertices);

            // Integration weight (includes volume element transformation)
            let w = weight * det_J.abs();

            // Compute B^T D B contribution
            // Strategy: First compute DB = D * B (6×30), then B^T * (DB)
            let DB = D * B;
            let BT_DB = B.transpose() * DB;

            // Accumulate weighted contribution
            K_elem += w * BT_DB;
        }

        K_elem
    }

    /// Compute element viscosity matrix for viscous flow
    ///
    /// K_e^viscous = ∫ B^T D_viscous B dV
    ///
    /// Where:
    /// - B is the 6×30 strain-rate-velocity matrix (same as elasticity B-matrix)
    /// - D_viscous is the 6×6 viscous constitutive matrix relating stress to strain-rate
    /// - Integration uses 4-point Gauss quadrature
    ///
    /// This matrix relates velocity DOFs to forces for steady-state Stokes flow.
    /// The formulation is identical to elasticity, just with different constitutive relation:
    /// - Elasticity: σ = D_elastic ε  (stress ∝ strain)
    /// - Viscous: τ = D_viscous ε̇     (stress ∝ strain-rate)
    ///
    /// # Arguments
    /// * `vertices` - Physical coordinates of the 4 element vertices
    /// * `material` - Newtonian viscosity material properties
    ///
    /// # Returns
    /// 30×30 element viscosity matrix
    pub fn viscosity_matrix(
        vertices: &[Point3<f64>; 4],
        material: &NewtonianViscosity,
    ) -> SMatrix<f64, 30, 30> {
        let mut K_elem = SMatrix::<f64, 30, 30>::zeros();

        // Get viscous constitutive matrix (relates ε̇ to τ)
        let D = material.constitutive_matrix();

        // 4-point Gauss quadrature (degree 2, sufficient for linear strain-rate)
        let quad = GaussQuadrature::tet_4point();

        // Compute Jacobian and determinant (constant for linear tets)
        let J = Tet10Basis::jacobian(vertices);
        let det_J = J.determinant();

        // Numerical integration: K_e = ∫ B^T D B dV
        for (qp, weight) in quad.points.iter().zip(quad.weights.iter()) {
            // Compute B matrix (strain-rate-velocity relation)
            let B = StrainDisplacement::compute_b_at_point(qp, vertices);

            // Integration weight includes Jacobian determinant
            let w = weight * det_J.abs();

            // Compute B^T D B
            let DB = D * B;
            let BT_DB = B.transpose() * DB;

            // Accumulate weighted contribution
            K_elem += w * BT_DB;
        }

        K_elem
    }

    /// Compute element mass matrix for dynamics (placeholder)
    ///
    /// M_e = ∫ ρ N^T N dV
    ///
    /// Currently returns zeros - will be implemented for transient problems.
    ///
    /// # Arguments
    /// * `vertices` - Physical coordinates of the 4 element vertices
    /// * `density` - Material density (kg/m³)
    ///
    /// # Returns
    /// 30×30 element mass matrix (currently zero for quasi-static)
    #[allow(unused_variables)]
    pub fn mass_matrix(
        vertices: &[Point3<f64>; 4],
        density: f64,
    ) -> SMatrix<f64, 30, 30> {
        // TODO: Implement for transient dynamics (Milestone 2.2+)
        // For now, only quasi-static problems supported
        SMatrix::<f64, 30, 30>::zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_reference_tet() -> [Point3<f64>; 4] {
        // Unit tetrahedron
        [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ]
    }

    #[test]
    fn test_stiffness_matrix_symmetry() {
        let vertices = create_reference_tet();
        let material = IsotropicElasticity::new(100e9, 0.25);

        let K = ElasticityElement::stiffness_matrix(&vertices, &material);

        // Check symmetry: K = K^T
        for i in 0..30 {
            for j in 0..30 {
                assert_relative_eq!(K[(i, j)], K[(j, i)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_stiffness_matrix_dimensions() {
        let vertices = create_reference_tet();
        let material = IsotropicElasticity::new(100e9, 0.25);

        let K = ElasticityElement::stiffness_matrix(&vertices, &material);

        assert_eq!(K.nrows(), 30);
        assert_eq!(K.ncols(), 30);
    }

    #[test]
    fn test_stiffness_matrix_positive_semidefinite() {
        // K should have 6 zero eigenvalues (rigid body modes)
        // and 24 positive eigenvalues

        let vertices = create_reference_tet();
        let material = IsotropicElasticity::new(100e9, 0.25);

        let K = ElasticityElement::stiffness_matrix(&vertices, &material);

        // Compute eigenvalues
        let eigen = K.symmetric_eigen();
        let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // First 6 eigenvalues should be very small (rigid body modes)
        for i in 0..6 {
            assert!(eigenvalues[i].abs() < 1e3,
                "Eigenvalue {} = {} should be near zero (rigid mode)", i, eigenvalues[i]);
        }

        // Remaining eigenvalues should be positive
        for i in 6..30 {
            assert!(eigenvalues[i] > 1e6,
                "Eigenvalue {} = {} should be positive", i, eigenvalues[i]);
        }
    }

    #[test]
    fn test_stiffness_scales_with_youngs_modulus() {
        let vertices = create_reference_tet();

        let mat1 = IsotropicElasticity::new(100e9, 0.25);
        let mat2 = IsotropicElasticity::new(200e9, 0.25);  // 2x stiffer

        let K1 = ElasticityElement::stiffness_matrix(&vertices, &mat1);
        let K2 = ElasticityElement::stiffness_matrix(&vertices, &mat2);

        // K2 should be approximately 2*K1
        for i in 0..30 {
            for j in 0..30 {
                assert_relative_eq!(K2[(i, j)], 2.0 * K1[(i, j)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_mass_matrix_placeholder() {
        let vertices = create_reference_tet();
        let density = 3000.0;

        let M = ElasticityElement::mass_matrix(&vertices, density);

        // Should be zeros for now (quasi-static only)
        for i in 0..30 {
            for j in 0..30 {
                assert_eq!(M[(i, j)], 0.0);
            }
        }
    }

    // ========================================================================
    // Viscosity Matrix Tests
    // ========================================================================

    #[test]
    fn test_viscosity_matrix_symmetry() {
        use super::NewtonianViscosity;

        let vertices = create_reference_tet();
        let material = NewtonianViscosity::new(1000.0);

        let K = ElasticityElement::viscosity_matrix(&vertices, &material);

        // Check symmetry
        for i in 0..30 {
            for j in 0..30 {
                assert_relative_eq!(K[(i, j)], K[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_viscosity_matrix_dimensions() {
        use super::NewtonianViscosity;

        let vertices = create_reference_tet();
        let material = NewtonianViscosity::new(1000.0);

        let K = ElasticityElement::viscosity_matrix(&vertices, &material);

        assert_eq!(K.nrows(), 30);
        assert_eq!(K.ncols(), 30);
    }

    #[test]
    fn test_viscosity_matrix_positive_definite() {
        use super::NewtonianViscosity;

        let vertices = create_reference_tet();
        let material = NewtonianViscosity::new(1000.0);

        let K = ElasticityElement::viscosity_matrix(&vertices, &material);

        // Compute eigenvalues
        let eigen = K.symmetric_eigen();
        let eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();

        // For viscous flow, all eigenvalues should be non-negative
        // (Unlike elasticity, no rigid body modes for velocity field)
        for (i, &eig) in eigenvalues.iter().enumerate() {
            assert!(eig >= -1e-6, "Eigenvalue {} = {} should be non-negative", i, eig);
        }
    }

    #[test]
    fn test_viscosity_matrix_scales_with_viscosity() {
        use super::NewtonianViscosity;

        let vertices = create_reference_tet();

        let mu1 = 1000.0;
        let mu2 = 2000.0;

        let mat1 = NewtonianViscosity::new(mu1);
        let mat2 = NewtonianViscosity::new(mu2);

        let K1 = ElasticityElement::viscosity_matrix(&vertices, &mat1);
        let K2 = ElasticityElement::viscosity_matrix(&vertices, &mat2);

        // K should scale linearly with viscosity
        for i in 0..30 {
            for j in 0..30 {
                assert_relative_eq!(K2[(i, j)], K1[(i, j)] * 2.0, epsilon = 1e-8);
            }
        }
    }
}
