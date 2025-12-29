/// Element matrices for linear elasticity
///
/// Implements element stiffness and mass matrices for displacement-based FEM.

use nalgebra::{Point3, SMatrix};
use crate::fem::{GaussQuadrature, Tet10Basis};
use super::{IsotropicElasticity, NewtonianViscosity, MaxwellViscoelasticity, StrainDisplacement};

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
        nodes: &[Point3<f64>; 10],
        material: &IsotropicElasticity,
    ) -> SMatrix<f64, 30, 30> {
        let mut k_elem = SMatrix::<f64, 30, 30>::zeros();

        // Get constitutive matrix (constant for linear isotropic elasticity)
        let d = material.constitutive_matrix();

        // Use 4-point Gauss quadrature (degree 2, sufficient for linear strain)
        let quad = GaussQuadrature::tet_4point();

        // Numerical integration over element volume
        for (qp, weight) in quad.points.iter().zip(quad.weights.iter()) {
            // Compute Jacobian and B matrix at this quadrature point
            let j = Tet10Basis::jacobian(qp, nodes);
            let det_j = j.determinant();
            let b = StrainDisplacement::compute_b_at_point(qp, nodes);

            // Integration weight (includes volume element transformation)
            let w = weight * det_j.abs();

            // Compute B^T D B contribution
            // Strategy: First compute DB = D * B (6×30), then B^T * (DB)
            let db = d * b;
            let bt_db = b.transpose() * db;

            // Accumulate weighted contribution
            k_elem += w * bt_db;
        }

        k_elem
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
        nodes: &[Point3<f64>; 10],
        material: &NewtonianViscosity,
    ) -> SMatrix<f64, 30, 30> {
        let mut k_elem = SMatrix::<f64, 30, 30>::zeros();

        // Get viscous constitutive matrix (relates ε̇ to τ)
        let d = material.constitutive_matrix();

        // 4-point Gauss quadrature (degree 2, sufficient for linear strain-rate)
        let quad = GaussQuadrature::tet_4point();

        // Numerical integration: K_e = ∫ B^T D B dV
        for (qp, weight) in quad.points.iter().zip(quad.weights.iter()) {
            // Compute Jacobian and B matrix at this quadrature point
            let j = Tet10Basis::jacobian(qp, nodes);
            let det_j = j.determinant();
            let b = StrainDisplacement::compute_b_at_point(qp, nodes);

            // Integration weight includes Jacobian determinant
            let w = weight * det_j.abs();

            // Compute B^T D B
            let db = d * b;
            let bt_db = b.transpose() * db;

            // Accumulate weighted contribution
            k_elem += w * bt_db;
        }

        k_elem
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
        nodes: &[Point3<f64>; 10],
        density: f64,
    ) -> SMatrix<f64, 30, 30> {
        let mut m_elem = SMatrix::<f64, 30, 30>::zeros();
        let quad = GaussQuadrature::tet_4point();

        for (qp, weight) in quad.points.iter().zip(quad.weights.iter()) {
            let j = Tet10Basis::jacobian(qp, nodes);
            let det_j = j.determinant().abs();
            let n = Tet10Basis::shape_functions(qp);
            
            let w = weight * det_j * density;

            for i in 0..10 {
                for j in 0..10 {
                    let val = w * n[i] * n[j];
                    m_elem[(3 * i, 3 * j)] += val;
                    m_elem[(3 * i + 1, 3 * j + 1)] += val;
                    m_elem[(3 * i + 2, 3 * j + 2)] += val;
                }
            }
        }
        m_elem
    }

    /// Compute element stiffness and history force for Maxwell viscoelasticity
    ///
    /// Returns:
    /// - K_elem: 30×30 effective stiffness matrix
    /// - f_history: 30×1 nodal force vector from stress history
    ///
    /// # Algorithm
    /// K_elem = ∫ B^T D_eff B dV
    /// f_history = -∫ B^T σ_n dV  (negative because moved to RHS)
    ///
    /// where:
    /// - D_eff = D_elastic / [1 + Δt/τ_M]
    /// - σ_n = stress from previous time step (6×1 Voigt notation)
    /// - B = strain-displacement matrix (6×30)
    ///
    /// # Arguments
    /// * `vertices` - Physical coordinates of 4 element vertices
    /// * `material` - Maxwell material properties
    /// * `dt` - Time step size (seconds)
    /// * `stress_n` - Deviatoric stress from previous time step (6×1)
    ///
    /// # Returns
    /// Tuple of (K_elem, f_history)
    ///
    /// # References
    /// - Malvern, "Introduction to the Mechanics of a Continuous Medium"
    /// - Zienkiewicz & Taylor, "The Finite Element Method", Vol. 2
    #[allow(non_snake_case)]
    pub fn maxwell_viscoelastic(
        nodes: &[Point3<f64>; 10],
        material: &MaxwellViscoelasticity,
        _dt: f64,
        stress_n: &SMatrix<f64, 6, 1>,
    ) -> (SMatrix<f64, 30, 30>, SMatrix<f64, 30, 1>) {
        let mut K_elem = SMatrix::<f64, 30, 30>::zeros();
        let mut f_history = SMatrix::<f64, 30, 1>::zeros();

        // Use ELASTIC constitutive matrix (relaxation handled in stress update)
        // This avoids double-relaxation between assembly and stress update
        let D = material.elastic_matrix();

        // 4-point Gauss quadrature (degree 2, sufficient for linear strain)
        let quad = GaussQuadrature::tet_4point();

        // Numerical integration: K_e = ∫ B^T D_eff B dV, f_h = -∫ B^T σ_n dV
        for (qp, weight) in quad.points.iter().zip(quad.weights.iter()) {
            // Compute Jacobian and B matrix at this quadrature point
            let J = Tet10Basis::jacobian(qp, nodes);
            let det_J = J.determinant();
            let B = StrainDisplacement::compute_b_at_point(qp, nodes);

            // Integration weight includes Jacobian determinant
            let w = weight * det_J.abs();

            // Stiffness: K += w * B^T D B (elastic, relaxation in stress update)
            let DB = D * B;
            let BT_DB = B.transpose() * DB;
            K_elem += w * BT_DB;

            // History force: f_history -= w * B^T σ_n
            // (negative because we move to RHS: K u = f_ext + f_history)
            let BT_sigma = B.transpose() * stress_n;
            f_history -= w * BT_sigma;
        }

        (K_elem, f_history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_unit_tet10() -> [Point3<f64>; 10] {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let v2 = Point3::new(0.0, 1.0, 0.0);
        let v3 = Point3::new(0.0, 0.0, 1.0);

        [
            v0, v1, v2, v3,
            Point3::new(0.5, 0.0, 0.0), // 0-1
            Point3::new(0.5, 0.5, 0.0), // 1-2
            Point3::new(0.0, 0.5, 0.0), // 2-0
            Point3::new(0.0, 0.0, 0.5), // 0-3
            Point3::new(0.5, 0.0, 0.5), // 1-3
            Point3::new(0.0, 0.5, 0.5), // 2-3
        ]
    }

    #[test]
    fn test_stiffness_matrix_symmetry() {
        let nodes = create_unit_tet10();
        let material = IsotropicElasticity::new(100e9, 0.25);

        let K = ElasticityElement::stiffness_matrix(&nodes, &material);

        // Check symmetry: K = K^T
        for i in 0..30 {
            for j in 0..30 {
                assert_relative_eq!(K[(i, j)], K[(j, i)], max_relative = 1e-12, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_stiffness_matrix_dimensions() {
        let nodes = create_unit_tet10();
        let material = IsotropicElasticity::new(100e9, 0.25);

        let K = ElasticityElement::stiffness_matrix(&nodes, &material);

        assert_eq!(K.nrows(), 30);
        assert_eq!(K.ncols(), 30);
    }

    #[test]
    fn test_stiffness_matrix_positive_semidefinite() {
        // K should have 6 zero eigenvalues (rigid body modes)
        // and 24 positive eigenvalues

        let nodes = create_unit_tet10();
        let material = IsotropicElasticity::new(100e9, 0.25);

        let K = ElasticityElement::stiffness_matrix(&nodes, &material);

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
        let nodes = create_unit_tet10();

        let mat1 = IsotropicElasticity::new(100e9, 0.25);
        let mat2 = IsotropicElasticity::new(200e9, 0.25);  // 2x stiffer

        let K1 = ElasticityElement::stiffness_matrix(&nodes, &mat1);
        let K2 = ElasticityElement::stiffness_matrix(&nodes, &mat2);

        // K2 should be approximately 2*K1
        for i in 0..30 {
            for j in 0..30 {
                assert_relative_eq!(K2[(i, j)], 2.0 * K1[(i, j)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_mass_matrix_properties() {
        let nodes = create_unit_tet10();
        let density = 3000.0;

        let M = ElasticityElement::mass_matrix(&nodes, density);

        // Should be non-zero and symmetric
        for i in 0..30 {
            assert!(M[(i, i)] > 0.0, "Diagonal entry {} should be positive", i);
            for j in 0..30 {
                assert_relative_eq!(M[(i, j)], M[(j, i)], epsilon = 1e-10);
            }
        }
    }

    // ========================================================================
    // Viscosity Matrix Tests
    // ========================================================================

    #[test]
    fn test_viscosity_matrix_symmetry() {
        use super::NewtonianViscosity;

        let nodes = create_unit_tet10();
        let material = NewtonianViscosity::new(1000.0);

        let K = ElasticityElement::viscosity_matrix(&nodes, &material);

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

        let nodes = create_unit_tet10();
        let material = NewtonianViscosity::new(1000.0);

        let K = ElasticityElement::viscosity_matrix(&nodes, &material);

        assert_eq!(K.nrows(), 30);
        assert_eq!(K.ncols(), 30);
    }

    #[test]
    fn test_viscosity_matrix_positive_definite() {
        use super::NewtonianViscosity;

        let nodes = create_unit_tet10();
        let material = NewtonianViscosity::new(1000.0);

        let K = ElasticityElement::viscosity_matrix(&nodes, &material);

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

        let nodes = create_unit_tet10();

        let mu1 = 1000.0;
        let mu2 = 2000.0;

        let mat1 = NewtonianViscosity::new(mu1);
        let mat2 = NewtonianViscosity::new(mu2);

        let K1 = ElasticityElement::viscosity_matrix(&nodes, &mat1);
        let K2 = ElasticityElement::viscosity_matrix(&nodes, &mat2);

        // K should scale linearly with viscosity
        for i in 0..30 {
            for j in 0..30 {
                assert_relative_eq!(K2[(i, j)], K1[(i, j)] * 2.0, epsilon = 1e-8);
            }
        }
    }

    // ========================================================================
    // Maxwell Viscoelastic Element Matrix Tests
    // ========================================================================

    #[test]
    fn test_maxwell_element_stiffness_symmetry() {
        use super::MaxwellViscoelasticity;
        use nalgebra::SMatrix;

        let nodes = create_unit_tet10();
        let material = MaxwellViscoelasticity::new(100e9, 0.25, 1e19);
        let dt = 3.16e7;  // 1 year
        let stress_n = SMatrix::<f64, 6, 1>::zeros();

        let (K, _) = ElasticityElement::maxwell_viscoelastic(&nodes, &material, dt, &stress_n);

        // Check symmetry (use relative tolerance for large matrix values)
        for i in 0..30 {
            for j in 0..30 {
                assert_relative_eq!(K[(i, j)], K[(j, i)], max_relative = 1e-12, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_maxwell_element_dimensions() {
        use super::MaxwellViscoelasticity;
        use nalgebra::SMatrix;

        let nodes = create_unit_tet10();
        let material = MaxwellViscoelasticity::new(100e9, 0.25, 1e19);
        let dt = 3.16e7;
        let stress_n = SMatrix::<f64, 6, 1>::zeros();

        let (K, f_hist) = ElasticityElement::maxwell_viscoelastic(&nodes, &material, dt, &stress_n);

        assert_eq!(K.nrows(), 30);
        assert_eq!(K.ncols(), 30);
        assert_eq!(f_hist.nrows(), 30);
        assert_eq!(f_hist.ncols(), 1);
    }

    #[test]
    fn test_maxwell_element_zero_stress_history() {
        use super::MaxwellViscoelasticity;
        use nalgebra::SMatrix;

        let nodes = create_unit_tet10();
        let material = MaxwellViscoelasticity::new(100e9, 0.25, 1e19);
        let dt = 3.16e7;
        let stress_n = SMatrix::<f64, 6, 1>::zeros();

        let (_, f_hist) = ElasticityElement::maxwell_viscoelastic(&nodes, &material, dt, &stress_n);

        // With zero stress history, history force should be zero
        for i in 0..30 {
            assert_relative_eq!(f_hist[i], 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_maxwell_element_elastic_limit() {
        use super::{MaxwellViscoelasticity, IsotropicElasticity};
        use nalgebra::SMatrix;

        let nodes = create_unit_tet10();
        let E = 100e9;
        let nu = 0.25;

        let maxwell = MaxwellViscoelasticity::new(E, nu, 1e19);
        let elastic = IsotropicElasticity::new(E, nu);
        let stress_n = SMatrix::<f64, 6, 1>::zeros();

        // Very small time step (elastic limit)
        let dt_small = 1e-20;
        let (K_maxwell, _) = ElasticityElement::maxwell_viscoelastic(&nodes, &maxwell, dt_small, &stress_n);
        let K_elastic = ElasticityElement::stiffness_matrix(&nodes, &elastic);

        // Should match elastic stiffness
        for i in 0..30 {
            for j in 0..30 {
                assert_relative_eq!(K_maxwell[(i, j)], K_elastic[(i, j)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_maxwell_element_relaxed_limit() {
        use super::MaxwellViscoelasticity;
        use nalgebra::SMatrix;

        let nodes = create_unit_tet10();
        let material = MaxwellViscoelasticity::new(100e9, 0.25, 1e19);
        let tau_M = material.relaxation_time();
        let stress_n = SMatrix::<f64, 6, 1>::zeros();

        // With elastic K formulation, stiffness matrix doesn't change with dt
        // (relaxation happens in stress update, not stiffness assembly)
        let dt_large = 1000.0 * tau_M;
        let (K_relaxed, _) = ElasticityElement::maxwell_viscoelastic(&nodes, &material, dt_large, &stress_n);

        let dt_small = 1e-20;
        let (K_elastic, _) = ElasticityElement::maxwell_viscoelastic(&nodes, &material, dt_small, &stress_n);

        // Stiffness should be the same (elastic) regardless of dt
        use approx::assert_relative_eq;
        assert_relative_eq!(K_relaxed[(0, 0)], K_elastic[(0, 0)], max_relative = 1e-10);
    }

    #[test]
    fn test_maxwell_element_with_nonzero_stress() {
        use super::MaxwellViscoelasticity;
        use nalgebra::SMatrix;

        let nodes = create_unit_tet10();
        let material = MaxwellViscoelasticity::new(100e9, 0.25, 1e19);
        let dt = 3.16e7;

        // Non-zero stress history
        let mut stress_n = SMatrix::<f64, 6, 1>::zeros();
        stress_n[0] = 1e6;  // σ_xx = 1 MPa
        stress_n[3] = 5e5;  // σ_xy = 0.5 MPa

        let (K, f_hist) = ElasticityElement::maxwell_viscoelastic(&nodes, &material, dt, &stress_n);

        // K should still be symmetric
        for i in 0..30 {
            for j in 0..30 {
                assert_relative_eq!(K[(i, j)], K[(j, i)], max_relative = 1e-12, epsilon = 1e-6);
            }
        }

        // History force should be non-zero
        let f_norm: f64 = f_hist.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(f_norm > 0.0, "History force should be non-zero with non-zero stress");
    }

    #[test]
    fn test_maxwell_element_stiffness_scales_with_timestep() {
        use super::MaxwellViscoelasticity;
        use nalgebra::SMatrix;

        let nodes = create_unit_tet10();
        let material = MaxwellViscoelasticity::new(100e9, 0.25, 1e19);
        let tau_M = material.relaxation_time();
        let stress_n = SMatrix::<f64, 6, 1>::zeros();

        let dt1 = tau_M;
        let dt2 = 2.0 * tau_M;

        let (K1, _) = ElasticityElement::maxwell_viscoelastic(&nodes, &material, dt1, &stress_n);
        let (K2, _) = ElasticityElement::maxwell_viscoelastic(&nodes, &material, dt2, &stress_n);

        // With elastic K formulation, stiffness is independent of dt
        // (relaxation handled in stress update, not assembly)
        use approx::assert_relative_eq;
        assert_relative_eq!(K1[(0, 0)], K2[(0, 0)], max_relative = 1e-10);

        // Both should equal elastic stiffness
        let elastic_mat = crate::mechanics::IsotropicElasticity::new(
            material.youngs_modulus,
            material.poisson_ratio
        );
        let K_elastic = ElasticityElement::stiffness_matrix(&nodes, &elastic_mat);
        assert_relative_eq!(K1[(0, 0)], K_elastic[(0, 0)], max_relative = 1e-10);
    }
}
