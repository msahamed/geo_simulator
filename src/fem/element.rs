use nalgebra::{Point3, SMatrix};
use crate::fem::{Tet10Basis, GaussQuadrature};

/// Element matrix computation for Tet10 elements
pub struct ElementMatrix;

impl ElementMatrix {
    /// Compute element stiffness matrix for thermal diffusion
    ///
    /// K_e = ∫ (∇N)^T k (∇N) dV
    ///
    /// where:
    /// - N is the shape function matrix
    /// - k is the thermal conductivity (scalar)
    /// - Integration over element volume
    ///
    /// # Arguments
    /// * `vertices` - The 4 vertex coordinates of the element
    /// * `conductivity` - Thermal conductivity k (W/m·K)
    ///
    /// # Returns
    /// 10×10 element stiffness matrix (symmetric)
    pub fn thermal_stiffness(
        vertices: &[Point3<f64>; 4],
        conductivity: f64,
    ) -> SMatrix<f64, 10, 10> {
        let mut k_elem = SMatrix::<f64, 10, 10>::zeros();

        // Use 4-point quadrature (degree 2, sufficient for linear gradients)
        let quad = GaussQuadrature::tet_4point();

        // Numerical integration
        #[allow(non_snake_case)]
        for (qp, weight) in quad.points.iter().zip(quad.weights.iter()) {
            // Evaluate shape function derivatives at quadrature point
            let dN_dx = Tet10Basis::shape_derivatives_cartesian(qp, vertices);

            // Compute Jacobian determinant for volume element
            let J = Tet10Basis::jacobian(vertices);
            let det_J = J.determinant();

            // Integration weight (includes volume element)
            let w = weight * det_J.abs();

            // Assemble B^T * k * B
            // For thermal: B = ∇N (gradient of shape functions)
            // K_ij = ∫ (∇N_i) · k · (∇N_j) dV
            for i in 0..10 {
                for j in 0..10 {
                    // Dot product of gradients
                    let grad_dot = dN_dx[i][0] * dN_dx[j][0]
                        + dN_dx[i][1] * dN_dx[j][1]
                        + dN_dx[i][2] * dN_dx[j][2];

                    k_elem[(i, j)] += conductivity * grad_dot * w;
                }
            }
        }

        k_elem
    }

    /// Compute element mass matrix for thermal problems
    ///
    /// M_e = ∫ ρ c_p N^T N dV
    ///
    /// # Arguments
    /// * `vertices` - The 4 vertex coordinates
    /// * `density` - Material density ρ (kg/m³)
    /// * `specific_heat` - Specific heat capacity c_p (J/kg·K)
    ///
    /// # Returns
    /// 10×10 element mass matrix (symmetric)
    pub fn thermal_mass(
        vertices: &[Point3<f64>; 4],
        density: f64,
        specific_heat: f64,
    ) -> SMatrix<f64, 10, 10> {
        let mut m_elem = SMatrix::<f64, 10, 10>::zeros();

        // Use 5-point quadrature for better accuracy with quadratic functions
        let quad = GaussQuadrature::tet_5point();

        let rho_cp = density * specific_heat;

        #[allow(non_snake_case)]
        for (qp, weight) in quad.points.iter().zip(quad.weights.iter()) {
            // Evaluate shape functions
            let N = Tet10Basis::shape_functions(qp);

            // Jacobian determinant
            let J = Tet10Basis::jacobian(vertices);
            let det_J = J.determinant();
            let w = weight * det_J.abs();

            // M_ij = ∫ ρ c_p N_i N_j dV
            for i in 0..10 {
                for j in 0..10 {
                    m_elem[(i, j)] += rho_cp * N[i] * N[j] * w;
                }
            }
        }

        m_elem
    }

    /// Compute element load vector for constant source term
    ///
    /// f_e = ∫ N^T Q dV
    ///
    /// # Arguments
    /// * `vertices` - The 4 vertex coordinates
    /// * `source` - Source term Q (e.g., heat generation W/m³)
    ///
    /// # Returns
    /// 10×1 element load vector
    pub fn thermal_load(
        vertices: &[Point3<f64>; 4],
        source: f64,
    ) -> [f64; 10] {
        let mut f_elem = [0.0; 10];

        let quad = GaussQuadrature::tet_4point();

        #[allow(non_snake_case)]
        for (qp, weight) in quad.points.iter().zip(quad.weights.iter()) {
            let N = Tet10Basis::shape_functions(qp);

            let J = Tet10Basis::jacobian(vertices);
            let det_J = J.determinant();
            let w = weight * det_J.abs();

            // f_i = ∫ N_i Q dV
            for i in 0..10 {
                f_elem[i] += N[i] * source * w;
            }
        }

        f_elem
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_stiffness_symmetry() {
        // Reference tetrahedron
        let vertices = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];

        let k = 1.0; // conductivity
        let K = ElementMatrix::thermal_stiffness(&vertices, k);

        // Check symmetry
        for i in 0..10 {
            for j in 0..10 {
                assert_relative_eq!(K[(i, j)], K[(j, i)], epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_stiffness_positive_definiteness() {
        let vertices = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];

        let k = 1.0;
        let K = ElementMatrix::thermal_stiffness(&vertices, k);

        // Check positive definiteness by checking diagonal entries
        // For a well-formed element, diagonal should be positive
        for i in 0..10 {
            assert!(K[(i, i)] > 0.0, "Diagonal entry {} is not positive", i);
        }

        // Sum of each row should be close to zero (for constant field)
        for i in 0..10 {
            let row_sum: f64 = (0..10).map(|j| K[(i, j)]).sum();
            assert_relative_eq!(row_sum, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_mass_matrix_symmetry() {
        let vertices = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];

        let M = ElementMatrix::thermal_mass(&vertices, 1.0, 1.0);

        // Check symmetry
        for i in 0..10 {
            for j in 0..10 {
                assert_relative_eq!(M[(i, j)], M[(j, i)], epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_load_vector_sum() {
        let vertices = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];

        let Q = 6.0; // source term
        let f = ElementMatrix::thermal_load(&vertices, Q);

        // Sum should equal Q * volume
        let volume = Tet10Basis::element_volume(&vertices);
        let sum: f64 = f.iter().sum();

        assert_relative_eq!(sum, Q * volume, epsilon = 1e-12);
    }
}
