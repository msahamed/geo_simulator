/// Strain-displacement relationships for solid mechanics
///
/// Implements the B-matrix that relates nodal displacements to element strains.

use nalgebra::{Point3, SMatrix};
use crate::fem::Tet10Basis;

/// Strain-displacement matrix computations
pub struct StrainDisplacement;

impl StrainDisplacement {
    /// Compute 6×30 strain-displacement matrix B from shape function derivatives
    ///
    /// Relates nodal displacements to element strains: ε = B · u_e
    ///
    /// # Arguments
    /// * `dN_dx` - Shape function derivatives [∂N_i/∂x, ∂N_i/∂y, ∂N_i/∂z] for i=0..9
    ///
    /// # Returns
    /// B matrix (6×30) where:
    /// - Rows: [ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_zx] (Voigt notation)
    /// - Columns: [u_0x, u_0y, u_0z, u_1x, ..., u_9z] (30 DOFs)
    ///
    /// For each node i, columns 3i, 3i+1, 3i+2 are:
    /// ```text
    ///     [∂N_i/∂x    0         0      ]   (ε_xx = ∂u_x/∂x)
    ///     [  0      ∂N_i/∂y     0      ]   (ε_yy = ∂u_y/∂y)
    ///     [  0        0      ∂N_i/∂z   ]   (ε_zz = ∂u_z/∂z)
    ///     [∂N_i/∂y  ∂N_i/∂x     0      ]   (γ_xy = ∂u_x/∂y + ∂u_y/∂x)
    ///     [  0      ∂N_i/∂z  ∂N_i/∂y   ]   (γ_yz = ∂u_y/∂z + ∂u_z/∂y)
    ///     [∂N_i/∂z    0      ∂N_i/∂x   ]   (γ_zx = ∂u_z/∂x + ∂u_x/∂z)
    /// ```
    ///
    /// # References
    /// - Zienkiewicz & Taylor, "The Finite Element Method", Vol. 1, Ch. 6
    /// - Bathe, "Finite Element Procedures", Section 6.2
    #[allow(non_snake_case)]
    pub fn compute_b_matrix(dN_dx: &[[f64; 3]; 10]) -> SMatrix<f64, 6, 30> {
        let mut B = SMatrix::<f64, 6, 30>::zeros();

        for i in 0..10 {
            let col_base = 3 * i;

            let dNi_dx = dN_dx[i][0];  // ∂N_i/∂x
            let dNi_dy = dN_dx[i][1];  // ∂N_i/∂y
            let dNi_dz = dN_dx[i][2];  // ∂N_i/∂z

            // Row 0: ε_xx = ∂u_x/∂x
            B[(0, col_base + 0)] = dNi_dx;

            // Row 1: ε_yy = ∂u_y/∂y
            B[(1, col_base + 1)] = dNi_dy;

            // Row 2: ε_zz = ∂u_z/∂z
            B[(2, col_base + 2)] = dNi_dz;

            // Row 3: γ_xy = ∂u_x/∂y + ∂u_y/∂x
            B[(3, col_base + 0)] = dNi_dy;
            B[(3, col_base + 1)] = dNi_dx;

            // Row 4: γ_yz = ∂u_y/∂z + ∂u_z/∂y
            B[(4, col_base + 1)] = dNi_dz;
            B[(4, col_base + 2)] = dNi_dy;

            // Row 5: γ_zx = ∂u_z/∂x + ∂u_x/∂z
            B[(5, col_base + 2)] = dNi_dx;
            B[(5, col_base + 0)] = dNi_dz;
        }

        B
    }

    /// Compute B-matrix at a quadrature point in physical element (Isoparametric)
    ///
    /// # Arguments
    /// * `barycentric` - Barycentric coordinates [L0, L1, L2, L3]
    /// * `nodes` - All 10 node coordinates
    #[allow(non_snake_case)]
    pub fn compute_b_at_point(
        barycentric: &[f64; 4],
        nodes: &[Point3<f64>; 10],
    ) -> SMatrix<f64, 6, 30> {
        let dN_dx = Tet10Basis::shape_derivatives_cartesian(barycentric, nodes);
        Self::compute_b_matrix(&dN_dx)
    }

    /// Compute B-matrix using linear mapping (Faster, accurate for straight-edged elements)
    ///
    /// # Arguments
    /// * `barycentric` - Barycentric coordinates
    /// * `vertices` - Only the 4 corner vertices
    #[allow(non_snake_case)]
    pub fn compute_b_at_point_linear(
        barycentric: &[f64; 4],
        vertices: &[Point3<f64>; 4],
    ) -> SMatrix<f64, 6, 30> {
        let dN_dL = Tet10Basis::shape_derivatives_barycentric(barycentric);
        let J = Tet10Basis::jacobian_linear(vertices);
        let J_inv = J.try_inverse().expect("Singular linear Jacobian");

        let mut dN_dx = [[0.0; 3]; 10];
        for i in 0..10 {
            let dN_dL1 = dN_dL[i][1] - dN_dL[i][0];
            let dN_dL2 = dN_dL[i][2] - dN_dL[i][0];
            let dN_dL3 = dN_dL[i][3] - dN_dL[i][0];
            let grad = J_inv.transpose() * nalgebra::Vector3::new(dN_dL1, dN_dL2, dN_dL3);
            dN_dx[i] = [grad[0], grad[1], grad[2]];
        }
        Self::compute_b_matrix(&dN_dx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::SVector;

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
    fn test_b_matrix_rigid_translation() {
        // For rigid body translation, B · u_rigid should be zero (no strain)

        // Simple reference tetrahedron vertices
        let nodes = create_unit_tet10();

        // Evaluate at element center
        let barycentric = [0.25, 0.25, 0.25, 0.25];
        let B = StrainDisplacement::compute_b_at_point(&barycentric, &nodes);

        // Rigid translation in x-direction: all nodes move by [1, 0, 0]
        let mut u_rigid = SVector::<f64, 30>::zeros();
        for i in 0..10 {
            u_rigid[3 * i] = 1.0;  // u_x = 1 for all nodes
        }

        // Compute strain: should be zero
        let strain = B * u_rigid;

        for i in 0..6 {
            assert_relative_eq!(strain[i], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_b_matrix_dimensions() {
        let dN_dx = [[1.0, 0.0, 0.0]; 10];  // Dummy derivatives
        let B = StrainDisplacement::compute_b_matrix(&dN_dx);

        assert_eq!(B.nrows(), 6);
        assert_eq!(B.ncols(), 30);
    }

    #[test]
    fn test_b_matrix_structure() {
        // Check that B has correct sparsity pattern

        let dN_dx = [
            [1.0, 2.0, 3.0],  // Node 0
            [0.0; 3],         // Nodes 1-9 (zeros)
            [0.0; 3],
            [0.0; 3],
            [0.0; 3],
            [0.0; 3],
            [0.0; 3],
            [0.0; 3],
            [0.0; 3],
            [0.0; 3],
        ];

        let B = StrainDisplacement::compute_b_matrix(&dN_dx);

        // Only first 3 columns (node 0) should be non-zero

        // ε_xx row: column 0 should have dN0/dx = 1.0
        assert_relative_eq!(B[(0, 0)], 1.0, epsilon = 1e-10);

        // ε_yy row: column 1 should have dN0/dy = 2.0
        assert_relative_eq!(B[(1, 1)], 2.0, epsilon = 1e-10);

        // ε_zz row: column 2 should have dN0/dz = 3.0
        assert_relative_eq!(B[(2, 2)], 3.0, epsilon = 1e-10);

        // γ_xy row: columns 0 and 1
        assert_relative_eq!(B[(3, 0)], 2.0, epsilon = 1e-10);  // dN0/dy
        assert_relative_eq!(B[(3, 1)], 1.0, epsilon = 1e-10);  // dN0/dx

        // γ_yz row: columns 1 and 2
        assert_relative_eq!(B[(4, 1)], 3.0, epsilon = 1e-10);  // dN0/dz
        assert_relative_eq!(B[(4, 2)], 2.0, epsilon = 1e-10);  // dN0/dy

        // γ_zx row: columns 0 and 2
        assert_relative_eq!(B[(5, 0)], 3.0, epsilon = 1e-10);  // dN0/dz
        assert_relative_eq!(B[(5, 2)], 1.0, epsilon = 1e-10);  // dN0/dx

        // All other columns should be zero
        for col in 3..30 {
            for row in 0..6 {
                assert_relative_eq!(B[(row, col)], 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_b_matrix_rank() {
        // B-matrix should have full rank (6) for non-degenerate element

        let nodes = create_unit_tet10();

        let barycentric = [0.25, 0.25, 0.25, 0.25];
        let B = StrainDisplacement::compute_b_at_point(&barycentric, &nodes);

        // B should not be all zeros
        let has_nonzero = B.iter().any(|&val| val.abs() > 1e-10);
        assert!(has_nonzero, "B-matrix should have non-zero entries");

        // Spot check: at least some entries should be non-zero
        let nonzero_count = B.iter().filter(|&&val| val.abs() > 1e-10).count();
        assert!(nonzero_count > 10, "B-matrix should have multiple non-zero entries");
    }
}
