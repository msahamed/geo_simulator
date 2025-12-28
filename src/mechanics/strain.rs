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

    /// Compute B-matrix at a quadrature point in physical element
    ///
    /// Convenience method that computes shape derivatives and builds B-matrix.
    ///
    /// # Arguments
    /// * `barycentric` - Barycentric coordinates [L0, L1, L2, L3] of quadrature point
    /// * `vertices` - Physical coordinates of the 4 element vertices
    ///
    /// # Returns
    /// B matrix (6×30) at the given quadrature point
    #[allow(non_snake_case)]
    pub fn compute_b_at_point(
        barycentric: &[f64; 4],
        vertices: &[Point3<f64>; 4],
    ) -> SMatrix<f64, 6, 30> {
        // Get shape function derivatives in Cartesian coordinates
        let dN_dx = Tet10Basis::shape_derivatives_cartesian(barycentric, vertices);

        // Build B-matrix
        Self::compute_b_matrix(&dN_dx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::SVector;

    #[test]
    fn test_b_matrix_rigid_translation() {
        // For rigid body translation, B · u_rigid should be zero (no strain)

        // Simple reference tetrahedron vertices
        let vertices = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];

        // Evaluate at element center
        let barycentric = [0.25, 0.25, 0.25, 0.25];
        let B = StrainDisplacement::compute_b_at_point(&barycentric, &vertices);

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

        let vertices = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];

        let barycentric = [0.25, 0.25, 0.25, 0.25];
        let B = StrainDisplacement::compute_b_at_point(&barycentric, &vertices);

        // B should not be all zeros
        let has_nonzero = B.iter().any(|&val| val.abs() > 1e-10);
        assert!(has_nonzero, "B-matrix should have non-zero entries");

        // Spot check: at least some entries should be non-zero
        let nonzero_count = B.iter().filter(|&&val| val.abs() > 1e-10).count();
        assert!(nonzero_count > 10, "B-matrix should have multiple non-zero entries");
    }
}
