use nalgebra::{Matrix3, Matrix4, Point3, Vector3, Vector4};

/// Tet10 (10-node quadratic tetrahedral) element basis functions
///
/// Node numbering:
///   Vertices: 0, 1, 2, 3
///   Edge midpoints:
///     4: edge 0-1
///     5: edge 1-2
///     6: edge 2-0
///     7: edge 0-3
///     8: edge 1-3
///     9: edge 2-3
///
/// Reference element (in barycentric coordinates):
///   Node 0: (1, 0, 0, 0) at origin
///   Node 1: (0, 1, 0, 0) at (1, 0, 0)
///   Node 2: (0, 0, 1, 0) at (0, 1, 0)
///   Node 3: (0, 0, 0, 1) at (0, 0, 1)
///
/// Shape functions use barycentric (area/volume) coordinates L0, L1, L2, L3
/// where L0 + L1 + L2 + L3 = 1
pub struct Tet10Basis;

impl Tet10Basis {
    /// Evaluate all 10 shape functions at barycentric coordinates (L0, L1, L2, L3)
    ///
    /// # Arguments
    /// * `L` - Barycentric coordinates [L0, L1, L2, L3] where Σ L_i = 1
    ///
    /// # Returns
    /// Array of 10 shape function values [N0, N1, ..., N9]
    ///
    /// # Shape Functions
    /// Vertices:
    ///   N_i = L_i (2 L_i - 1)  for i = 0,1,2,3
    ///
    /// Edge midpoints:
    ///   N_4 = 4 L_0 L_1
    ///   N_5 = 4 L_1 L_2
    ///   N_6 = 4 L_2 L_0
    ///   N_7 = 4 L_0 L_3
    ///   N_8 = 4 L_1 L_3
    ///   N_9 = 4 L_2 L_3
    #[allow(non_snake_case)]
    pub fn shape_functions(L: &[f64; 4]) -> [f64; 10] {
        let [L0, L1, L2, L3] = *L;

        [
            // Vertex nodes
            L0 * (2.0 * L0 - 1.0), // N0
            L1 * (2.0 * L1 - 1.0), // N1
            L2 * (2.0 * L2 - 1.0), // N2
            L3 * (2.0 * L3 - 1.0), // N3
            // Edge midpoint nodes
            4.0 * L0 * L1, // N4 (edge 0-1)
            4.0 * L1 * L2, // N5 (edge 1-2)
            4.0 * L2 * L0, // N6 (edge 2-0)
            4.0 * L0 * L3, // N7 (edge 0-3)
            4.0 * L1 * L3, // N8 (edge 1-3)
            4.0 * L2 * L3, // N9 (edge 2-3)
        ]
    }

    /// Evaluate shape function derivatives with respect to barycentric coordinates
    ///
    /// # Arguments
    /// * `L` - Barycentric coordinates [L0, L1, L2, L3]
    ///
    /// # Returns
    /// Array of 10 derivative vectors, each with [∂N/∂L0, ∂N/∂L1, ∂N/∂L2, ∂N/∂L3]
    ///
    /// Note: Only 3 are independent since L0 + L1 + L2 + L3 = 1
    #[allow(non_snake_case)]
    pub fn shape_derivatives_barycentric(L: &[f64; 4]) -> [[f64; 4]; 10] {
        let [L0, L1, L2, L3] = *L;

        [
            // Vertex node 0: ∂N0/∂Li
            [4.0 * L0 - 1.0, 0.0, 0.0, 0.0],
            // Vertex node 1: ∂N1/∂Li
            [0.0, 4.0 * L1 - 1.0, 0.0, 0.0],
            // Vertex node 2: ∂N2/∂Li
            [0.0, 0.0, 4.0 * L2 - 1.0, 0.0],
            // Vertex node 3: ∂N3/∂Li
            [0.0, 0.0, 0.0, 4.0 * L3 - 1.0],
            // Edge node 4 (0-1): ∂N4/∂Li
            [4.0 * L1, 4.0 * L0, 0.0, 0.0],
            // Edge node 5 (1-2): ∂N5/∂Li
            [0.0, 4.0 * L2, 4.0 * L1, 0.0],
            // Edge node 6 (2-0): ∂N6/∂Li
            [4.0 * L2, 0.0, 4.0 * L0, 0.0],
            // Edge node 7 (0-3): ∂N7/∂Li
            [4.0 * L3, 0.0, 0.0, 4.0 * L0],
            // Edge node 8 (1-3): ∂N8/∂Li
            [0.0, 4.0 * L3, 0.0, 4.0 * L1],
            // Edge node 9 (2-3): ∂N9/∂Li
            [0.0, 0.0, 4.0 * L3, 4.0 * L2],
        ]
    }

    /// Convert from Cartesian coordinates (x,y,z) to barycentric coordinates (L0,L1,L2,L3)
    ///
    /// # Arguments
    /// * `point` - Point in Cartesian coordinates
    /// * `vertices` - The 4 vertex coordinates of the tetrahedron
    ///
    /// # Returns
    /// Barycentric coordinates [L0, L1, L2, L3] where Σ L_i = 1
    pub fn cartesian_to_barycentric(
        point: &Point3<f64>,
        vertices: &[Point3<f64>; 4],
    ) -> [f64; 4] {
        // Compute 6V where V is the volume of the tetrahedron
        let v0 = &vertices[0];
        let v1 = &vertices[1];
        let v2 = &vertices[2];
        let v3 = &vertices[3];

        // Matrix form: [L0, L1, L2, L3]^T = M^{-1} [1, x, y, z]^T
        // where M = [[1, 1, 1, 1],
        //            [x0, x1, x2, x3],
        //            [y0, y1, y2, y3],
        //            [z0, z1, z2, z3]]

        let m = Matrix4::new(
            1.0, 1.0, 1.0, 1.0,
            v0.x, v1.x, v2.x, v3.x,
            v0.y, v1.y, v2.y, v3.y,
            v0.z, v1.z, v2.z, v3.z,
        );

        // Solve M * L = [1, x, y, z]^T for L
        let rhs = Vector4::new(1.0, point.x, point.y, point.z);

        // Use LU decomposition for stability
        let lu = m.lu();
        #[allow(non_snake_case)]
        let L_vec = lu.solve(&rhs).expect("Singular matrix in barycentric conversion");

        [L_vec[0], L_vec[1], L_vec[2], L_vec[3]]
    }

    /// Evaluate shape functions at Cartesian coordinates
    ///
    /// # Arguments
    /// * `point` - Point in Cartesian coordinates
    /// * `vertices` - The 4 vertex coordinates of the tetrahedron
    ///
    /// # Returns
    /// Array of 10 shape function values
    #[allow(non_snake_case)]
    pub fn shape_functions_cartesian(
        point: &Point3<f64>,
        vertices: &[Point3<f64>; 4],
    ) -> [f64; 10] {
        let L = Self::cartesian_to_barycentric(point, vertices);
        Self::shape_functions(&L)
    }

    /// Compute the Jacobian matrix for transformation from barycentric to Cartesian
    ///
    /// J_ij = ∂x_i / ∂L_j
    ///
    /// # Arguments
    /// * `vertices` - The 4 vertex coordinates [x0, x1, x2, x3]
    ///
    /// # Returns
    /// 3×3 Jacobian matrix (only 3 columns since L0 + L1 + L2 + L3 = 1)
    ///
    /// We use L1, L2, L3 as independent variables (L0 = 1 - L1 - L2 - L3)
    #[allow(non_snake_case)]
    pub fn jacobian(vertices: &[Point3<f64>; 4]) -> Matrix3<f64> {
        let v0 = &vertices[0];
        let v1 = &vertices[1];
        let v2 = &vertices[2];
        let v3 = &vertices[3];

        // x = L0*x0 + L1*x1 + L2*x2 + L3*x3
        // Since L0 = 1 - L1 - L2 - L3:
        // x = (1-L1-L2-L3)*x0 + L1*x1 + L2*x2 + L3*x3
        //   = x0 + L1*(x1-x0) + L2*(x2-x0) + L3*(x3-x0)
        //
        // ∂x/∂L1 = x1 - x0
        // ∂x/∂L2 = x2 - x0
        // ∂x/∂L3 = x3 - x0

        Matrix3::new(
            v1.x - v0.x, v2.x - v0.x, v3.x - v0.x,
            v1.y - v0.y, v2.y - v0.y, v3.y - v0.y,
            v1.z - v0.z, v2.z - v0.z, v3.z - v0.z,
        )
    }

    /// Compute shape function derivatives with respect to Cartesian coordinates
    ///
    /// # Arguments
    /// * `vertices` - The 4 vertex coordinates
    ///
    /// # Returns
    /// Array of 10 derivative vectors [∂N/∂x, ∂N/∂y, ∂N/∂z]
    ///
    /// Uses chain rule: ∂N/∂x = ∂N/∂Li * ∂Li/∂x = ∂N/∂Li * J^{-T}
    #[allow(non_snake_case)]
    pub fn shape_derivatives_cartesian(
        L: &[f64; 4],
        vertices: &[Point3<f64>; 4],
    ) -> [[f64; 3]; 10] {
        // Get derivatives in barycentric coordinates
        let dN_dL = Self::shape_derivatives_barycentric(L);

        // Get Jacobian J = ∂x/∂L
        let J = Self::jacobian(vertices);

        // Compute inverse Jacobian
        let J_inv = J.try_inverse().expect("Singular Jacobian");

        // For each shape function, compute dN/dx = J^{-T} * dN/dL
        // Since we have 4 barycentric coords but only 3 are independent,
        // we use only L1, L2, L3 derivatives (L0 = 1 - L1 - L2 - L3)
        let mut dN_dx = [[0.0; 3]; 10];

        for i in 0..10 {
            // Convert from ∂N/∂L to ∂N/∂x using chain rule
            // ∂N/∂L0 = -∂N/∂x·(∂L0/∂x) = -∂N/∂x·(-1,-1,-1)
            // where the gradient of barycentric coords comes from J^{-T}

            // dN_dL[i] = [∂N_i/∂L0, ∂N_i/∂L1, ∂N_i/∂L2, ∂N_i/∂L3]
            // We need to account for constraint L0 + L1 + L2 + L3 = 1

            // ∂N/∂x = ∂N/∂L1 * ∂L1/∂x + ∂N/∂L2 * ∂L2/∂x + ∂N/∂L3 * ∂L3/∂x
            //       = (∂N/∂L1 - ∂N/∂L0) * ∂L1/∂x + (∂N/∂L2 - ∂N/∂L0) * ∂L2/∂x + (∂N/∂L3 - ∂N/∂L0) * ∂L3/∂x

            let dN_dL1 = dN_dL[i][1] - dN_dL[i][0];
            let dN_dL2 = dN_dL[i][2] - dN_dL[i][0];
            let dN_dL3 = dN_dL[i][3] - dN_dL[i][0];

            // Multiply by J^{-T}: [∂L1/∂x, ∂L2/∂x, ∂L3/∂x] = J^{-T}
            let grad = J_inv.transpose() * Vector3::new(dN_dL1, dN_dL2, dN_dL3);

            dN_dx[i] = [grad[0], grad[1], grad[2]];
        }

        dN_dx
    }

    /// Compute element volume (6V where V is tetrahedron volume)
    ///
    /// # Arguments
    /// * `vertices` - The 4 vertex coordinates
    ///
    /// # Returns
    /// Volume of the tetrahedron
    #[allow(non_snake_case)]
    pub fn element_volume(vertices: &[Point3<f64>; 4]) -> f64 {
        let J = Self::jacobian(vertices);
        J.determinant().abs() / 6.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_partition_of_unity() {
        // Shape functions should sum to 1 at any point
        let test_points = [
            [0.25, 0.25, 0.25, 0.25], // Center
            [1.0, 0.0, 0.0, 0.0],     // Vertex 0
            [0.0, 1.0, 0.0, 0.0],     // Vertex 1
            [0.5, 0.5, 0.0, 0.0],     // Edge 0-1
            [0.1, 0.2, 0.3, 0.4],     // Arbitrary point
        ];

        for L in &test_points {
            let N = Tet10Basis::shape_functions(L);
            let sum: f64 = N.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_kronecker_delta() {
        // N_i should be 1 at node i and 0 at other vertex nodes
        let vertices_L = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        for (i, L) in vertices_L.iter().enumerate() {
            let N = Tet10Basis::shape_functions(L);
            assert_relative_eq!(N[i], 1.0, epsilon = 1e-14);
            for j in 0..4 {
                if i != j {
                    assert_relative_eq!(N[j], 0.0, epsilon = 1e-14);
                }
            }
        }
    }

    #[test]
    fn test_edge_midpoints() {
        // Edge nodes should be 1 at their midpoint, 0 at vertices
        let edge_points = [
            ([0.5, 0.5, 0.0, 0.0], 4), // Edge 0-1
            ([0.0, 0.5, 0.5, 0.0], 5), // Edge 1-2
            ([0.5, 0.0, 0.5, 0.0], 6), // Edge 2-0
            ([0.5, 0.0, 0.0, 0.5], 7), // Edge 0-3
            ([0.0, 0.5, 0.0, 0.5], 8), // Edge 1-3
            ([0.0, 0.0, 0.5, 0.5], 9), // Edge 2-3
        ];

        for (L, edge_node) in &edge_points {
            let N = Tet10Basis::shape_functions(L);
            assert_relative_eq!(N[*edge_node], 1.0, epsilon = 1e-14);
            // Vertex nodes should be 0 at edge midpoints
            for i in 0..4 {
                assert_relative_eq!(N[i], 0.0, epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_reference_element_volume() {
        // Reference tetrahedron: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        let vertices = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];

        let volume = Tet10Basis::element_volume(&vertices);
        // Volume of reference tet is 1/6
        assert_relative_eq!(volume, 1.0 / 6.0, epsilon = 1e-14);
    }

    #[test]
    fn test_barycentric_conversion() {
        // Reference tetrahedron
        let vertices = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];

        // Test center point
        let center = Point3::new(0.25, 0.25, 0.25);
        let L = Tet10Basis::cartesian_to_barycentric(&center, &vertices);

        // All coordinates should be equal for center
        for &Li in &L {
            assert_relative_eq!(Li, 0.25, epsilon = 1e-12);
        }

        // Test vertex 1
        let v1 = Point3::new(1.0, 0.0, 0.0);
        let L = Tet10Basis::cartesian_to_barycentric(&v1, &vertices);
        assert_relative_eq!(L[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(L[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(L[2], 0.0, epsilon = 1e-12);
        assert_relative_eq!(L[3], 0.0, epsilon = 1e-12);
    }
}
