/// Boundary condition handling for FEM
///
/// Supports Dirichlet (fixed value) and Neumann (flux) boundary conditions

use nalgebra::Point3;
use std::collections::HashMap;

/// Boundary face identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundaryFace {
    /// Element index
    pub element_id: usize,
    /// Local face index (0-3 for tetrahedra)
    pub local_face_id: usize,
}

/// Neumann boundary condition specification
#[derive(Debug, Clone)]
pub struct NeumannBC {
    /// Prescribed flux value (e.g., heat flux in W/m²)
    /// Positive = outward flux
    pub flux: f64,
}

/// Boundary condition manager
#[derive(Debug, Clone)]
pub struct BoundaryConditions {
    /// Neumann BCs: map from boundary face to flux value
    neumann_bcs: HashMap<BoundaryFace, NeumannBC>,
}

impl BoundaryConditions {
    /// Create new boundary condition manager
    pub fn new() -> Self {
        Self {
            neumann_bcs: HashMap::new(),
        }
    }

    /// Add Neumann BC to a boundary face
    pub fn add_neumann(&mut self, face: BoundaryFace, flux: f64) {
        self.neumann_bcs.insert(face, NeumannBC { flux });
    }

    /// Get Neumann BC for a face, if it exists
    pub fn get_neumann(&self, face: &BoundaryFace) -> Option<&NeumannBC> {
        self.neumann_bcs.get(face)
    }

    /// Get all Neumann boundary faces
    pub fn neumann_faces(&self) -> impl Iterator<Item = (&BoundaryFace, &NeumannBC)> {
        self.neumann_bcs.iter()
    }

    /// Number of Neumann BCs
    pub fn num_neumann_bcs(&self) -> usize {
        self.neumann_bcs.len()
    }
}

impl Default for BoundaryConditions {
    fn default() -> Self {
        Self::new()
    }
}

/// Face of a tet10 element
///
/// Tet10 has 4 faces, each is a 6-node quadratic triangle
#[derive(Debug, Clone)]
pub struct Tet10Face {
    /// Node indices (6 nodes for quadratic triangle)
    /// Ordering: 3 vertices + 3 midside nodes
    pub nodes: [usize; 6],
}

impl Tet10Face {
    /// Get the face nodes for a local face of a tet10 element
    ///
    /// Face numbering for tetrahedron (vertices 0,1,2,3):
    /// - Face 0: vertices (1,2,3), opposite vertex 0
    /// - Face 1: vertices (0,2,3), opposite vertex 1
    /// - Face 2: vertices (0,1,3), opposite vertex 2
    /// - Face 3: vertices (0,1,2), opposite vertex 3 (bottom)
    ///
    /// Tet10 node numbering:
    /// - Vertices: 0,1,2,3
    /// - Edge midpoints: 4(01), 5(12), 6(20), 7(03), 8(13), 9(23)
    pub fn from_element_face(element_nodes: &[usize; 10], local_face: usize) -> Self {
        let nodes = match local_face {
            // Face 0: vertices (1,2,3), edges (12,23,31)
            0 => [
                element_nodes[1],
                element_nodes[2],
                element_nodes[3],
                element_nodes[5], // edge 12
                element_nodes[9], // edge 23
                element_nodes[8], // edge 31
            ],
            // Face 1: vertices (0,2,3), edges (02,23,30)
            1 => [
                element_nodes[0],
                element_nodes[2],
                element_nodes[3],
                element_nodes[6], // edge 20 (reversed)
                element_nodes[9], // edge 23
                element_nodes[7], // edge 03
            ],
            // Face 2: vertices (0,1,3), edges (01,13,30)
            2 => [
                element_nodes[0],
                element_nodes[1],
                element_nodes[3],
                element_nodes[4], // edge 01
                element_nodes[8], // edge 13
                element_nodes[7], // edge 30 (reversed)
            ],
            // Face 3: vertices (0,1,2), edges (01,12,20)
            3 => [
                element_nodes[0],
                element_nodes[1],
                element_nodes[2],
                element_nodes[4], // edge 01
                element_nodes[5], // edge 12
                element_nodes[6], // edge 20
            ],
            _ => panic!("Invalid face index for tet10: {}", local_face),
        };

        Self { nodes }
    }

    /// Compute face area using quadratic triangle integration
    ///
    /// Uses 3-point Gauss quadrature on reference triangle
    pub fn area(&self, node_coords: &[Point3<f64>]) -> f64 {
        // Get coordinates of the 6 face nodes
        let coords: Vec<_> = self.nodes.iter().map(|&i| node_coords[i]).collect();

        // Quadratic triangle shape functions in area coordinates (L1, L2, L3)
        // where L1 + L2 + L3 = 1
        let _shape_funcs = |l1: f64, l2: f64| -> [f64; 6] {
            let l3 = 1.0 - l1 - l2;
            [
                l1 * (2.0 * l1 - 1.0), // vertex 0
                l2 * (2.0 * l2 - 1.0), // vertex 1
                l3 * (2.0 * l3 - 1.0), // vertex 2
                4.0 * l1 * l2,         // edge 01
                4.0 * l2 * l3,         // edge 12
                4.0 * l3 * l1,         // edge 20
            ]
        };

        // Derivatives of shape functions w.r.t. area coordinates
        let shape_derivs = |l1: f64, l2: f64| -> [[f64; 2]; 6] {
            let l3 = 1.0 - l1 - l2;
            [
                [4.0 * l1 - 1.0, 0.0],           // vertex 0 (dN/dL1, dN/dL2)
                [0.0, 4.0 * l2 - 1.0],           // vertex 1
                [-4.0 * l3 + 1.0, -4.0 * l3 + 1.0], // vertex 2
                [4.0 * l2, 4.0 * l1],            // edge 01
                [-4.0 * l2, 4.0 * (l3 - l2)],    // edge 12
                [4.0 * (l3 - l1), -4.0 * l1],    // edge 20
            ]
        };

        // 3-point Gauss quadrature for triangle
        // Weights sum to 0.5 (area of reference triangle in (L1,L2) space)
        let qp = [
            (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0), // (L1, L2, weight)
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        ];

        let mut area = 0.0;

        for &(l1, l2, w) in &qp {
            let derivs = shape_derivs(l1, l2);

            // Compute dx/dL1, dx/dL2 (and similar for y, z)
            let mut dx_dl1 = Point3::new(0.0, 0.0, 0.0);
            let mut dx_dl2 = Point3::new(0.0, 0.0, 0.0);

            for i in 0..6 {
                dx_dl1 += coords[i].coords * derivs[i][0];
                dx_dl2 += coords[i].coords * derivs[i][1];
            }

            // Surface element: |dx/dL1 × dx/dL2|
            let cross = dx_dl1.coords.cross(&dx_dl2.coords);
            let det_j = cross.norm();

            area += w * det_j;
        }

        area
    }

    /// Integrate a constant flux over the face
    ///
    /// Returns nodal contributions to load vector
    pub fn integrate_flux(&self, node_coords: &[Point3<f64>], flux: f64) -> [f64; 6] {
        // Get coordinates of the 6 face nodes
        let coords: Vec<_> = self.nodes.iter().map(|&i| node_coords[i]).collect();

        // Quadratic triangle shape functions
        let shape_funcs = |l1: f64, l2: f64| -> [f64; 6] {
            let l3 = 1.0 - l1 - l2;
            [
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                l3 * (2.0 * l3 - 1.0),
                4.0 * l1 * l2,
                4.0 * l2 * l3,
                4.0 * l3 * l1,
            ]
        };

        // Derivatives for Jacobian
        let shape_derivs = |l1: f64, l2: f64| -> [[f64; 2]; 6] {
            let l3 = 1.0 - l1 - l2;
            [
                [4.0 * l1 - 1.0, 0.0],
                [0.0, 4.0 * l2 - 1.0],
                [-4.0 * l3 + 1.0, -4.0 * l3 + 1.0],
                [4.0 * l2, 4.0 * l1],
                [-4.0 * l2, 4.0 * (l3 - l2)],
                [4.0 * (l3 - l1), -4.0 * l1],
            ]
        };

        // 3-point Gauss quadrature
        // Weights sum to 0.5 (area of reference triangle in (L1,L2) space)
        let qp = [
            (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        ];

        let mut f_face = [0.0; 6];

        for &(l1, l2, w) in &qp {
            let n = shape_funcs(l1, l2);
            let derivs = shape_derivs(l1, l2);

            // Compute Jacobian
            let mut dx_dl1 = Point3::new(0.0, 0.0, 0.0);
            let mut dx_dl2 = Point3::new(0.0, 0.0, 0.0);

            for i in 0..6 {
                dx_dl1 += coords[i].coords * derivs[i][0];
                dx_dl2 += coords[i].coords * derivs[i][1];
            }

            let cross = dx_dl1.coords.cross(&dx_dl2.coords);
            let det_j = cross.norm();

            // Integrate: f_i = ∫ N_i * flux * dA
            for i in 0..6 {
                f_face[i] += w * n[i] * flux * det_j;
            }
        }

        f_face
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_conditions_neumann() {
        let mut bcs = BoundaryConditions::new();

        let face = BoundaryFace {
            element_id: 0,
            local_face_id: 3,
        };

        bcs.add_neumann(face, 100.0);

        assert_eq!(bcs.num_neumann_bcs(), 1);
        assert!(bcs.get_neumann(&face).is_some());
        assert_eq!(bcs.get_neumann(&face).unwrap().flux, 100.0);
    }

    #[test]
    fn test_tet10_face_extraction() {
        // Example tet10 element with node indices 0-9
        let elem_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        // Face 3 (bottom): vertices 0,1,2
        let face = Tet10Face::from_element_face(&elem_nodes, 3);
        assert_eq!(face.nodes, [0, 1, 2, 4, 5, 6]);
    }

    #[test]
    fn test_triangle_area() {
        // Simple right triangle: (0,0,0), (1,0,0), (0,1,0)
        // With midpoints at (0.5,0,0), (0.5,0.5,0), (0,0.5,0)
        let coords = vec![
            Point3::new(0.0, 0.0, 0.0), // vertex 0
            Point3::new(1.0, 0.0, 0.0), // vertex 1
            Point3::new(0.0, 1.0, 0.0), // vertex 2
            Point3::new(0.5, 0.0, 0.0), // midpoint 01
            Point3::new(0.5, 0.5, 0.0), // midpoint 12
            Point3::new(0.0, 0.5, 0.0), // midpoint 20
        ];

        let face = Tet10Face {
            nodes: [0, 1, 2, 3, 4, 5],
        };

        let area = face.area(&coords);

        // Area should be 0.5 for unit right triangle
        assert!((area - 0.5).abs() < 1e-10, "Area = {}, expected 0.5", area);
    }
}
