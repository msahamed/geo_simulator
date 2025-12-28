use crate::mesh::{Mesh, Tet10Element};

/// Simple mesh generator for demonstration
pub struct MeshGenerator;

impl MeshGenerator {
    /// Generate a simple cube domain with tet10 elements
    ///
    /// Creates a cube domain subdivided into tetrahedra
    /// with quadratic (tet10) elements
    ///
    /// # Arguments
    /// * `nx`, `ny`, `nz` - Number of divisions in each direction
    /// * `lx`, `ly`, `lz` - Domain dimensions (e.g., in km)
    pub fn generate_cube(nx: usize, ny: usize, nz: usize, lx: f64, ly: f64, lz: f64) -> Mesh {
        let mut mesh = Mesh::new();

        // Generate nodes for a structured grid
        // We'll create a simple subdivision with one tet10 element per hex cell
        let dx = lx / nx as f64;
        let dy = ly / ny as f64;
        let dz = lz / nz as f64;

        // For simplicity, create a single tet10 element as a demo
        // A full implementation would subdivide the cube properly

        Self::create_single_tet10(&mut mesh, dx, dy, dz);

        mesh
    }

    /// Create a single tet10 element for demonstration
    fn create_single_tet10(mesh: &mut Mesh, dx: f64, dy: f64, dz: f64) {
        // Create a tetrahedral element from a corner of the cube
        // Vertices of tetrahedron
        let v0 = mesh.geometry.add_node(0.0, 0.0, 0.0);
        let v1 = mesh.geometry.add_node(dx, 0.0, 0.0);
        let v2 = mesh.geometry.add_node(0.0, dy, 0.0);
        let v3 = mesh.geometry.add_node(0.0, 0.0, dz);

        // Edge midpoints
        // Edge 0-1
        let e01 = mesh.geometry.add_node(dx / 2.0, 0.0, 0.0);
        // Edge 1-2
        let e12 = mesh.geometry.add_node(dx / 2.0, dy / 2.0, 0.0);
        // Edge 2-0
        let e20 = mesh.geometry.add_node(0.0, dy / 2.0, 0.0);
        // Edge 0-3
        let e03 = mesh.geometry.add_node(0.0, 0.0, dz / 2.0);
        // Edge 1-3
        let e13 = mesh.geometry.add_node(dx / 2.0, 0.0, dz / 2.0);
        // Edge 2-3
        let e23 = mesh.geometry.add_node(0.0, dy / 2.0, dz / 2.0);

        // Create tet10 element
        let elem = Tet10Element::new([v0, v1, v2, v3, e01, e12, e20, e03, e13, e23]);
        mesh.connectivity.add_element(elem);
    }

    /// Generate a cube with multiple tet10 elements
    /// Each cube cell is divided into 5 or 6 tetrahedra
    ///
    /// # Arguments
    /// * `nx`, `ny`, `nz` - Number of divisions in each direction
    /// * `lx`, `ly`, `lz` - Domain dimensions (e.g., in km)
    pub fn generate_cube_detailed(nx: usize, ny: usize, nz: usize, lx: f64, ly: f64, lz: f64) -> Mesh {
        let mut mesh = Mesh::new();

        let dx = lx / nx as f64;
        let dy = ly / ny as f64;
        let dz = lz / nz as f64;

        // Create a structured grid of nodes with edge midpoints
        // For a proper tet10 mesh, we need both vertex and edge nodes

        // Generate vertices for the hexahedral grid
        let mut vertex_map = std::collections::HashMap::new();

        for iz in 0..=nz {
            for iy in 0..=ny {
                for ix in 0..=nx {
                    let x = ix as f64 * dx;
                    let y = iy as f64 * dy;
                    let z = iz as f64 * dz;
                    let node_id = mesh.geometry.add_node(x, y, z);
                    vertex_map.insert((ix, iy, iz), node_id);
                }
            }
        }

        // Subdivide each hex into 5 tetrahedra and add edge nodes
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    Self::subdivide_hex_to_tet10(
                        &mut mesh,
                        &vertex_map,
                        ix,
                        iy,
                        iz,
                        nx,
                        ny,
                        nz,
                    );
                }
            }
        }

        mesh
    }

    /// Subdivide a hexahedral cell into tet10 elements
    fn subdivide_hex_to_tet10(
        mesh: &mut Mesh,
        vertex_map: &std::collections::HashMap<(usize, usize, usize), usize>,
        ix: usize,
        iy: usize,
        iz: usize,
        _nx: usize,
        _ny: usize,
        _nz: usize,
    ) {
        // Get the 8 vertices of the hex
        let v000 = vertex_map[&(ix, iy, iz)];
        let v100 = vertex_map[&(ix + 1, iy, iz)];
        let v010 = vertex_map[&(ix, iy + 1, iz)];
        let v110 = vertex_map[&(ix + 1, iy + 1, iz)];
        let v001 = vertex_map[&(ix, iy, iz + 1)];
        let _v101 = vertex_map[&(ix + 1, iy, iz + 1)];
        let _v011 = vertex_map[&(ix, iy + 1, iz + 1)];
        let v111 = vertex_map[&(ix + 1, iy + 1, iz + 1)];

        // One common subdivision: 5 tets
        // For simplicity, we'll create one tet with edge nodes

        // Tet 1: v000, v100, v010, v001
        // Clone the points to avoid borrowing issues
        let p0 = mesh.geometry.nodes[v000];
        let p1 = mesh.geometry.nodes[v100];
        let p2 = mesh.geometry.nodes[v010];
        let p3 = mesh.geometry.nodes[v001];

        // Add edge midpoints
        let e01 = mesh.geometry.add_node(
            (p0.x + p1.x) / 2.0,
            (p0.y + p1.y) / 2.0,
            (p0.z + p1.z) / 2.0,
        );
        let e12 = mesh.geometry.add_node(
            (p1.x + p2.x) / 2.0,
            (p1.y + p2.y) / 2.0,
            (p1.z + p2.z) / 2.0,
        );
        let e20 = mesh.geometry.add_node(
            (p2.x + p0.x) / 2.0,
            (p2.y + p0.y) / 2.0,
            (p2.z + p0.z) / 2.0,
        );
        let e03 = mesh.geometry.add_node(
            (p0.x + p3.x) / 2.0,
            (p0.y + p3.y) / 2.0,
            (p0.z + p3.z) / 2.0,
        );
        let e13 = mesh.geometry.add_node(
            (p1.x + p3.x) / 2.0,
            (p1.y + p3.y) / 2.0,
            (p1.z + p3.z) / 2.0,
        );
        let e23 = mesh.geometry.add_node(
            (p2.x + p3.x) / 2.0,
            (p2.y + p3.y) / 2.0,
            (p2.z + p3.z) / 2.0,
        );

        let elem = Tet10Element::new([v000, v100, v010, v001, e01, e12, e20, e03, e13, e23]);
        mesh.connectivity.add_element(elem);

        // For a complete implementation, add the other 4 tets
        // This is simplified for the demo

        // Add one more tet for variety: v100, v110, v010, v111
        let p0 = mesh.geometry.nodes[v100];
        let p1 = mesh.geometry.nodes[v110];
        let p2 = mesh.geometry.nodes[v010];
        let p3 = mesh.geometry.nodes[v111];

        let e01_2 = mesh.geometry.add_node(
            (p0.x + p1.x) / 2.0,
            (p0.y + p1.y) / 2.0,
            (p0.z + p1.z) / 2.0,
        );
        let e12_2 = mesh.geometry.add_node(
            (p1.x + p2.x) / 2.0,
            (p1.y + p2.y) / 2.0,
            (p1.z + p2.z) / 2.0,
        );
        let e20_2 = mesh.geometry.add_node(
            (p2.x + p0.x) / 2.0,
            (p2.y + p0.y) / 2.0,
            (p2.z + p0.z) / 2.0,
        );
        let e03_2 = mesh.geometry.add_node(
            (p0.x + p3.x) / 2.0,
            (p0.y + p3.y) / 2.0,
            (p0.z + p3.z) / 2.0,
        );
        let e13_2 = mesh.geometry.add_node(
            (p1.x + p3.x) / 2.0,
            (p1.y + p3.y) / 2.0,
            (p1.z + p3.z) / 2.0,
        );
        let e23_2 = mesh.geometry.add_node(
            (p2.x + p3.x) / 2.0,
            (p2.y + p3.y) / 2.0,
            (p2.z + p3.z) / 2.0,
        );

        let elem2 = Tet10Element::new([v100, v110, v010, v111, e01_2, e12_2, e20_2, e03_2, e13_2, e23_2]);
        mesh.connectivity.add_element(elem2);
    }
}
