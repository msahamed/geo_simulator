/// Improved mesh generator with proper hex-to-tet subdivision
///
/// Uses 6-tet subdivision pattern for better element quality

use crate::mesh::{Mesh, Tet10Element};
use std::collections::HashMap;

pub struct ImprovedMeshGenerator;

impl ImprovedMeshGenerator {
    /// Generate a cube with 6-tet subdivision per hex
    ///
    /// Each hexahedral cell is subdivided into 6 tetrahedra using
    /// a diagonal pattern that ensures good element quality.
    ///
    /// # Arguments
    /// * `nx`, `ny`, `nz` - Number of divisions in each direction
    /// * `lx`, `ly`, `lz` - Domain dimensions
    pub fn generate_cube(nx: usize, ny: usize, nz: usize, lx: f64, ly: f64, lz: f64) -> Mesh {
        let mut mesh = Mesh::new();

        let dx = lx / nx as f64;
        let dy = ly / ny as f64;
        let dz = lz / nz as f64;

        // Create structured vertex grid
        let mut vertex_map = HashMap::new();

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

        // Edge midpoint cache to avoid duplicates
        let mut edge_cache: HashMap<(usize, usize), usize> = HashMap::new();

        // Helper to get or create edge midpoint
        let mut get_edge_midpoint = |mesh: &mut Mesh, v1: usize, v2: usize| -> usize {
            let key = if v1 < v2 { (v1, v2) } else { (v2, v1) };

            *edge_cache.entry(key).or_insert_with(|| {
                let p1 = mesh.geometry.nodes[v1];
                let p2 = mesh.geometry.nodes[v2];
                mesh.geometry.add_node(
                    (p1.x + p2.x) / 2.0,
                    (p1.y + p2.y) / 2.0,
                    (p1.z + p2.z) / 2.0,
                )
            })
        };

        // Subdivide each hex into 6 tets
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    Self::subdivide_hex_6tet(
                        &mut mesh,
                        &vertex_map,
                        &mut get_edge_midpoint,
                        ix,
                        iy,
                        iz,
                    );
                }
            }
        }

        mesh
    }

    /// Subdivide a hex into 6 tets using diagonal pattern
    ///
    /// Uses diagonal from v000 to v111
    fn subdivide_hex_6tet<F>(
        mesh: &mut Mesh,
        vertex_map: &HashMap<(usize, usize, usize), usize>,
        get_edge_midpoint: &mut F,
        ix: usize,
        iy: usize,
        iz: usize,
    )
    where
        F: FnMut(&mut Mesh, usize, usize) -> usize,
    {
        // Get 8 hex vertices
        let v000 = vertex_map[&(ix, iy, iz)];
        let v100 = vertex_map[&(ix + 1, iy, iz)];
        let v010 = vertex_map[&(ix, iy + 1, iz)];
        let v110 = vertex_map[&(ix + 1, iy + 1, iz)];
        let v001 = vertex_map[&(ix, iy, iz + 1)];
        let v101 = vertex_map[&(ix + 1, iy, iz + 1)];
        let v011 = vertex_map[&(ix, iy + 1, iz + 1)];
        let v111 = vertex_map[&(ix + 1, iy + 1, iz + 1)];

        // 6-tet subdivision around diagonal v000 → v111
        // This ensures each tet has vertices distributed across z-levels

        let tets = [
            // Tet 1: bottom-front-right
            (v000, v100, v110, v111),
            // Tet 2: bottom-front-left
            (v000, v100, v111, v101),
            // Tet 3: bottom-back-left
            (v000, v010, v110, v111),
            // Tet 4: top-back-left
            (v000, v010, v111, v011),
            // Tet 5: bottom-left
            (v000, v001, v111, v101),
            // Tet 6: top-left
            (v000, v001, v111, v011),
        ];

        for &(v0, v1, v2, v3) in &tets {
            Self::create_tet10(mesh, get_edge_midpoint, v0, v1, v2, v3);
        }
    }

    /// Create a single tet10 element with edge midpoints
    fn create_tet10<F>(
        mesh: &mut Mesh,
        get_edge_midpoint: &mut F,
        v0: usize,
        v1: usize,
        v2: usize,
        v3: usize,
    )
    where
        F: FnMut(&mut Mesh, usize, usize) -> usize,
    {
        // Check orientation and swap vertices if inverted
        // Compute Jacobian determinant: det(J) = (v1-v0) · ((v2-v0) × (v3-v0))
        let p0 = &mesh.geometry.nodes[v0];
        let p1 = &mesh.geometry.nodes[v1];
        let p2 = &mesh.geometry.nodes[v2];
        let p3 = &mesh.geometry.nodes[v3];

        let e1 = nalgebra::Vector3::new(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
        let e2 = nalgebra::Vector3::new(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
        let e3 = nalgebra::Vector3::new(p3.x - p0.x, p3.y - p0.y, p3.z - p0.z);

        let det_j = e1.dot(&e2.cross(&e3));

        // Swap v2 and v3 if inverted (det < 0)
        let (v2_corrected, v3_corrected) = if det_j < 0.0 {
            (v3, v2)  // Swap to fix orientation
        } else {
            (v2, v3)
        };

        // Get edge midpoints (will reuse if already created)
        let e01 = get_edge_midpoint(mesh, v0, v1);
        let e12 = get_edge_midpoint(mesh, v1, v2_corrected);
        let e20 = get_edge_midpoint(mesh, v2_corrected, v0);
        let e03 = get_edge_midpoint(mesh, v0, v3_corrected);
        let e13 = get_edge_midpoint(mesh, v1, v3_corrected);
        let e23 = get_edge_midpoint(mesh, v2_corrected, v3_corrected);

        // Create tet10 element
        // Node ordering: [v0, v1, v2, v3, e01, e12, e20, e03, e13, e23]
        let elem = Tet10Element::new([v0, v1, v2_corrected, v3_corrected, e01, e12, e20, e03, e13, e23]);
        mesh.connectivity.add_element(elem);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_6tet_subdivision_count() {
        let mesh = ImprovedMeshGenerator::generate_cube(1, 1, 1, 1.0, 1.0, 1.0);

        // 1x1x1 should give 6 elements
        assert_eq!(mesh.num_elements(), 6);

        // 2x2x2 should give 8 * 6 = 48 elements
        let mesh2 = ImprovedMeshGenerator::generate_cube(2, 2, 2, 1.0, 1.0, 1.0);
        assert_eq!(mesh2.num_elements(), 48);
    }

    #[test]
    fn test_no_duplicate_midpoints() {
        let mesh = ImprovedMeshGenerator::generate_cube(2, 2, 2, 1.0, 1.0, 1.0);

        // Count unique node positions
        let mut positions = std::collections::HashSet::new();
        for node in &mesh.geometry.nodes {
            let key = (
                (node.x * 1000.0).round() as i64,
                (node.y * 1000.0).round() as i64,
                (node.z * 1000.0).round() as i64,
            );
            assert!(positions.insert(key), "Duplicate node at {:?}", node);
        }
    }

    #[test]
    fn test_element_z_distribution() {
        let mesh = ImprovedMeshGenerator::generate_cube(1, 1, 1, 1.0, 1.0, 1.0);

        // Check that elements have vertices at different z-levels
        for elem in &mesh.connectivity.tet10_elements {
            let vertices = elem.vertices();
            let z_coords: Vec<_> = vertices.iter()
                .map(|&v| mesh.geometry.nodes[v].z)
                .collect();

            // Count unique z-coordinates
            let mut unique_z = z_coords.clone();
            unique_z.sort_by(|a, b| a.partial_cmp(b).unwrap());
            unique_z.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

            // Should have at least 2 different z-levels
            assert!(
                unique_z.len() >= 2,
                "Element should span multiple z-levels, got: {:?}",
                z_coords
            );
        }
    }
}
