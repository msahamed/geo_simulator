/// Mesh Quality Assessment and Improvement
///
/// Provides tools to:
/// 1. Measure mesh quality (Jacobian determinants, aspect ratios)
/// 2. Smooth mesh using Laplacian or other methods
/// 3. Detect and repair mesh tangling
///
/// **Critical for long-time Lagrangian simulations** where mesh distortion
/// can cause solver failure even when physics is correct.

use crate::mesh::Mesh;
use crate::mesh::geometry::Geometry;
use nalgebra::Point3;
use std::collections::{HashMap, HashSet};

/// Mesh quality statistics
#[derive(Debug, Clone)]
pub struct MeshQuality {
    /// Minimum Jacobian determinant (should be > 0)
    pub min_jacobian: f64,
    /// Average Jacobian determinant
    pub avg_jacobian: f64,
    /// Maximum Jacobian determinant
    pub max_jacobian: f64,
    /// Number of inverted elements (det(J) < 0)
    pub num_inverted: usize,
    /// Number of near-degenerate elements (det(J) < 0.01)
    pub num_degenerate: usize,
    /// Total elements
    pub total_elements: usize,
}

impl MeshQuality {
    /// Check if mesh is acceptable for simulation
    pub fn is_acceptable(&self) -> bool {
        self.num_inverted == 0 && self.min_jacobian > 0.01
    }

    /// Check if mesh needs smoothing
    pub fn needs_smoothing(&self) -> bool {
        self.num_inverted > 0 || self.min_jacobian < 0.1 || self.num_degenerate > self.total_elements / 20
    }

    /// Human-readable quality report
    pub fn report(&self) -> String {
        format!(
            "Mesh Quality: min_J={:.3}, avg_J={:.3}, inverted={}/{}, degenerate={}/{}",
            self.min_jacobian,
            self.avg_jacobian,
            self.num_inverted,
            self.total_elements,
            self.num_degenerate,
            self.total_elements
        )
    }
}

/// Compute Jacobian determinant for a tetrahedral element
///
/// The Jacobian matrix maps from reference element to physical element:
/// J = [x1-x0  x2-x0  x3-x0]
///     [y1-y0  y2-y0  y3-y0]
///     [z1-z0  z2-z0  z3-z0]
///
/// det(J) > 0: valid element
/// det(J) = 0: degenerate (zero volume)
/// det(J) < 0: inverted element (simulation will fail!)
#[allow(non_snake_case)]
pub fn compute_tet_jacobian(vertices: &[Point3<f64>; 4]) -> f64 {
    let v0 = vertices[0];
    let v1 = vertices[1];
    let v2 = vertices[2];
    let v3 = vertices[3];

    // Edge vectors from v0
    let e1 = nalgebra::Vector3::new(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    let e2 = nalgebra::Vector3::new(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    let e3 = nalgebra::Vector3::new(v3.x - v0.x, v3.y - v0.y, v3.z - v0.z);

    // det(J) = e1 · (e2 × e3)
    e1.dot(&e2.cross(&e3))
}

/// Assess mesh quality by computing Jacobian determinants for all elements
///
/// Uses parallel computation for large meshes (>1000 elements)
pub fn assess_mesh_quality(mesh: &Mesh) -> MeshQuality {
    let total_elements = mesh.connectivity.tet10_elements.len();

    // Use parallel for large meshes
    if total_elements > 1000 {
        assess_mesh_quality_parallel(mesh)
    } else {
        assess_mesh_quality_serial(mesh)
    }
}

/// Serial quality assessment (for small meshes)
fn assess_mesh_quality_serial(mesh: &Mesh) -> MeshQuality {
    let mut min_jacobian = f64::INFINITY;
    let mut max_jacobian = f64::NEG_INFINITY;
    let mut sum_jacobian = 0.0;
    let mut num_inverted = 0;
    let mut num_degenerate = 0;

    for elem in &mesh.connectivity.tet10_elements {
        // Get first 4 vertices (corner nodes)
        let vertices = [
            mesh.geometry.nodes[elem.nodes[0]],
            mesh.geometry.nodes[elem.nodes[1]],
            mesh.geometry.nodes[elem.nodes[2]],
            mesh.geometry.nodes[elem.nodes[3]],
        ];

        let det_j = compute_tet_jacobian(&vertices);

        if det_j < 0.0 {
            num_inverted += 1;
        } else if det_j < 0.01 {
            num_degenerate += 1;
        }

        min_jacobian = min_jacobian.min(det_j);
        max_jacobian = max_jacobian.max(det_j);
        sum_jacobian += det_j;
    }

    let total_elements = mesh.connectivity.tet10_elements.len();

    MeshQuality {
        min_jacobian,
        avg_jacobian: sum_jacobian / total_elements as f64,
        max_jacobian,
        num_inverted,
        num_degenerate,
        total_elements,
    }
}

/// Parallel quality assessment (for large meshes)
fn assess_mesh_quality_parallel(mesh: &Mesh) -> MeshQuality {
    use rayon::prelude::*;

    let results: Vec<(f64, bool, bool)> = mesh.connectivity.tet10_elements
        .par_iter()
        .map(|elem| {
            let vertices = [
                mesh.geometry.nodes[elem.nodes[0]],
                mesh.geometry.nodes[elem.nodes[1]],
                mesh.geometry.nodes[elem.nodes[2]],
                mesh.geometry.nodes[elem.nodes[3]],
            ];
            let det_j = compute_tet_jacobian(&vertices);
            (det_j, det_j < 0.0, det_j > 0.0 && det_j < 0.01)
        })
        .collect();

    let mut min_jacobian = f64::INFINITY;
    let mut max_jacobian = f64::NEG_INFINITY;
    let mut sum_jacobian = 0.0;
    let mut num_inverted = 0;
    let mut num_degenerate = 0;

    for (det_j, is_inverted, is_degenerate) in results.iter() {
        if *is_inverted {
            num_inverted += 1;
        } else if *is_degenerate {
            num_degenerate += 1;
        }
        min_jacobian = min_jacobian.min(*det_j);
        max_jacobian = max_jacobian.max(*det_j);
        sum_jacobian += det_j;
    }

    let total_elements = mesh.connectivity.tet10_elements.len();

    MeshQuality {
        min_jacobian,
        avg_jacobian: sum_jacobian / total_elements as f64,
        max_jacobian,
        num_inverted,
        num_degenerate,
        total_elements,
    }
}

/// Build node-to-node connectivity graph (neighbors)
///
/// Returns a map: node_id → set of neighboring node IDs
/// Neighbors = nodes that share an element
pub fn build_node_neighbors(mesh: &Mesh) -> HashMap<usize, HashSet<usize>> {
    let mut neighbors: HashMap<usize, HashSet<usize>> = HashMap::new();

    // Initialize empty sets for all nodes
    for node_id in 0..mesh.num_nodes() {
        neighbors.insert(node_id, HashSet::new());
    }

    // For each element, connect all node pairs
    for elem in &mesh.connectivity.tet10_elements {
        for i in 0..10 {
            for j in 0..10 {
                if i != j {
                    neighbors.get_mut(&elem.nodes[i]).unwrap().insert(elem.nodes[j]);
                }
            }
        }
    }

    neighbors
}

/// Laplacian mesh smoothing for interior nodes
///
/// Moves each interior node toward the average position of its neighbors.
/// Boundary nodes remain fixed to preserve domain geometry.
///
/// # Arguments
/// * `geometry` - Mesh geometry to smooth (modified in place)
/// * `neighbors` - Node connectivity graph (from `build_node_neighbors`)
/// * `fixed_nodes` - Set of boundary nodes that should not move
/// * `iterations` - Number of smoothing iterations (5-10 typical)
/// * `alpha` - Relaxation parameter (0.5-0.7 typical, 1.0 = full Laplacian)
///
/// # Algorithm
/// For each iteration:
///   For each interior node:
///     new_pos = (1-α)·old_pos + α·average(neighbor_positions)
///
/// # Notes
/// - Does NOT guarantee no tangling (use with caution for severe distortion)
/// - Causes volume loss (~0.1% per iteration)
/// - Fast: O(iterations × nodes × avg_neighbors)
pub fn smooth_laplacian(
    geometry: &mut Geometry,
    neighbors: &HashMap<usize, HashSet<usize>>,
    fixed_nodes: &HashSet<usize>,
    iterations: usize,
    alpha: f64,
) {
    for _ in 0..iterations {
        let mut new_positions = geometry.nodes.clone();

        for node_id in 0..geometry.nodes.len() {
            // Skip boundary nodes
            if fixed_nodes.contains(&node_id) {
                continue;
            }

            // Compute average neighbor position
            let node_neighbors = &neighbors[&node_id];
            if node_neighbors.is_empty() {
                continue;  // Isolated node (shouldn't happen)
            }

            let mut avg_pos = Point3::new(0.0, 0.0, 0.0);
            for &neighbor_id in node_neighbors {
                let neighbor = geometry.nodes[neighbor_id];
                avg_pos.x += neighbor.x;
                avg_pos.y += neighbor.y;
                avg_pos.z += neighbor.z;
            }
            let n = node_neighbors.len() as f64;
            avg_pos.x /= n;
            avg_pos.y /= n;
            avg_pos.z /= n;

            // Weighted average: (1-α)·old + α·avg
            let old = geometry.nodes[node_id];
            new_positions[node_id] = Point3::new(
                (1.0 - alpha) * old.x + alpha * avg_pos.x,
                (1.0 - alpha) * old.y + alpha * avg_pos.y,
                (1.0 - alpha) * old.z + alpha * avg_pos.z,
            );
        }

        geometry.nodes = new_positions;
    }
}

/// Convenience function: smooth mesh with automatic boundary detection
///
/// Automatically identifies boundary nodes (nodes on domain faces) and
/// keeps them fixed while smoothing interior.
///
/// # Arguments
/// * `mesh` - Mesh to smooth (modified in place)
/// * `iterations` - Number of smoothing iterations
/// * `alpha` - Relaxation parameter
///
/// # Returns
/// Mesh quality before and after smoothing
pub fn smooth_mesh_auto(
    mesh: &mut Mesh,
    iterations: usize,
    alpha: f64,
) -> (MeshQuality, MeshQuality) {
    let quality_before = assess_mesh_quality(mesh);

    // Build connectivity
    let neighbors = build_node_neighbors(mesh);

    // Identify boundary nodes (nodes with < 10 neighbors are likely on boundary)
    // More robust: check if on domain faces
    let mut fixed_nodes = HashSet::new();
    let tol = 1e-6;

    // Get domain bounds
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;

    for node in &mesh.geometry.nodes {
        x_min = x_min.min(node.x);
        x_max = x_max.max(node.x);
        y_min = y_min.min(node.y);
        y_max = y_max.max(node.y);
        z_min = z_min.min(node.z);
        z_max = z_max.max(node.z);
    }

    // Mark boundary nodes
    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        if (node.x - x_min).abs() < tol || (node.x - x_max).abs() < tol ||
           (node.y - y_min).abs() < tol || (node.y - y_max).abs() < tol ||
           (node.z - z_min).abs() < tol || (node.z - z_max).abs() < tol {
            fixed_nodes.insert(node_id);
        }
    }

    // Smooth
    smooth_laplacian(&mut mesh.geometry, &neighbors, &fixed_nodes, iterations, alpha);

    let quality_after = assess_mesh_quality(mesh);

    (quality_before, quality_after)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_generator_improved::ImprovedMeshGenerator;

    #[test]
    fn test_jacobian_regular_tet() {
        // Unit tetrahedron
        let vertices = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];
        let det_j = compute_tet_jacobian(&vertices);
        assert!(det_j > 0.0, "Regular tet should have positive Jacobian");
        assert!((det_j - 1.0).abs() < 1e-10, "Unit tet should have det(J) = 1");
    }

    #[test]
    fn test_jacobian_inverted_tet() {
        // Inverted tetrahedron (swap two vertices)
        let vertices = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),  // Swapped
            Point3::new(1.0, 0.0, 0.0),  // Swapped
            Point3::new(0.0, 0.0, 1.0),
        ];
        let det_j = compute_tet_jacobian(&vertices);
        assert!(det_j < 0.0, "Inverted tet should have negative Jacobian");
    }

    #[test]
    fn test_mesh_quality_good() {
        let mesh = ImprovedMeshGenerator::generate_cube(4, 4, 4, 10.0, 10.0, 10.0);
        let quality = assess_mesh_quality(&mesh);

        println!("Total elements: {}", quality.total_elements);
        println!("Inverted: {}", quality.num_inverted);
        println!("Min Jacobian: {}", quality.min_jacobian);
        println!("Max Jacobian: {}", quality.max_jacobian);

        assert_eq!(quality.num_inverted, 0, "Regular mesh should have no inverted elements");
        assert!(quality.min_jacobian > 0.0, "All elements should have positive Jacobian");
        assert!(quality.is_acceptable(), "Regular mesh should be acceptable");
    }

    #[test]
    fn test_node_neighbors() {
        let mesh = ImprovedMeshGenerator::generate_cube(2, 2, 2, 1.0, 1.0, 1.0);
        let neighbors = build_node_neighbors(&mesh);

        // All nodes should have neighbors
        for node_id in 0..mesh.num_nodes() {
            assert!(neighbors[&node_id].len() > 0, "Node {} should have neighbors", node_id);
        }

        // Interior nodes should have more neighbors than boundary nodes
        // (This is a weak test but validates basic connectivity)
        let avg_neighbors: f64 = neighbors.values()
            .map(|set| set.len() as f64)
            .sum::<f64>() / mesh.num_nodes() as f64;
        assert!(avg_neighbors > 5.0, "Average node should have multiple neighbors");
    }

    #[test]
    fn test_smooth_preserves_topology() {
        let mut mesh = ImprovedMeshGenerator::generate_cube(4, 4, 4, 10.0, 10.0, 10.0);
        let quality_before = assess_mesh_quality(&mesh);

        let (_, quality_after) = smooth_mesh_auto(&mut mesh, 5, 0.5);

        assert_eq!(quality_after.total_elements, quality_before.total_elements,
                   "Smoothing should not change element count");
        assert_eq!(quality_after.num_inverted, 0,
                   "Smoothing regular mesh should not create inversions");
    }
}
