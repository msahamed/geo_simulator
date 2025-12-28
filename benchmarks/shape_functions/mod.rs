/// Analytical benchmarks for Tet10 shape functions
///
/// These tests verify mathematical properties that the shape functions must satisfy

use geo_simulator::{Tet10Basis};
use nalgebra::Point3;
use approx::assert_relative_eq;

fn generate_nodes(vertices: &[Point3<f64>; 4]) -> [Point3<f64>; 10] {
    let v0 = vertices[0];
    let v1 = vertices[1];
    let v2 = vertices[2];
    let v3 = vertices[3];

    [
        v0, v1, v2, v3,
        Point3::new((v0.x + v1.x) * 0.5, (v0.y + v1.y) * 0.5, (v0.z + v1.z) * 0.5),
        Point3::new((v1.x + v2.x) * 0.5, (v1.y + v2.y) * 0.5, (v1.z + v2.z) * 0.5),
        Point3::new((v2.x + v0.x) * 0.5, (v2.y + v0.y) * 0.5, (v2.z + v0.z) * 0.5),
        Point3::new((v0.x + v3.x) * 0.5, (v0.y + v3.y) * 0.5, (v0.z + v3.z) * 0.5),
        Point3::new((v1.x + v3.x) * 0.5, (v1.y + v3.y) * 0.5, (v1.z + v3.z) * 0.5),
        Point3::new((v2.x + v3.x) * 0.5, (v2.y + v3.y) * 0.5, (v2.z + v3.z) * 0.5),
    ]
}

#[test]
fn benchmark_partition_of_unity() {
    println!("\n=== Benchmark: Partition of Unity ===");
    println!("Property: Σ N_i(ξ) = 1 for all points ξ");

    // Test at many random points
    let test_points = vec![
        [0.25, 0.25, 0.25, 0.25], // Center
        [1.0, 0.0, 0.0, 0.0],     // Vertex 0
        [0.0, 1.0, 0.0, 0.0],     // Vertex 1
        [0.0, 0.0, 1.0, 0.0],     // Vertex 2
        [0.0, 0.0, 0.0, 1.0],     // Vertex 3
        [0.5, 0.5, 0.0, 0.0],     // Edge midpoint
        [0.1, 0.2, 0.3, 0.4],     // Arbitrary interior
        [0.7, 0.1, 0.1, 0.1],     // Near vertex 0
        [0.0, 0.33, 0.33, 0.34],  // Face interior
    ];

    for (i, coords) in test_points.iter().enumerate() {
        let n = Tet10Basis::shape_functions(coords);
        let sum: f64 = n.iter().sum();

        println!("  Point {}: L = [{:.2}, {:.2}, {:.2}, {:.2}] → Σ N_i = {:.15}",
                 i, coords[0], coords[1], coords[2], coords[3], sum);

        assert_relative_eq!(sum, 1.0, epsilon = 1e-14);
    }

    println!("  ✓ All {} test points satisfy partition of unity\n", test_points.len());
}

#[test]
fn benchmark_kronecker_delta() {
    println!("\n=== Benchmark: Kronecker Delta Property ===");
    println!("Property: N_i(x_j) = δ_ij (shape function equals 1 at its node, 0 at others)");

    // Barycentric coordinates of the 10 nodes
    let node_coords = [
        // Vertices
        [1.0, 0.0, 0.0, 0.0],  // Node 0
        [0.0, 1.0, 0.0, 0.0],  // Node 1
        [0.0, 0.0, 1.0, 0.0],  // Node 2
        [0.0, 0.0, 0.0, 1.0],  // Node 3
        // Edge midpoints
        [0.5, 0.5, 0.0, 0.0],  // Node 4 (edge 0-1)
        [0.0, 0.5, 0.5, 0.0],  // Node 5 (edge 1-2)
        [0.5, 0.0, 0.5, 0.0],  // Node 6 (edge 2-0)
        [0.5, 0.0, 0.0, 0.5],  // Node 7 (edge 0-3)
        [0.0, 0.5, 0.0, 0.5],  // Node 8 (edge 1-3)
        [0.0, 0.0, 0.5, 0.5],  // Node 9 (edge 2-3)
    ];

    for (i, coords) in node_coords.iter().enumerate() {
        let n = Tet10Basis::shape_functions(coords);

        println!("  At node {} (L=[{:.1}, {:.1}, {:.1}, {:.1}]):",
                 i, coords[0], coords[1], coords[2], coords[3]);

        for (j, &val) in n.iter().enumerate() {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(val, expected, epsilon = 1e-14);

            if i == j {
                println!("    N_{} = {:.15} ✓", j, val);
            } else if val.abs() > 1e-12 {
                println!("    N_{} = {:.2e} (should be 0)", j, val);
            }
        }
    }

    println!("  ✓ All nodes satisfy Kronecker delta property\n");
}

#[test]
fn benchmark_linear_field_reproduction() {
    println!("\n=== Benchmark: Linear Field Reproduction ===");
    println!("Property: Shape functions exactly reproduce linear fields");

    // Reference tetrahedron vertices
    let vertices = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
    ];

    // Test linear field: f(x, y, z) = 2x + 3y + 4z + 5
    let linear_field = |p: &Point3<f64>| 2.0 * p.x + 3.0 * p.y + 4.0 * p.z + 5.0;

    // Nodal values (at vertices and edge midpoints)
    let node_positions = [
        vertices[0],  // Node 0
        vertices[1],  // Node 1
        vertices[2],  // Node 2
        vertices[3],  // Node 3
        Point3::new(0.5, 0.0, 0.0),  // Node 4 (edge 0-1)
        Point3::new(0.5, 0.5, 0.0),  // Node 5 (edge 1-2)
        Point3::new(0.0, 0.5, 0.0),  // Node 6 (edge 2-0)
        Point3::new(0.0, 0.0, 0.5),  // Node 7 (edge 0-3)
        Point3::new(0.5, 0.0, 0.5),  // Node 8 (edge 1-3)
        Point3::new(0.0, 0.5, 0.5),  // Node 9 (edge 2-3)
    ];

    let nodal_values: Vec<f64> = node_positions.iter().map(linear_field).collect();

    // Test at several interior points
    let test_points = vec![
        Point3::new(0.25, 0.25, 0.25),
        Point3::new(0.1, 0.2, 0.3),
        Point3::new(0.4, 0.3, 0.2),
        Point3::new(0.6, 0.2, 0.1),
    ];

    for point in &test_points {
        // Interpolate using shape functions
        let n = Tet10Basis::shape_functions_cartesian(point, &vertices);
        let interpolated: f64 = n.iter()
            .zip(nodal_values.iter())
            .map(|(ni, vi)| ni * vi)
            .sum();

        let exact = linear_field(point);
        let error = (interpolated - exact).abs();

        println!("  Point ({:.2}, {:.2}, {:.2}): interpolated = {:.15}, exact = {:.15}, error = {:.2e}",
                 point.x, point.y, point.z, interpolated, exact, error);

        assert_relative_eq!(interpolated, exact, epsilon = 1e-12);
    }

    println!("  ✓ Linear fields exactly reproduced (error < 1e-12)\n");
}

#[test]
fn benchmark_derivative_sum() {
    println!("\n=== Benchmark: Derivative Sum Property ===");
    println!("Property: Σ ∂N_i/∂x = 0 (and same for y, z)");

    let vertices = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
    ];

    let test_coords = [
        [0.25, 0.25, 0.25, 0.25],
        [0.1, 0.2, 0.3, 0.4],
    ];

    for coords in &test_coords {
        let nodes = generate_nodes(&vertices);
        let derivs = Tet10Basis::shape_derivatives_cartesian(coords, &nodes);

        let sum_dx: f64 = derivs.iter().map(|d| d[0]).sum();
        let sum_dy: f64 = derivs.iter().map(|d| d[1]).sum();
        let sum_dz: f64 = derivs.iter().map(|d| d[2]).sum();

        println!("  L = [{:.2}, {:.2}, {:.2}, {:.2}]:",
                 coords[0], coords[1], coords[2], coords[3]);
        println!("    Σ ∂N_i/∂x = {:.2e}", sum_dx);
        println!("    Σ ∂N_i/∂y = {:.2e}", sum_dy);
        println!("    Σ ∂N_i/∂z = {:.2e}", sum_dz);

        assert_relative_eq!(sum_dx, 0.0, epsilon = 1e-12);
        assert_relative_eq!(sum_dy, 0.0, epsilon = 1e-12);
        assert_relative_eq!(sum_dz, 0.0, epsilon = 1e-12);
    }

    println!("  ✓ Derivative sums equal zero\n");
}

#[test]
fn benchmark_jacobian_volume() {
    println!("\n=== Benchmark: Element Volume Computation ===");

    // Test various tetrahedra
    let test_cases = vec![
        (
            "Reference tetrahedron",
            [
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
            ],
            1.0 / 6.0,
        ),
        (
            "Unit cube divided",
            [
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(0.0, 0.0, 1.0),
            ],
            1.0 / 6.0,
        ),
        (
            "Scaled by 2",
            [
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(2.0, 0.0, 0.0),
                Point3::new(0.0, 2.0, 0.0),
                Point3::new(0.0, 0.0, 2.0),
            ],
            8.0 / 6.0,  // Volume scales as h^3
        ),
    ];

    for (name, vertices, expected_volume) in test_cases {
        let nodes = generate_nodes(&vertices);
        let computed_volume = Tet10Basis::element_volume(&nodes);
        let error = (computed_volume - expected_volume).abs();

        println!("  {}: V = {:.15} (expected {:.15}), error = {:.2e}",
                 name, computed_volume, expected_volume, error);

        assert_relative_eq!(computed_volume, expected_volume, epsilon = 1e-14);
    }

    println!("  ✓ All volumes computed correctly\n");
}
