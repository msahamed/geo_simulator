use geo_simulator::{Tet10Basis, GaussQuadrature};
use nalgebra::Point3;
use approx::assert_relative_eq;

#[test]
fn test_curved_element_volume() {
    // Reference tetrahedron vertices
    let v0 = Point3::new(0.0, 0.0, 0.0);
    let v1 = Point3::new(1.0, 0.0, 0.0);
    let v2 = Point3::new(0.0, 1.0, 0.0);
    let v3 = Point3::new(0.0, 0.0, 1.0);

    // Midpoints (initially straight)
    let e01 = Point3::new(0.5, 0.0, 0.0);
    let e12 = Point3::new(0.5, 0.5, 0.0);
    let e20 = Point3::new(0.0, 0.5, 0.0);
    let e03 = Point3::new(0.0, 0.0, 0.5);
    let e13 = Point3::new(0.5, 0.0, 0.5);
    let e23 = Point3::new(0.0, 0.5, 0.5);

    let mut nodes = [v0, v1, v2, v3, e01, e12, e20, e03, e13, e23];

    // 1. Straight edge case: Volume should be exactly 1/6
    let quad = GaussQuadrature::tet_4point();
    let mut vol_straight = 0.0;
    for (qp, w) in quad.points.iter().zip(quad.weights.iter()) {
        let det_j = Tet10Basis::jacobian(qp, &nodes).determinant();
        vol_straight += w * det_j.abs();
    }
    assert_relative_eq!(vol_straight, 1.0/6.0, epsilon = 1e-10);

    // 2. Perturb a mid-side node outward (curve the edge)
    // Move e01 from (0.5, 0, 0) to (0.5, -0.1, 0)
    nodes[4].y -= 0.1;

    let mut vol_curved = 0.0;
    for (qp, w) in quad.points.iter().zip(quad.weights.iter()) {
        let det_j = Tet10Basis::jacobian(qp, &nodes).determinant();
        vol_curved += w * det_j.abs();
    }

    // The volume should have increased because of the outward bowing
    println!("Straight volume: {:.10}", vol_straight);
    println!("Curved volume:   {:.10}", vol_curved);
    assert!(vol_curved > vol_straight, "Bowing outward should increase volume");

    // 3. Partition of unity test on curved element
    // Evaluate shape functions at a random interior point
    let test_qp = [0.25, 0.25, 0.25, 0.25];
    let N = Tet10Basis::shape_functions(&test_qp);
    let sum_N: f64 = N.iter().sum();
    assert_relative_eq!(sum_N, 1.0, epsilon = 1e-10);
}
