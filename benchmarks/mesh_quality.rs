/// Mesh quality analysis tool

use geo_simulator::MeshGenerator;

fn main() {
    println!("=== Mesh Quality Analysis ===\n");

    let test_cases = vec![
        (1, 1, 1, "1x1x1"),
        (2, 2, 2, "2x2x2"),
        (3, 3, 3, "3x3x3"),
    ];

    for (nx, ny, nz, label) in test_cases {
        println!("--- {} Mesh ---", label);

        let mesh = MeshGenerator::generate_cube_detailed(nx, ny, nz, 1.0, 1.0, 1.0);

        println!("Nodes: {}, Elements: {}", mesh.num_nodes(), mesh.num_elements());

        // Analyze element quality
        let mut aspect_ratios = Vec::new();
        let mut min_edge_lengths = Vec::new();
        let mut max_edge_lengths = Vec::new();

        for elem in &mesh.connectivity.tet10_elements {
            let vertices = elem.vertices();

            // Get vertex coordinates
            let v0 = mesh.geometry.nodes[vertices[0]];
            let v1 = mesh.geometry.nodes[vertices[1]];
            let v2 = mesh.geometry.nodes[vertices[2]];
            let v3 = mesh.geometry.nodes[vertices[3]];

            // Compute all 6 edge lengths
            let edges = [
                (v0 - v1).norm(), // edge 01
                (v0 - v2).norm(), // edge 02
                (v0 - v3).norm(), // edge 03
                (v1 - v2).norm(), // edge 12
                (v1 - v3).norm(), // edge 13
                (v2 - v3).norm(), // edge 23
            ];

            let min_edge = edges.iter().copied().fold(f64::INFINITY, f64::min);
            let max_edge = edges.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            let aspect_ratio = max_edge / min_edge;

            aspect_ratios.push(aspect_ratio);
            min_edge_lengths.push(min_edge);
            max_edge_lengths.push(max_edge);
        }

        // Statistics
        let mean_ar: f64 = aspect_ratios.iter().sum::<f64>() / aspect_ratios.len() as f64;
        let max_ar = aspect_ratios.iter().copied().fold(0.0, f64::max);
        let min_ar = aspect_ratios.iter().copied().fold(f64::INFINITY, f64::min);

        println!("\nElement Quality Metrics:");
        println!("  Aspect Ratio:");
        println!("    Mean: {:.2}", mean_ar);
        println!("    Min:  {:.2}", min_ar);
        println!("    Max:  {:.2}", max_ar);

        let mean_min_edge: f64 = min_edge_lengths.iter().sum::<f64>() / min_edge_lengths.len() as f64;
        let mean_max_edge: f64 = max_edge_lengths.iter().sum::<f64>() / max_edge_lengths.len() as f64;

        println!("  Edge Lengths:");
        println!("    Mean min edge: {:.3}", mean_min_edge);
        println!("    Mean max edge: {:.3}", mean_max_edge);

        // Quality assessment
        if max_ar < 2.0 {
            println!("  ✓✓ EXCELLENT: All elements well-shaped (AR < 2)");
        } else if max_ar < 3.0 {
            println!("  ✓ GOOD: Reasonable quality (AR < 3)");
        } else if max_ar < 5.0 {
            println!("  ⚠ ACCEPTABLE: Some distortion (AR < 5)");
        } else {
            println!("  ✗ POOR: Highly distorted elements (AR >= 5)");
        }

        // Show first element in detail
        if mesh.connectivity.tet10_elements.len() > 0 {
            println!("\n  First element detail:");
            let elem = &mesh.connectivity.tet10_elements[0];
            let vertices = elem.vertices();

            println!("    Vertices:");
            for (i, &v_idx) in vertices.iter().enumerate() {
                let v = mesh.geometry.nodes[v_idx];
                println!("      v{}: ({:.3}, {:.3}, {:.3})", i, v.x, v.y, v.z);
            }

            // Show z-coordinates
            let z_coords: Vec<_> = vertices.iter()
                .map(|&v| mesh.geometry.nodes[v].z)
                .collect();
            println!("    Z-coords: {:?}", z_coords);

            let z_min = z_coords.iter().copied().fold(f64::INFINITY, f64::min);
            let z_max = z_coords.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let z_range = z_max - z_min;

            println!("    Z-range: {:.3} (min={:.3}, max={:.3})", z_range, z_min, z_max);
        }

        println!();
    }
}
