/// Benchmark: Laplacian Mesh Smoothing on Severely Distorted Mesh
///
/// **Goal:** Validate that Laplacian smoothing can recover highly distorted meshes
///
/// **Problem Setup:**
/// - Generate regular mesh
/// - Deliberately distort interior nodes to create inverted elements
/// - Apply Laplacian smoothing iteratively
/// - Verify mesh quality recovery
///
/// **Success Criteria:**
/// - All inverted elements recovered (0 inverted after smoothing)
/// - min_J > 0.01 (acceptable quality threshold)
/// - Convergence within reasonable iterations (<100)

use geo_simulator::{
    ImprovedMeshGenerator, assess_mesh_quality,
    build_node_neighbors, smooth_laplacian,
};
use std::collections::HashSet;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Mesh Smoothing: Distorted Mesh Recovery Test");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =====================================================================
    // Generate Initial Mesh
    // =====================================================================

    let nx = 8;
    let ny = 8;
    let nz = 8;
    let lx = 10.0;
    let ly = 10.0;
    let lz = 10.0;

    let mut mesh = ImprovedMeshGenerator::generate_cube(nx, ny, nz, lx, ly, lz);

    println!("Initial Mesh:");
    println!("  Domain: {:.0} × {:.0} × {:.0} m", lx, ly, lz);
    println!("  Resolution: {}×{}×{} cells", nx, ny, nz);
    println!("  Nodes: {}", mesh.num_nodes());
    println!("  Elements: {}\n", mesh.num_elements());

    // Check initial quality (should be perfect)
    let initial_quality = assess_mesh_quality(&mesh);
    println!("Initial Quality (before distortion):");
    println!("  {}\n", initial_quality.report());

    assert!(initial_quality.is_acceptable(), "Generated mesh should be valid!");

    // =====================================================================
    // Identify Boundary Nodes (must remain fixed during smoothing)
    // =====================================================================

    let tol = 1e-6;
    let mut boundary_nodes = HashSet::new();

    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        // Mark as boundary if on bottom or sides (but NOT top, so we can see deformation)
        if node.x.abs() < tol || (node.x - lx).abs() < tol
            || node.y.abs() < tol || (node.y - ly).abs() < tol
            || node.z.abs() < tol  // Only fix BOTTOM in z
        {
            boundary_nodes.insert(node_id);
        }
    }

    let num_interior = mesh.num_nodes() - boundary_nodes.len();
    println!("Boundary Constraints:");
    println!("  Boundary nodes: {}", boundary_nodes.len());
    println!("  Interior nodes: {} (will be distorted)\n", num_interior);

    // =====================================================================
    // Apply Severe Random Distortion to Interior Nodes
    // =====================================================================

    println!("Applying severe random distortion to interior nodes...");

    let distortion_factor = 1.5; // Move nodes up to 150% of element size (EXTREME!)
    let dx = lx / (nx as f64);
    let dy = ly / (ny as f64);
    let dz = lz / (nz as f64);
    let max_displacement = distortion_factor * dx.min(dy).min(dz);

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for node_id in 0..mesh.num_nodes() {
        if boundary_nodes.contains(&node_id) {
            continue; // Don't distort boundary
        }

        // Random displacement in all directions
        let rand_x = rng.gen_range(-max_displacement..max_displacement);
        let rand_y = rng.gen_range(-max_displacement..max_displacement);
        let rand_z = rng.gen_range(-max_displacement..max_displacement);

        mesh.geometry.nodes[node_id].x += rand_x;
        mesh.geometry.nodes[node_id].y += rand_y;
        mesh.geometry.nodes[node_id].z += rand_z;
    }

    // Check quality after distortion
    let distorted_quality = assess_mesh_quality(&mesh);
    println!("  {}\n", distorted_quality.report());

    if distorted_quality.num_inverted == 0 {
        println!("WARNING: Distortion did not create inverted elements!");
        println!("Try increasing distortion_factor.\n");
    } else {
        println!("✓ Successfully created {} inverted elements\n", distorted_quality.num_inverted);
    }

    // Export distorted mesh BEFORE smoothing for visualization
    use geo_simulator::{VtkWriter, ScalarField, compute_tet_jacobian};
    std::fs::create_dir_all("output/mesh_smoothing").ok();

    // Add Jacobian determinant as CELL (element) field for visualization
    let jacobians_before: Vec<f64> = mesh.connectivity.tet10_elements.iter()
        .map(|elem| {
            let vertices = [
                mesh.geometry.nodes[elem.nodes[0]],
                mesh.geometry.nodes[elem.nodes[1]],
                mesh.geometry.nodes[elem.nodes[2]],
                mesh.geometry.nodes[elem.nodes[3]],
            ];
            compute_tet_jacobian(&vertices)
        })
        .collect();
    mesh.cell_data.add_field(ScalarField::new("Jacobian", jacobians_before));

    VtkWriter::write_vtu(&mesh, "output/mesh_smoothing/distorted_before.vtu").unwrap();
    println!("  VTK output (distorted): output/mesh_smoothing/distorted_before.vtu");
    println!("    → In ParaView: Color by 'Jacobian' field to see inverted elements (negative values)\n");

    // =====================================================================
    // Apply Laplacian Smoothing Iteratively
    // =====================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("Applying Laplacian Smoothing...\n");

    let max_iterations = 100;
    let smoothing_rounds = 2;    // Iterations per smoothing call (very small for gradual progress)
    let alpha = 0.3;              // Smoothing factor (0.3 = gentle smoothing to show multiple rounds)

    let mut round = 0;
    let mut converged = false;

    while round < max_iterations / smoothing_rounds {
        let quality_before = assess_mesh_quality(&mesh);

        // Build neighbor connectivity
        let neighbors = build_node_neighbors(&mesh);

        // Apply smoothing
        smooth_laplacian(
            &mut mesh.geometry,
            &neighbors,
            &boundary_nodes,
            smoothing_rounds,
            alpha,
        );

        let quality_after = assess_mesh_quality(&mesh);

        round += 1;
        let total_iters = round * smoothing_rounds;

        // Compute improvement statistics
        let inverted_fixed = quality_before.num_inverted.saturating_sub(quality_after.num_inverted);
        let degenerate_fixed = quality_before.num_degenerate.saturating_sub(quality_after.num_degenerate);
        let min_j_improvement = quality_after.min_jacobian - quality_before.min_jacobian;
        let min_j_pct_improvement = if quality_before.min_jacobian.abs() > 1e-10 {
            (min_j_improvement / quality_before.min_jacobian.abs()) * 100.0
        } else {
            0.0
        };

        println!("Round {:2} ({:3} total iters):", round, total_iters);
        println!("  BEFORE: {}", quality_before.report());
        println!("  AFTER:  {}", quality_after.report());
        println!("  CHANGE: inverted fixed={}, degenerate fixed={}, Δmin_J={:+.3} ({:+.1}%)\n",
                 inverted_fixed, degenerate_fixed, min_j_improvement, min_j_pct_improvement);

        // Check convergence
        if quality_after.num_inverted == 0 && quality_after.min_jacobian > 0.01 {
            println!("✓✓✓ Converged! All elements recovered.");
            converged = true;
            break;
        }

        // Check if stuck
        if (quality_after.min_jacobian - quality_before.min_jacobian).abs() < 1e-8 {
            println!("⚠ Smoothing stalled (no improvement)");
            break;
        }
    }

    // =====================================================================
    // Final Validation
    // =====================================================================

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Validation Results");
    println!("═══════════════════════════════════════════════════════════════\n");

    let final_quality = assess_mesh_quality(&mesh);

    println!("Before distortion: {}", initial_quality.report());
    println!("After distortion:  {}", distorted_quality.report());
    println!("After smoothing:   {}\n", final_quality.report());

    // Success criteria
    let mut success = true;

    if final_quality.num_inverted == 0 {
        println!("✓✓✓ EXCELLENT: All inverted elements recovered (0 inverted)");
    } else {
        println!("✗ FAILED: {} inverted elements remain", final_quality.num_inverted);
        success = false;
    }

    if final_quality.min_jacobian > 0.01 {
        println!("✓✓ GOOD: Minimum Jacobian > 0.01 (acceptable quality)");
    } else {
        println!("✗ FAILED: Minimum Jacobian = {:.6} (too small)", final_quality.min_jacobian);
        success = false;
    }

    if converged {
        println!("✓ Converged within {} iterations", round * smoothing_rounds);
    } else {
        println!("⚠ Did not converge in {} iterations", max_iterations);
    }

    // Export final smoothed mesh for inspection
    // Add Jacobian field as CELL data to smoothed mesh too
    let jacobians_after: Vec<f64> = mesh.connectivity.tet10_elements.iter()
        .map(|elem| {
            let vertices = [
                mesh.geometry.nodes[elem.nodes[0]],
                mesh.geometry.nodes[elem.nodes[1]],
                mesh.geometry.nodes[elem.nodes[2]],
                mesh.geometry.nodes[elem.nodes[3]],
            ];
            compute_tet_jacobian(&vertices)
        })
        .collect();
    mesh.cell_data.add_field(ScalarField::new("Jacobian_After", jacobians_after));

    VtkWriter::write_vtu(&mesh, "output/mesh_smoothing/smoothed_after.vtu").unwrap();
    println!("\n  VTK Outputs:");
    println!("    BEFORE smoothing: output/mesh_smoothing/distorted_before.vtu (color by 'Jacobian')");
    println!("    AFTER smoothing:  output/mesh_smoothing/smoothed_after.vtu (color by 'Jacobian_After')");

    if success {
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("  ✓✓✓ Mesh Smoothing Validation PASSED");
        println!("═══════════════════════════════════════════════════════════════");
    } else {
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("  ✗ Mesh Smoothing Validation FAILED");
        println!("═══════════════════════════════════════════════════════════════");
    }
}
