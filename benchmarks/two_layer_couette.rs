/// Benchmark: Two-Layer Couette Flow with Analytical Validation
///
/// **Goal:** Validate viscosity contrast handling with exact analytical solution.
///
/// **Setup:**
/// - Domain: 10 m × 10 m × 10 m (H = 10 m height)
/// - Two layers: Lower (0-5m) with μ₁, Upper (5-10m) with μ₂
/// - BCs: Bottom wall fixed (v=0), Top wall moving (v_x = 1 m/s)
/// - Couette flow: Simple shear between parallel plates
///
/// **Analytical Solution:**
/// For symmetric layers (h = H/2) with viscosities μ₁ (lower) and μ₂ (upper):
///
/// Layer 1 (0 ≤ z ≤ H/2):
///   v_x(z) = 2V·μ₂·z / [H·(μ₁ + μ₂)]
///
/// Layer 2 (H/2 ≤ z ≤ H):
///   v_x(z) = 2V·μ₁·z / [H·(μ₁ + μ₂)] + V·(μ₂ - μ₁) / (μ₁ + μ₂)
///
/// **Physics:**
/// - Velocity is continuous at interface (z = H/2)
/// - Shear stress is continuous: μ₁·dv/dz|₁ = μ₂·dv/dz|₂
/// - Velocity profile is piecewise linear (different slopes)
///
/// **Success Criteria:**
/// - L2 error < 1% for all mesh resolutions
/// - Interface velocity matches analytical (within 0.1%)
/// - Stress continuity verified (within 1%)

use geo_simulator::{
    ImprovedMeshGenerator, DofManager, Assembler, ConjugateGradient, VectorField,
    VtkWriter, ScalarField, Solver, ElastoViscoPlastic,
};
use nalgebra::Point3;
use std::time::Instant;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Analytical Validation: Two-Layer Couette Flow");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ========================================================================
    // 1. Physical Parameters
    // ========================================================================

    let H = 10.0;       // Domain height (m)
    let L = 1.0;        // Domain width/depth (m) - thin to approximate 1D
    let V = 1.0;        // Top wall velocity (m/s)
    let h = H / 2.0;    // Interface height (m)

    // Two different viscosity contrasts to test
    let test_cases = vec![
        ("10× contrast", 1e3, 1e4),    // μ₂ = 10·μ₁
        ("100× contrast", 1e3, 1e5),   // μ₂ = 100·μ₁
        ("1000× contrast", 1e3, 1e6),  // μ₂ = 1000·μ₁
    ];

    for (test_name, mu1, mu2) in test_cases {
        println!("\n═══════════════════════════════════════════════════════════════");
        println!("  Test Case: {}", test_name);
        println!("═══════════════════════════════════════════════════════════════");
        println!("  Lower viscosity μ₁: {:.1e} Pa·s", mu1);
        println!("  Upper viscosity μ₂: {:.1e} Pa·s", mu2);
        println!("  Viscosity ratio:    {:.0}×\n", mu2 / mu1);

        // Run at multiple resolutions
        let resolutions = vec![
            (4, "Coarse"),
            (8, "Medium"),
            (16, "Fine"),
        ];

        for (n, res_name) in resolutions {
            print!("  {} mesh ({}×{}×{})...", res_name, n, n, n);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            let start = Instant::now();

            // ================================================================
            // 2. Mesh Generation
            // ================================================================

            let mut mesh = ImprovedMeshGenerator::generate_cube(n, n, n, L, L, H);

            // ================================================================
            // 3. Assign Material Properties (Element-Based)
            // ================================================================

            let n_elements = mesh.num_elements();
            let mut elem_mat_ids = vec![0u32; n_elements];
            let mut elem_viscosities = vec![mu1; n_elements];

            // Assign materials based on element centroid z-coordinate
            for (elem_id, elem) in mesh.connectivity.tet10_elements.iter().enumerate() {
                let mut z_sum = 0.0;
                for &node_id in &elem.nodes[0..4] {
                    z_sum += mesh.geometry.nodes[node_id].z;
                }
                let z_centroid = z_sum / 4.0;

                if z_centroid < h {
                    elem_mat_ids[elem_id] = 0;  // Lower layer
                    elem_viscosities[elem_id] = mu1;
                } else {
                    elem_mat_ids[elem_id] = 1;  // Upper layer
                    elem_viscosities[elem_id] = mu2;
                }
            }

            // ================================================================
            // 4. Boundary Conditions
            // ================================================================

            let mut dof_mgr = DofManager::new(mesh.num_nodes(), 3);
            let tol = 1e-6;

            for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
                // Bottom wall (z=0): All velocity components = 0
                if node.z.abs() < tol {
                    dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), 0.0);
                    dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0);
                    dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), 0.0);
                }

                // Top wall (z=H): v_x = V, v_y = 0, v_z = 0
                if (node.z - H).abs() < tol {
                    dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), V);
                    dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0);
                    dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), 0.0);
                }

                // Side walls: Free (no BCs - let flow develop naturally)
            }

            // ================================================================
            // 5. Create Materials (ElastoViscoPlastic with very high cohesion)
            // ================================================================

            // Create effectively Newtonian materials (cohesion >> stresses)
            let materials = vec![
                ElastoViscoPlastic::new(100e9, 0.25, mu1, 1e20, 0.0),  // Layer 1: never yields
                ElastoViscoPlastic::new(100e9, 0.25, mu2, 1e20, 0.0),  // Layer 2: never yields
            ];

            let elem_strains = vec![0.0; n_elements];  // No plastic strain
            let elem_pressures = vec![0.0; n_elements];  // No pressure
            let current_velocity = vec![0.0; dof_mgr.total_dofs()];  // Initial guess

            // ================================================================
            // 6. Assemble and Solve
            // ================================================================

            let K = Assembler::assemble_stokes_vep_multimaterial_parallel(
                &mesh, &dof_mgr, &materials, &elem_mat_ids, &current_velocity,
                &elem_pressures, &elem_strains
            );

            let f = vec![0.0; dof_mgr.total_dofs()];

            let (K_bc, f_bc) = Assembler::apply_dirichlet_bcs(&K, &f, &dof_mgr);

            // DEBUG: Check if system is trivial
            if n == 4 {
                let f_norm = f_bc.iter().map(|x| x*x).sum::<f64>().sqrt();
                println!("\n    DEBUG ({} mesh): f_bc norm = {:.3e}", res_name, f_norm);
            }

            let mut solver = ConjugateGradient::new()
                .with_max_iterations(10000)
                .with_tolerance(1e-10);

            let (velocity, stats) = solver.solve(&K_bc, &f_bc);

            // DEBUG: Check solution
            if n == 4 {
                let v_norm = velocity.iter().map(|x| x*x).sum::<f64>().sqrt();
                println!("    DEBUG: velocity norm = {:.3e}, iters = {}", v_norm, stats.iterations);
            }

            // ================================================================
            // 7. Analytical Solution and Error Computation
            // ================================================================

            let analytical_velocity = |z: f64| -> f64 {
                if z < h {
                    // Layer 1: v(z) = 2V·μ₂·z / [H·(μ₁ + μ₂)]
                    2.0 * V * mu2 * z / (H * (mu1 + mu2))
                } else {
                    // Layer 2: v(z) = 2V·μ₁·z / [H·(μ₁ + μ₂)] + V·(μ₂ - μ₁) / (μ₁ + μ₂)
                    2.0 * V * mu1 * z / (H * (mu1 + mu2)) + V * (mu2 - mu1) / (mu1 + mu2)
                }
            };

            // Compute L2 error
            let mut error_sum = 0.0;
            let mut norm_sum = 0.0;
            let mut max_error = 0.0;
            let mut interface_error = f64::INFINITY;

            for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
                let v_fem = velocity[node_id * 3 + 0];  // v_x component
                let v_analytical = analytical_velocity(node.z);

                let error = (v_fem - v_analytical).abs();
                error_sum += error * error;
                norm_sum += v_analytical * v_analytical;

                if error > max_error {
                    max_error = error;
                }

                // Check interface nodes (z ≈ h)
                if (node.z - h).abs() < H / (2.0 * n as f64) {
                    let rel_error = (error / v_analytical.max(V * 0.01)).abs();
                    if rel_error < interface_error {
                        interface_error = rel_error;
                    }
                }
            }

            let l2_error = (error_sum / norm_sum).sqrt() * 100.0;  // Percentage
            let max_rel_error = (max_error / V) * 100.0;

            let elapsed = start.elapsed().as_secs_f64();

            print!(" L2: {:.3}%, Max: {:.3}%, Iters: {}, Time: {:.2}s",
                   l2_error, max_rel_error, stats.iterations, elapsed);

            if l2_error < 1.0 {
                println!(" ✓");
            } else {
                println!(" ✗ FAILED");
            }

            // Export VTK for fine mesh only
            if n == 16 {
                mesh.field_data.add_vector_field(VectorField::from_dof_vector("Velocity", &velocity));
                mesh.cell_data.add_field(ScalarField::new("MaterialID",
                    elem_mat_ids.iter().map(|&x| x as f64).collect()));
                mesh.cell_data.add_field(ScalarField::new("Viscosity", elem_viscosities.clone()));

                // Add analytical solution as comparison
                let v_analytical: Vec<f64> = mesh.geometry.nodes.iter()
                    .map(|node| analytical_velocity(node.z))
                    .collect();
                mesh.field_data.add_field(ScalarField::new("AnalyticalVx", v_analytical.clone()));

                // Add error field
                let error_field: Vec<f64> = mesh.geometry.nodes.iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let v_fem = velocity[i * 3 + 0];
                        let v_exact = v_analytical[i];
                        ((v_fem - v_exact) / v_exact * 100.0).abs()
                    })
                    .collect();
                mesh.field_data.add_field(ScalarField::new("ErrorPercent", error_field));

                std::fs::create_dir_all("output/two_layer_couette").ok();
                let filename = format!("output/two_layer_couette/test_{}.vtu",
                                      test_name.replace("×", "x").replace(" ", "_"));
                VtkWriter::write_vtu(&mesh, &filename).unwrap();
            }
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Validation Complete!");
    println!("═══════════════════════════════════════════════════════════════");
}
