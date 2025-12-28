/// Validation benchmark: Planar Couette Flow
///
/// **Problem Setup:**
/// - Two infinite parallel plates separated by height H
/// - Bottom plate fixed: v = 0
/// - Top plate moving: v_x = V, v_y = 0, v_z = 0
/// - Newtonian viscous fluid with viscosity μ
///
/// **Analytical Solution:**
/// ```
/// v_x(z) = V · (z/H)     (linear velocity profile)
/// v_y(z) = 0
/// v_z(z) = 0
/// τ_xz = μ · (V/H)       (constant shear stress)
/// ```
///
/// **Expected Results:**
/// - <1% error for medium mesh (4×4×5)
/// - <0.1% error for fine mesh (6×6×8)
/// - Linear velocity profile visible in ParaView
///
/// **Physics:**
/// Steady-state Stokes flow: ∇·τ = 0 where τ = 2μ ε̇ (deviatoric stress)

use geo_simulator::{
    ImprovedMeshGenerator, Mesh, DofManager, Assembler, ConjugateGradient, VectorField,
    VtkWriter, NewtonianViscosity, Solver,
};
use std::time::Instant;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Viscous Rheology Validation: Planar Couette Flow");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ========================================================================
    // Problem Parameters
    // ========================================================================

    let H = 10.0;       // Height between plates (m)
    let L = 100.0;      // Lateral extent (m) - large to approximate infinite plates
    let V = 1.0;        // Top plate velocity (m/s)
    let mu = 1000.0;    // Viscosity (Pa·s, like honey or silicate melt)

    println!("Physical Parameters:");
    println!("  Height (H):         {} m", H);
    println!("  Lateral extent (L): {} m", L);
    println!("  Top plate velocity: {} m/s", V);
    println!("  Viscosity (μ):      {} Pa·s", mu);
    println!();

    // Expected shear stress
    let tau_xz_exact = mu * (V / H);
    println!("  Expected shear stress τ_xz = μ·(V/H) = {:.2} Pa\n", tau_xz_exact);

    // ========================================================================
    // Run Multiple Mesh Refinements
    // ========================================================================

    let mesh_configs = vec![
        (2, 2, 3, "Coarse"),
        (4, 4, 5, "Medium"),
        (6, 6, 8, "Fine"),
    ];

    for (nx, ny, nz, label) in mesh_configs {
        println!("──────────────────────────────────────────────────────────────");
        println!("  {} Mesh: {}×{}×{}", label, nx, ny, nz);
        println!("──────────────────────────────────────────────────────────────\n");

        let (mesh, stats) = run_couette_flow(nx, ny, nz, L, L, H, V, mu);

        // Validate solution
        println!("\nValidation:");
        validate_couette_solution(&mesh, &stats.velocity, L, H, V, nx, ny);

        // Export VTK for finest mesh
        if label == "Fine" {
            println!("\nExporting VTK...");
            std::fs::create_dir_all("output/viscous_couette_flow")
                .expect("Failed to create output directory");

            VtkWriter::write_vtu(&mesh, "output/viscous_couette_flow/solution.vtu")
                .expect("Failed to write VTK file");

            println!("  Saved to: output/viscous_couette_flow/solution.vtu");
        }

        println!();
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Validation Complete!");
    println!("═══════════════════════════════════════════════════════════════");
}

struct SolverStats {
    velocity: Vec<f64>,
    n_dofs: usize,
    n_elements: usize,
    assembly_time_ms: f64,
    solve_time_ms: f64,
    iterations: usize,
}

fn run_couette_flow(
    nx: usize,
    ny: usize,
    nz: usize,
    lx: f64,
    ly: f64,
    lz: f64,
    v_top: f64,
    viscosity: f64,
) -> (Mesh, SolverStats) {
    // ========================================================================
    // 1. Generate Mesh
    // ========================================================================

    println!("Generating mesh...");
    let start = Instant::now();
    let mut mesh = ImprovedMeshGenerator::generate_cube(nx, ny, nz, lx, ly, lz);
    let n_nodes = mesh.num_nodes();
    let n_elems = mesh.connectivity.tet10_elements.len();
    println!("  Nodes: {}, Elements: {}", n_nodes, n_elems);
    println!("  Mesh generation: {:.2} ms", start.elapsed().as_secs_f64() * 1000.0);

    // ========================================================================
    // 2. Setup DOF Manager (3 DOF/node for velocity)
    // ========================================================================

    let mut dof_mgr = DofManager::new(n_nodes, 3);
    let n_dofs = dof_mgr.total_dofs();
    println!("  Total DOFs: {}", n_dofs);

    // ========================================================================
    // 3. Apply Boundary Conditions
    // ========================================================================

    println!("\nApplying boundary conditions...");
    let tol = 1e-6;
    let mut n_bottom = 0;
    let mut n_top = 0;

    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        // Bottom plate (z ≈ 0): v = 0 (no-slip)
        if node.z.abs() < tol {
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), 0.0); // v_x = 0
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0); // v_y = 0
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), 0.0); // v_z = 0
            n_bottom += 1;
        }

        // Top plate (z ≈ H): v_x = V, v_y = 0, v_z = 0
        if (node.z - lz).abs() < tol {
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), v_top);  // v_x = V
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0);    // v_y = 0
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), 0.0);    // v_z = 0
            n_top += 1;
        }
    }

    println!("  Bottom plate nodes: {}", n_bottom);
    println!("  Top plate nodes:    {}", n_top);

    // ========================================================================
    // 4. Assemble Global Viscosity Matrix
    // ========================================================================

    println!("\nAssembling global viscosity matrix...");
    let material = NewtonianViscosity::new(viscosity);

    let start = Instant::now();
    let K = Assembler::assemble_viscosity_stiffness_parallel(&mesh, &dof_mgr, &material);
    let assembly_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("  Matrix size: {}×{}", K.rows(), K.cols());
    println!("  Non-zeros: {}", K.nnz());
    println!("  Assembly time: {:.2} ms", assembly_time);

    // ========================================================================
    // 5. Setup Load Vector (no body forces for Couette flow)
    // ========================================================================

    let f = vec![0.0; n_dofs];
    println!("  Load vector: all zeros (no body forces)");

    // ========================================================================
    // 6. Apply Dirichlet Boundary Conditions
    // ========================================================================

    println!("\nApplying Dirichlet BCs...");
    let start = Instant::now();
    let (K_bc, f_bc) = Assembler::apply_dirichlet_bcs(&K, &f, &dof_mgr);
    println!("  BC application: {:.2} ms", start.elapsed().as_secs_f64() * 1000.0);

    // ========================================================================
    // 7. Solve Linear System: K v = f
    // ========================================================================

    println!("\nSolving linear system (CG)...");
    let mut solver = ConjugateGradient::new()
        .with_max_iterations(5000)
        .with_tolerance(1e-10);

    let start = Instant::now();
    let (velocity, stats) = solver.solve(&K_bc, &f_bc);
    let solve_time = start.elapsed().as_secs_f64() * 1000.0;

    if stats.converged {
        println!("  ✓ Converged in {} iterations", stats.iterations);
    } else {
        println!("  ✗ Did NOT converge (max iterations reached)");
    }
    println!("  Final residual: {:.3e}", stats.residual_norm);
    println!("  Solve time: {:.2} ms", solve_time);

    // ========================================================================
    // 8. Attach Solution to Mesh
    // ========================================================================

    let vel_field = VectorField::from_dof_vector("Velocity", &velocity);
    mesh.field_data.add_vector_field(vel_field);

    let solver_stats = SolverStats {
        velocity,
        n_dofs,
        n_elements: n_elems,
        assembly_time_ms: assembly_time,
        solve_time_ms: solve_time,
        iterations: stats.iterations,
    };

    (mesh, solver_stats)
}

fn validate_couette_solution(
    mesh: &Mesh,
    velocity: &[f64],
    lx: f64,
    lz: f64,
    v_top: f64,
    nx: usize,
    ny: usize,
) {
    // Sample velocity profile at center of domain (x, y) ≈ (L/2, L/2)
    let center_x = lx / 2.0;
    let center_y = lx / 2.0;  // Assuming square domain
    let tol_xy = lx / (nx as f64);  // Tolerance for "near center"

    let mut samples: Vec<(f64, f64, f64)> = Vec::new();  // (z, v_x_fem, v_x_exact)

    let dof_mgr = DofManager::new(mesh.num_nodes(), 3);

    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        // Check if near center
        if (node.x - center_x).abs() < tol_xy && (node.y - center_y).abs() < tol_xy {
            let v_x_fem = velocity[dof_mgr.global_dof(node_id, 0)];
            let v_x_exact = v_top * (node.z / lz);

            samples.push((node.z, v_x_fem, v_x_exact));
        }
    }

    // Sort by z coordinate
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Compute errors
    let mut max_abs_error: f64 = 0.0;
    let mut max_rel_error: f64 = 0.0;

    println!("  Velocity profile at center (x, y) ≈ ({:.1}, {:.1}):", center_x, center_y);
    println!("  ┌────────┬──────────┬──────────┬─────────┬──────────┐");
    println!("  │ z (m)  │  v_x FEM │ v_x Exact│ Abs Err │  Rel Err │");
    println!("  ├────────┼──────────┼──────────┼─────────┼──────────┤");

    for (z, v_fem, v_exact) in &samples {
        let abs_err = (v_fem - v_exact).abs();
        let rel_err = if v_exact.abs() > 1e-12 {
            (abs_err / v_exact.abs()) * 100.0
        } else {
            0.0
        };

        max_abs_error = max_abs_error.max(abs_err);
        max_rel_error = max_rel_error.max(rel_err);

        println!("  │ {:6.2} │ {:8.6} │ {:8.6} │ {:7.2e} │ {:7.3}% │",
                 z, v_fem, v_exact, abs_err, rel_err);
    }

    println!("  └────────┴──────────┴──────────┴─────────┴──────────┘");
    println!();
    println!("  Maximum absolute error: {:.3e} m/s", max_abs_error);
    println!("  Maximum relative error: {:.3}%", max_rel_error);

    // Success criteria
    if max_rel_error < 1.0 {
        println!("  ✓ EXCELLENT: Error < 1%");
    } else if max_rel_error < 5.0 {
        println!("  ✓ GOOD: Error < 5%");
    } else {
        println!("  ✗ FAILED: Error > 5%");
    }
}
