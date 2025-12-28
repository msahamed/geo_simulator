/// 1D Thermal diffusion benchmark with IMPROVED MESH

use geo_simulator::{
    ImprovedMeshGenerator,
    DofManager, Assembler,
    Solver, ConjugateGradient,
};

fn analytical_solution(z: f64, lz: f64, t_bottom: f64, t_top: f64, k: f64, q: f64) -> f64 {
    let linear = t_bottom + (t_top - t_bottom) * (z / lz);
    let source_term = (q / (2.0 * k)) * z * (lz - z);
    linear + source_term
}

fn main() {
    println!("=== 1D Thermal with IMPROVED Mesh ===\n");

    let lx = 1000.0;
    let ly = 1000.0;
    let lz = 600.0;

    let t_bottom = 1280.0;
    let t_top = 0.0;
    let conductivity = 3.0;
    let heat_source = 1e-6;

    let test_cases = vec![
        (2, 2, 2, "Coarse"),
        (4, 4, 4, "Medium"),
        (6, 6, 6, "Fine"),
    ];

    println!("Problem: 1D steady-state thermal diffusion with heat source");
    println!("  k = {}, Q = {:.2e}", conductivity, heat_source);
    println!("  BC: T(z=0) = {} °C, T(z={}) = {} °C\n", t_bottom, lz, t_top);

    for (nx, ny, nz, label) in test_cases {
        println!("--- {} Mesh ({} x {} x {}) ---", label, nx, ny, nz);

        let mesh = ImprovedMeshGenerator::generate_cube(nx, ny, nz, lx, ly, lz);
        println!("  Nodes: {}, Elements: {}", mesh.num_nodes(), mesh.num_elements());

        let mut dof_mgr = DofManager::new(mesh.num_nodes(), 1);

        // Apply BCs
        for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
            if node.z.abs() < 1.0 {
                dof_mgr.set_dirichlet_node(node_id, t_bottom);
            } else if (node.z - lz).abs() < 1.0 {
                dof_mgr.set_dirichlet_node(node_id, t_top);
            }
        }

        println!("  Free DOFs: {}/{}", dof_mgr.num_free_dofs(), dof_mgr.total_dofs());

        // Assemble
        let k_orig = Assembler::assemble_thermal_stiffness_parallel(&mesh, &dof_mgr, conductivity);
        let f_orig = Assembler::assemble_thermal_load(&mesh, &dof_mgr, heat_source);
        let (k_mat, f_vec) = Assembler::apply_dirichlet_bcs(&k_orig, &f_orig, &dof_mgr);

        // Solve
        let mut solver = ConjugateGradient::new()
            .with_max_iterations(1000)
            .with_tolerance(1e-8);

        let (t_fem, stats) = solver.solve(&k_mat, &f_vec);

        println!("  Solver: iterations={}, converged={}", stats.iterations, stats.converged);

        // Validate
        let mut errors = Vec::new();

        for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
            if dof_mgr.is_dirichlet(node_id) {
                continue;
            }

            let t_analytical = analytical_solution(node.z, lz, t_bottom, t_top, conductivity, heat_source);
            let t_numerical = t_fem[node_id];
            let error = (t_numerical - t_analytical).abs();
            errors.push(error);
        }

        if errors.is_empty() {
            println!("  No free DOFs!");
            continue;
        }

        let mean_error: f64 = errors.iter().sum::<f64>() / errors.len() as f64;
        let rms_error = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
        let max_error = errors.iter().copied().fold(0.0, f64::max);

        let temp_range = t_bottom - t_top;
        let relative_error = (rms_error / temp_range) * 100.0;

        println!("\n  Error Analysis:");
        println!("    Mean: {:.4} °C", mean_error);
        println!("    RMS:  {:.4} °C", rms_error);
        println!("    Max:  {:.4} °C", max_error);
        println!("    Relative: {:.4}%", relative_error);

        if relative_error < 0.1 {
            println!("    ✓✓✓ PERFECT: <0.1% error!");
        } else if relative_error < 1.0 {
            println!("    ✓✓ EXCELLENT: <1% error!");
        } else if relative_error < 5.0 {
            println!("    ✓ PASS: <5% error");
        } else {
            println!("    ✗ FAIL: >5% error");
        }

        println!();
    }

    println!("=== Benchmark Complete ===");
}
