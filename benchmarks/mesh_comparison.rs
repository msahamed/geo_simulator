/// Compare old vs improved mesh generator with FEM

use geo_simulator::{
    MeshGenerator, ImprovedMeshGenerator,
    DofManager, Assembler,
    Solver, DirectSolver,
};

fn analytical_solution(z: f64, lz: f64, t_bottom: f64, t_top: f64) -> f64 {
    t_bottom + (t_top - t_bottom) * (z / lz)
}

fn test_mesh_fem(mesh: &geo_simulator::Mesh, label: &str) -> f64 {
    let t_bottom = 100.0;
    let t_top = 0.0;
    let lz = 1.0;
    let conductivity = 1.0;
    let heat_source = 0.0; // Linear case

    let mut dof_mgr = DofManager::new(mesh.num_nodes(), 1);

    // Apply BCs
    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        if node.z.abs() < 0.01 {
            dof_mgr.set_dirichlet_node(node_id, t_bottom);
        } else if (node.z - lz).abs() < 0.01 {
            dof_mgr.set_dirichlet_node(node_id, t_top);
        }
    }

    // Assemble
    let k_orig = Assembler::assemble_thermal_stiffness_parallel(&mesh, &dof_mgr, conductivity);
    let f_orig = Assembler::assemble_thermal_load(&mesh, &dof_mgr, heat_source);
    let (k_mat, f_vec) = Assembler::apply_dirichlet_bcs(&k_orig, &f_orig, &dof_mgr);

    // Solve
    let mut solver = DirectSolver::new();
    let (t_fem, _stats) = solver.solve(&k_mat, &f_vec);

    // Compute error
    let mut errors = Vec::new();

    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        if dof_mgr.is_dirichlet(node_id) {
            continue;
        }

        let t_exact = analytical_solution(node.z, lz, t_bottom, t_top);
        let t_computed = t_fem[node_id];
        let error = (t_computed - t_exact).abs();
        errors.push(error);
    }

    if errors.is_empty() {
        return 0.0;
    }

    let rms_error = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
    let max_error = errors.iter().copied().fold(0.0, f64::max);

    println!("{}:", label);
    println!("  Nodes: {}, Elements: {}", mesh.num_nodes(), mesh.num_elements());
    println!("  RMS Error: {:.4}", rms_error);
    println!("  Max Error: {:.4}", max_error);
    println!();

    rms_error
}

fn main() {
    println!("=== Mesh Generator Comparison ===\n");
    println!("Problem: Linear thermal diffusion (T = 100 - 100z)\n");

    let (nx, ny, nz) = (2, 2, 2);
    let (lx, ly, lz) = (1.0, 1.0, 1.0);

    println!("--- Old Mesh Generator (2 tets/hex) ---");
    let mesh_old = MeshGenerator::generate_cube_detailed(nx, ny, nz, lx, ly, lz);
    let error_old = test_mesh_fem(&mesh_old, "Old Mesh");

    println!("--- Improved Mesh Generator (6 tets/hex) ---");
    let mesh_new = ImprovedMeshGenerator::generate_cube(nx, ny, nz, lx, ly, lz);
    let error_new = test_mesh_fem(&mesh_new, "Improved Mesh");

    println!("--- Results ---");
    println!("Old mesh RMS error:      {:.4}", error_old);
    println!("Improved mesh RMS error: {:.4}", error_new);
    println!("Improvement factor:      {:.2}x", error_old / error_new);

    if error_new < error_old / 2.0 {
        println!("\n✓✓ EXCELLENT: Improved mesh has <50% error!");
    } else if error_new < error_old {
        println!("\n✓ GOOD: Improved mesh is better");
    } else {
        println!("\n✗ FAIL: Improved mesh is not better");
    }
}
