use geo_simulator::{
    Assembler, ConjugateGradient, DofManager, MeshGenerator, ScalarField, Solver, VtkWriter,
};

#[allow(non_snake_case)]
fn main() {
    println!("=== Thermal Diffusion Solver Demo ===\n");

    // Problem setup: Steady-state thermal diffusion in 3D
    // ∇·(k∇T) + Q = 0
    //
    // Domain: 1000 x 1000 x 600 km
    // BC: T = 0°C at surface (z = 600 km)
    //     T = 1280°C at bottom (z = 0 km)
    // Material: k = 3.0 W/m·K (typical for rock)
    // Source: Q = 1e-6 W/m³ (radiogenic heat)

    let lx = 1000.0; // km
    let ly = 1000.0;
    let lz = 600.0;

    // Generate mesh (medium size for demo)
    let (nx, ny, nz) = (5, 5, 3);

    println!("Generating mesh...");
    let mut mesh = MeshGenerator::generate_cube_detailed(nx, ny, nz, lx, ly, lz);
    println!("  Nodes: {}", mesh.num_nodes());
    println!("  Elements: {}", mesh.num_elements());

    // Create DOF manager
    let mut dof_mgr = DofManager::new(mesh.num_nodes(), 1);

    // Apply boundary conditions
    println!("\nApplying boundary conditions...");

    // Surface nodes (z ≈ lz)
    let mut surface_count = 0;
    let mut bottom_count = 0;

    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        // Surface: z = lz → T = 0°C
        if (node.z - lz).abs() < 1.0 {
            dof_mgr.set_dirichlet_node(node_id, 0.0);
            surface_count += 1;
        }
        // Bottom: z = 0 → T = 1280°C
        else if node.z.abs() < 1.0 {
            dof_mgr.set_dirichlet_node(node_id, 1280.0);
            bottom_count += 1;
        }
    }

    println!("  Surface BC (T=0°C): {} nodes", surface_count);
    println!("  Bottom BC (T=1280°C): {} nodes", bottom_count);
    println!(
        "  Free DOFs: {} / {}",
        dof_mgr.num_free_dofs(),
        dof_mgr.total_dofs()
    );

    // Material properties
    let conductivity = 3.0; // W/m·K
    let heat_source = 1e-6; // W/m³

    // Assemble system
    println!("\nAssembling system...");
    let K_orig = Assembler::assemble_thermal_stiffness_parallel(&mesh, &dof_mgr, conductivity);
    let f_orig = Assembler::assemble_thermal_load(&mesh, &dof_mgr, heat_source);

    println!("  Matrix: {} x {}", K_orig.rows(), K_orig.cols());
    println!("  Non-zeros: {}", K_orig.nnz());
    println!(
        "  Sparsity: {:.2}%",
        100.0 * K_orig.nnz() as f64 / (K_orig.rows() * K_orig.cols()) as f64
    );

    // Apply Dirichlet boundary conditions using penalty method
    println!("\nApplying boundary conditions to system...");
    let (K, f) = Assembler::apply_dirichlet_bcs(&K_orig, &f_orig, &dof_mgr);

    // Solve with Conjugate Gradient
    println!("\n--- Solving with Conjugate Gradient ---");
    let mut cg_solver = ConjugateGradient::new()
        .with_max_iterations(1000)
        .with_tolerance(1e-8)
        .with_preconditioner(true);

    let (T_cg, stats_cg) = cg_solver.solve(&K, &f);

    println!("  Iterations: {}", stats_cg.iterations);
    println!("  Relative residual: {:.2e}", stats_cg.relative_residual);
    println!("  Converged: {}", stats_cg.converged);
    println!("  Solve time: {:.3} s", stats_cg.solve_time);

    // Solution statistics
    let T_min = T_cg.iter().copied().fold(f64::INFINITY, f64::min);
    let T_max = T_cg.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let T_mean: f64 = T_cg.iter().sum::<f64>() / T_cg.len() as f64;

    println!("\n  Solution statistics:");
    println!("    Min temperature: {:.2} °C", T_min);
    println!("    Max temperature: {:.2} °C", T_max);
    println!("    Mean temperature: {:.2} °C", T_mean);

    // Export solution to VTK
    println!("\nExporting solution to VTK...");

    // Add temperature field
    mesh.field_data
        .add_field(ScalarField::new("Temperature", T_cg.clone()));

    // Add depth field for reference
    let depth: Vec<f64> = mesh.geometry.nodes.iter().map(|node| lz - node.z).collect();
    mesh.field_data
        .add_field(ScalarField::new("Depth_km", depth));

    std::fs::create_dir_all("output/thermal_solver_demo").expect("Failed to create output directory");
    let vtk_file = "output/thermal_solver_demo/solution.vtu";
    VtkWriter::write_vtu(&mesh, vtk_file).expect("Failed to write VTK");
    println!("  ✓ Wrote {}", vtk_file);

    println!("\n=== Solve Complete ===");
    println!("\nVisualize with:");
    println!("  paraview {}", vtk_file);
    println!("\nExpected: Linear temperature gradient from bottom (1280°C) to surface (0°C)");
}
