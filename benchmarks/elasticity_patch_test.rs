/// Patch test for linear elasticity implementation
///
/// The patch test verifies that the FEM implementation can exactly represent
/// constant strain states. This is a fundamental requirement for convergence.
///
/// Test: Uniaxial tension in z-direction
/// - Apply uniform stress σ_zz = σ₀ (tension)
/// - All other stress components = 0
/// - Exact solution: ε_zz = σ₀/E, ε_xx = ε_yy = -ν·σ₀/E
/// - Displacement: u_z = (σ₀/E)·z, u_x = -(ν·σ₀/E)·x, u_y = -(ν·σ₀/E)·y

use geo_simulator::*;

fn main() {
    println!("=== Elasticity Patch Test: Uniaxial Tension ===\n");

    // Problem parameters
    let lx = 10e3;  // 10 km
    let ly = 10e3;
    let lz = 10e3;

    // Material properties
    let E = 100e9;    // 100 GPa
    let nu = 0.25;    // Poisson's ratio

    // Applied stress (tension in z-direction)
    let sigma_0 = 10e6; // 10 MPa

    // Exact solution (constant strain everywhere)
    let epsilon_zz_exact = sigma_0 / E;
    let epsilon_xx_exact = -nu * sigma_0 / E;
    let epsilon_yy_exact = -nu * sigma_0 / E;

    println!("Problem Setup:");
    println!("  Domain: {:.0} × {:.0} × {:.0} km", lx/1e3, ly/1e3, lz/1e3);
    println!("  Material: E = {:.0} GPa, ν = {:.2}", E/1e9, nu);
    println!("  Applied stress: σ_zz = {:.1} MPa (tension)", sigma_0/1e6);
    println!("  Exact strains:");
    println!("    ε_zz = {:.6e}", epsilon_zz_exact);
    println!("    ε_xx = ε_yy = {:.6e}\n", epsilon_xx_exact);

    // Test different mesh resolutions
    let test_cases = vec![
        (2, 2, 2, "Coarse"),
        (3, 3, 3, "Medium"),
        (4, 4, 4, "Fine"),
    ];

    for (nx, ny, nz, label) in test_cases {
        println!("--- {} Mesh ({} × {} × {}) ---", label, nx, ny, nz);

        // Generate mesh
        let mesh = ImprovedMeshGenerator::generate_cube(nx, ny, nz, lx, ly, lz);
        println!("  Nodes: {}, Elements: {}", mesh.num_nodes(), mesh.num_elements());

        // Setup DOF manager
        let mut dof_mgr = DofManager::new(mesh.num_nodes(), 3);

        // Apply boundary conditions to enforce displacement field
        // u_x(x,y,z) = ε_xx · x = -(ν·σ₀/E)·x
        // u_y(x,y,z) = ε_yy · y = -(ν·σ₀/E)·y
        // u_z(x,y,z) = ε_zz · z = (σ₀/E)·z

        for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
            // Bottom face (z = 0): u_z = 0
            if node.z.abs() < 1.0 {
                let u_z = epsilon_zz_exact * node.z;
                dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), u_z);
            }

            // Top face (z = lz): u_z = ε_zz · lz
            if (node.z - lz).abs() < 1.0 {
                let u_z = epsilon_zz_exact * node.z;
                dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), u_z);
            }

            // x = 0 face: u_x = 0
            if node.x.abs() < 1.0 {
                let u_x = epsilon_xx_exact * node.x;
                dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), u_x);
            }

            // y = 0 face: u_y = 0
            if node.y.abs() < 1.0 {
                let u_y = epsilon_yy_exact * node.y;
                dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), u_y);
            }
        }

        println!("  Free DOFs: {} / {}", dof_mgr.num_free_dofs(), dof_mgr.total_dofs());

        // Assemble stiffness matrix (no body forces for patch test)
        let material = IsotropicElasticity::new(E, nu);
        let K = Assembler::assemble_elasticity_stiffness_parallel(&mesh, &dof_mgr, &material);

        // Zero force vector (displacement-driven test)
        let f = vec![0.0; dof_mgr.total_dofs()];

        // Apply boundary conditions
        let (K_bc, f_bc) = Assembler::apply_dirichlet_bcs(&K, &f, &dof_mgr);

        // Solve
        let mut solver = ConjugateGradient::new()
            .with_max_iterations(5000)
            .with_tolerance(1e-12);

        let (u, stats) = solver.solve(&K_bc, &f_bc);

        println!("  Solver: {} iterations, residual: {:.2e}", stats.iterations, stats.relative_residual);

        // Validation: Check displacements at a test point
        let test_node = mesh.geometry.nodes.len() / 2; // Approximate center
        let test_pt = &mesh.geometry.nodes[test_node];

        let u_x_fem = u[dof_mgr.global_dof(test_node, 0)];
        let u_y_fem = u[dof_mgr.global_dof(test_node, 1)];
        let u_z_fem = u[dof_mgr.global_dof(test_node, 2)];

        let u_x_exact = epsilon_xx_exact * test_pt.x;
        let u_y_exact = epsilon_yy_exact * test_pt.y;
        let u_z_exact = epsilon_zz_exact * test_pt.z;

        let error_x = (u_x_fem - u_x_exact).abs();
        let error_y = (u_y_fem - u_y_exact).abs();
        let error_z = (u_z_fem - u_z_exact).abs();

        let rel_error_x = error_x / u_x_exact.abs().max(1e-10) * 100.0;
        let rel_error_y = error_y / u_y_exact.abs().max(1e-10) * 100.0;
        let rel_error_z = error_z / u_z_exact.abs().max(1e-10) * 100.0;

        println!("\n  Test point: ({:.1}, {:.1}, {:.1}) km",
                 test_pt.x/1e3, test_pt.y/1e3, test_pt.z/1e3);
        println!("    u_x: FEM = {:.6e}, Exact = {:.6e}, Error = {:.2e} ({:.3}%)",
                 u_x_fem, u_x_exact, error_x, rel_error_x);
        println!("    u_y: FEM = {:.6e}, Exact = {:.6e}, Error = {:.2e} ({:.3}%)",
                 u_y_fem, u_y_exact, error_y, rel_error_y);
        println!("    u_z: FEM = {:.6e}, Exact = {:.6e}, Error = {:.2e} ({:.3}%)",
                 u_z_fem, u_z_exact, error_z, rel_error_z);

        let max_rel_error = rel_error_x.max(rel_error_y).max(rel_error_z);

        if max_rel_error < 0.01 {
            println!("    ✓✓✓ EXCELLENT: < 0.01% error (machine precision)");
        } else if max_rel_error < 0.1 {
            println!("    ✓✓ VERY GOOD: < 0.1% error");
        } else if max_rel_error < 1.0 {
            println!("    ✓ GOOD: < 1% error");
        } else {
            println!("    ✗ FAILED: Patch test should give near-exact results!");
        }

        println!();
    }

    println!("=== Patch Test Complete ===");
    println!("\nNote: Patch test verifies FEM can represent constant strain states.");
    println!("Near-exact agreement (<0.1% error) indicates correct implementation.");
}
