/// Simple Newton-Krylov (JFNK) + GMRES + Block-Diagonal benchmark for core complex
///
/// Purpose: Compare Newton-Krylov with Picard iteration
/// Strategy: Same problem setup as picard_simple.rs but using Newton's method
///
/// Solver Stack:
/// - Nonlinear: Newton-Krylov (quadratic convergence)
/// - Linear: GMRES with block-diagonal preconditioner (ILU(0) for velocity, Identity for pressure)
/// - Scaling: Non-dimensionalization (critical!)

use geo_simulator::*;
use geo_simulator::ImprovedMeshGenerator;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SIMPLE TEST: Newton-Krylov + GMRES + Block-Diagonal");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Load config
    let config_path = "inputs/core_complex_2d/config_debug_single_material.toml";
    let config = config::SimulationConfig::from_file(config_path)
        .expect("Failed to load config");

    println!("Loaded config: {}", config_path);
    println!("Domain: {}×{}×{} km",
        config.domain.lx/1e3, config.domain.ly/1e3, config.domain.lz/1e3);
    println!("Resolution: {}×{}×{} km\n",
        config.domain.dx/1e3, config.domain.dy/1e3, config.domain.dz/1e3);

    // Generate mesh
    println!("Generating mesh...");
    let (lx, ly, lz) = (config.domain.lx, config.domain.ly, config.domain.lz);
    let (dx, dy, dz) = (config.domain.dx, config.domain.dy, config.domain.dz);
    let (nx, ny, nz) = ((lx/dx) as usize, (ly/dy) as usize, (lz/dz) as usize);
    let mesh = ImprovedMeshGenerator::generate_cube(nx, ny, nz, lx, ly, lz);

    let n_elements = mesh.connectivity.tet10_elements.len();
    let n_nodes = mesh.num_nodes();
    println!("  Mesh: {} elements, {} nodes", n_elements, n_nodes);

    // DOF manager (mixed formulation: velocity on all nodes, pressure on corners only)
    let mut dof_mgr = fem::DofManager::new_mixed(n_nodes, &mesh.connectivity.corner_nodes());
    let n_dofs = dof_mgr.total_dofs();
    println!("  DOFs: {} velocity, {} pressure\n",
        dof_mgr.total_vel_dofs(), n_dofs - dof_mgr.total_vel_dofs());

    // Boundary conditions
    println!("Setting up boundary conditions...");
    let v_extension = config.boundary_conditions.extension_rate_m_per_s;

    // Find boundary nodes manually
    let mut left_nodes = Vec::new();
    let mut right_nodes = Vec::new();
    let mut bottom_nodes = Vec::new();
    let mut back_nodes = Vec::new();
    let mut front_nodes = Vec::new();
    let tol = 1.0;
    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        if node.x < tol { left_nodes.push(node_id); }
        if (node.x - lx).abs() < tol { right_nodes.push(node_id); }
        if node.z < tol { bottom_nodes.push(node_id); }
        if node.y < tol { back_nodes.push(node_id); }
        if (node.y - ly).abs() < tol { front_nodes.push(node_id); }
    }

    for &node_id in &left_nodes {
        dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), -v_extension);
    }
    for &node_id in &right_nodes {
        dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), v_extension);
    }
    for &node_id in &bottom_nodes {
        dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), 0.0);
    }
    for &node_id in &back_nodes {
        dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0);
    }
    for &node_id in &front_nodes {
        dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0);
    }

    println!("  Extension: {:.2e} m/s ({:.2} cm/yr)",
        v_extension, v_extension * 365.25 * 24.0 * 3600.0 * 100.0);

    // Materials
    println!("\nSetting up materials...");
    let rho = 2700.0;
    let mu = 1.0e21;
    let cohesion = 44.0e6;
    let friction = 30.0_f64.to_radians();

    let material = mechanics::ElastoViscoPlastic::new(
        100e9,      // bulk_modulus
        0.25,       // poisson
        mu,         // viscosity
        cohesion,   // cohesion
        friction    // friction_angle
    ).with_softening(4.0e6, 0.5, 0.5);

    println!("  Single material: μ={:.1e} Pa·s, c={:.1} MPa", mu, cohesion/1e6);

    // Initial conditions
    println!("\nInitializing...");
    let velocity = vec![0.0; n_dofs];

    // Compute lithostatic pressure manually
    let mut element_pressures = vec![0.0; n_elements];
    for (elem_id, elem) in mesh.connectivity.tet10_elements.iter().enumerate() {
        let mut z_sum = 0.0;
        for &node_id in &elem.nodes {
            z_sum += mesh.geometry.nodes[node_id].z;
        }
        let depth = (lz - z_sum / 10.0).max(0.0);
        element_pressures[elem_id] = rho * 9.81 * depth;
    }

    let _elem_strains = vec![0.0; n_elements];
    let elem_mat_ids = vec![0u32; n_elements];  // All material 0 (u32 required)
    let gravity_vec = nalgebra::Vector3::new(0.0, 0.0, -9.81);

    // Non-dimensionalization scales (CRITICAL!)
    let scales = linalg::scaling::CharacteristicScales::new(
        lx,          // Length: 100 km
        v_extension, // Velocity: 1 cm/yr
        mu,          // Viscosity: 1e21 Pa·s
        rho,         // Density: 2700 kg/m³
        9.81         // Gravity
    );

    println!("\nNon-dimensionalization scales:");
    scales.print_summary();

    // GMRES solver (better for saddle-point systems)
    let mut linear_solver = linalg::iterative::GMRES::new()
        .with_restart(200)  // Larger restart for better convergence
        .with_max_iterations(8000)
        .with_tolerance(5e-3)  // Same as Picard benchmark for fair comparison
        .with_abs_tolerance(1e-8)
        .with_verbose(false)  // Turn off to see Newton convergence clearly
        .with_preconditioner(false);  // Use custom block preconditioner

    println!("\nLinear solver: GMRES");
    println!("  Max iterations: 8000");
    println!("  Restart: 200");
    println!("  Tolerance: 5e-3");
    println!("  Preconditioner: Block-Diagonal (ILU(0) for velocity, Identity for pressure)\n");

    // Newton-Krylov config
    let newton_config = linalg::jfnk::JFNKConfig {
        max_newton_iterations: 20,
        tolerance: 5e-4,      // Same as Picard for fair comparison
        abs_tolerance: 1e-9,  // Dimensionless absolute tolerance (very tight - forces relative check)
        max_line_search: 10,
        line_search_alpha: 1.0,
        line_search_rho: 0.5,
        verbose: true,        // Print Newton iterations
        use_amg: false,       // Use ILU(0) for velocity block (like Picard benchmark)
        amg_strength_threshold: 0.25,
    };

    println!("Nonlinear solver: Newton-Krylov (JFNK)");
    println!("  Max iterations: {}", newton_config.max_newton_iterations);
    println!("  Tolerance: {:.1e}", newton_config.tolerance);
    println!("  Preconditioner: Block-Diagonal (built internally)");
    println!("    - Velocity block: ILU(0)");
    println!("    - Pressure block: Scaled identity");
    println!("  Line search: enabled\n");

    println!("═══════════════════════════════════════════════════════════════");
    println!("  STARTING SOLVE - TIMESTEP 0");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Assembler closure (PHYSICAL units) - returns (K, f, energy)
    let materials = vec![material];  // Just one material
    let assembler = |v: &[f64]| {
        let (k_matrix, _, _) = fem::Assembler::assemble_stokes_vep_multimaterial_parallel(
            &mesh,
            &dof_mgr,
            &materials,
            &elem_mat_ids,
            v,
            &element_pressures,
        );

        // Gravity force
        let mut f = vec![0.0; n_dofs];
        for (_elem_id, elem) in mesh.connectivity.tet10_elements.iter().enumerate() {
            let mut nodes = [nalgebra::Point3::origin(); 10];
            for i in 0..10 {
                nodes[i] = mesh.geometry.nodes[elem.nodes[i]];
            }

            let f_elem = mechanics::BodyForce::gravity_load(&nodes, rho, &gravity_vec);
            for i in 0..10 {
                for comp in 0..3 {
                    let gi = dof_mgr.global_dof(elem.nodes[i], comp);
                    f[gi] += f_elem[3 * i + comp];
                }
            }
        }

        // Energy (optional, for monitoring)
        let energy = 0.0;

        (k_matrix, f, energy)
    };

    // Residual evaluator: R(u) = K(u)*u - f
    let residual_evaluator = |v: &[f64]| {
        let (k_matrix, f, _) = assembler(v);
        let mut residual = vec![0.0; n_dofs];

        // R = K*v - f
        for i in 0..n_dofs {
            residual[i] = -f[i];
        }
        for (row_idx, row) in k_matrix.outer_iterator().enumerate() {
            for (col_idx, &val) in row.iter() {
                residual[row_idx] += val * v[col_idx];
            }
        }

        // Apply boundary conditions to residual
        for i in 0..n_dofs {
            if dof_mgr.is_dirichlet(i) {
                residual[i] = 0.0;
            }
        }

        residual
    };

    // SOLVE with Newton-Krylov + GMRES + Block-Diagonal
    // jfnk_solve_nondimensional handles non-dimensionalization internally
    let start = std::time::Instant::now();

    let mut velocity_guess = velocity.clone();

    let (solution, newton_stats) = linalg::jfnk::jfnk_solve_nondimensional(
        assembler,
        residual_evaluator,
        &mut linear_solver,
        &mut velocity_guess,
        &dof_mgr,
        &newton_config,
        &scales,
    );

    let solve_time = start.elapsed();

    let _solution = solution;

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Newton iterations: {}", newton_stats.newton_iterations);
    println!("Converged: {}", newton_stats.converged);
    println!("Residual norm: {:.3e}", newton_stats.residual_norm);
    println!("Relative residual: {:.3e}", newton_stats.relative_residual);
    println!("Total linear iterations: {}", newton_stats.total_linear_iterations);
    println!("Avg linear iters/Newton: {:.1}",
        newton_stats.total_linear_iterations as f64 / newton_stats.newton_iterations as f64);
    println!("Total solve time: {:.2} seconds", solve_time.as_secs_f64());
    println!("═══════════════════════════════════════════════════════════════\n");

    if newton_stats.converged {
        println!("✓ SUCCESS: Newton-Krylov converged!");
        println!("\nComparison with Picard (from picard_simple):");
        println!("  - Newton should converge in fewer iterations (quadratic convergence)");
        println!("  - But each Newton iteration is more expensive (same linear solve cost)");
        println!("  - Newton is more robust for highly nonlinear problems");
    } else {
        println!("✗ FAILED: Did not converge");
        println!("\nDiagnostics needed:");
        println!("  - Check if line search is working");
        println!("  - Try smaller initial step (line_search_alpha = 0.5)");
        println!("  - Increase max_newton_iterations");
    }
}
