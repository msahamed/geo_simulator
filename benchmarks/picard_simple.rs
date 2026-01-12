/// Simple Picard + GMRES + Block-Diagonal benchmark for core complex
///
/// Purpose: Test if Picard iteration with GMRES + block preconditioner converges when JFNK fails
/// Strategy: Strip out all complexity, use robust solver stack for saddle-point systems
///
/// Solver Stack:
/// - Nonlinear: Picard iteration (fixed-point)
/// - Linear: GMRES with block-diagonal preconditioner (ILU(0) for velocity, Identity for pressure)
/// - Scaling: Non-dimensionalization (critical!)

use geo_simulator::*;
use geo_simulator::ImprovedMeshGenerator;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SIMPLE TEST: Picard + GMRES + Block-Diagonal");
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
    let mut mesh = ImprovedMeshGenerator::generate_cube(nx, ny, nz, lx, ly, lz);

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
    let mut velocity = vec![0.0; n_dofs];

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

    let elem_strains = vec![0.0; n_elements];
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
        .with_max_iterations(8000)  // Increased for later Picard iterations
        .with_tolerance(5e-3)  // Tightened to reduce inexact solve error
        .with_abs_tolerance(1e-8)  // Tightened
        .with_verbose(false)  // Turn off to see Picard convergence clearly
        .with_preconditioner(false);  // Disable built-in preconditioner

    println!("\nLinear solver: GMRES");
    println!("  Max iterations: 8000");
    println!("  Restart: 200");
    println!("  Tolerance: 5e-3 (tightened to help Picard convergence)");
    println!("  Preconditioner: Block-Diagonal (ILU(0) for velocity, Identity for pressure)\n");

    // Picard config
    let picard_config = linalg::picard::PicardConfig {
        max_iterations: 30,  // Increase iterations for conservative relaxation
        tolerance: 5e-4,
        relaxation: 0.5,  // Very conservative - smaller steps but more stable
        abs_tolerance: 1e-15,
    };

    println!("Nonlinear solver: Picard iteration");
    println!("  Max iterations: {}", picard_config.max_iterations);
    println!("  Tolerance: {:.1e}", picard_config.tolerance);
    println!("  Relaxation: {:.2}\n", picard_config.relaxation);

    println!("═══════════════════════════════════════════════════════════════");
    println!("  STARTING SOLVE - TIMESTEP 0");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Assembler closure (PHYSICAL units)
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
        for (elem_id, elem) in mesh.connectivity.tet10_elements.iter().enumerate() {
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

        (k_matrix, f)
    };

    // Wrap assembler with NON-DIMENSIONALIZATION
    let assembler_nd = |v_nd: &[f64]| {
        // Convert to physical
        let v_phys = scales.dim_solution(v_nd, &dof_mgr);

        // Assemble in physical units
        let (k_phys, f_phys) = assembler(&v_phys);

        // Convert to dimensionless
        let k_nd = scales.nondim_matrix(&k_phys, &dof_mgr);
        let f_nd = scales.nondim_residual(&f_phys, &dof_mgr);

        (k_nd, f_nd)
    };

    // Convert initial guess to dimensionless
    let mut velocity_nd = scales.nondim_solution(&velocity, &dof_mgr);

    // SOLVE with Picard + GMRES + Block-Diagonal
    let start = std::time::Instant::now();

    // Preconditioner factory: creates Block-Diagonal preconditioner for saddle-point system
    // Block structure: [A B^T; B 0] where A is velocity (nv×nv), pressure block is zero
    let num_vel_dofs = dof_mgr.total_vel_dofs();

    let precond_factory = |matrix: &sprs::CsMat<f64>| -> Box<dyn linalg::preconditioner::Preconditioner> {
        use sprs::TriMat;

        // Manually extract velocity block A (top-left nv×nv)
        let mut vel_triplets = TriMat::new((num_vel_dofs, num_vel_dofs));
        for i in 0..num_vel_dofs {
            for (col, val) in matrix.outer_iterator().nth(i).unwrap().iter() {
                if col < num_vel_dofs {
                    vel_triplets.add_triplet(i, col, *val);
                }
            }
        }
        let vel_block = vel_triplets.to_csr();

        // Create preconditioners for each block
        let vel_precond = match linalg::preconditioner::ILUPreconditioner::new(&vel_block) {
            Ok(ilu) => ilu,
            Err(e) => {
                eprintln!("ILU(0) failed for velocity block: {}. Using Jacobi.", e);
                return Box::new(linalg::preconditioner::JacobiPreconditioner::new(matrix));
            }
        };

        // For pressure: use identity (pressure block is zero in saddle-point)
        let press_precond = linalg::preconditioner::IdentityPreconditioner;

        Box::new(linalg::preconditioner::BlockDiagonalPreconditioner::new(
            vel_precond,
            press_precond,
            num_vel_dofs,
        ))
    };

    let (sol_nd, picard_stats) = linalg::picard::picard_solve(
        assembler_nd,
        &mut linear_solver,
        &mut velocity_nd,
        &dof_mgr,
        &picard_config,
        Some(precond_factory),
    );

    let solve_time = start.elapsed();

    // Convert back to physical
    let solution = scales.dim_solution(&sol_nd, &dof_mgr);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Picard iterations: {}", picard_stats.iterations);
    println!("Converged: {}", picard_stats.converged);
    println!("Relative change: {:.3e}", picard_stats.relative_change);
    println!("Total linear iterations: {}", picard_stats.total_linear_iterations);
    println!("Avg linear iters/Picard: {:.1}",
        picard_stats.total_linear_iterations as f64 / picard_stats.iterations as f64);
    println!("Total solve time: {:.2} seconds", solve_time.as_secs_f64());
    println!("═══════════════════════════════════════════════════════════════\n");

    if picard_stats.converged {
        println!("✓ SUCCESS: Picard+GMRES+BlockDiagonal converged!");
        println!("\nNext steps:");
        println!("  1. Run multiple timesteps");
        println!("  2. Test with multi-material configuration");
        println!("  3. Migrate to full core_complex simulation");
    } else {
        println!("✗ FAILED: Did not converge");
        println!("\nDiagnostics needed:");
        println!("  - Check GMRES residual behavior");
        println!("  - Try adjusting restart parameter");
        println!("  - Try increasing max iterations");
    }
}
