/// Benchmark: 3D Core Complex Formation
///
use geo_simulator::{
    ImprovedMeshGenerator, DofManager, Assembler, GMRES, VectorField,
    VtkWriter, ElastoViscoPlastic, TracerSwarm, SearchGrid, ScalarField,
    HillslopeDiffusion, assess_mesh_quality, smooth_mesh_auto, WinklerFoundation,
    picard_solve, PicardConfig, CharacteristicScales, AMGPreconditioner,
    SimulationConfig,
    compute_element_properties, compute_visualization_fields, update_tracer_properties, advect_tracers_and_mesh, compute_adaptive_timestep,
};
use geo_simulator::linalg::preconditioner::{ILUPreconditioner, IdentityPreconditioner, BlockDiagonalPreconditioner, JacobiPreconditioner};
use geo_simulator::ic_bc::{setup_boundary_conditions, setup_initial_tracers, get_material_properties, MaterialProps};
use nalgebra::{Point3, Vector3};
use std::time::Instant;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Geodynamic Benchmark: 3D Core Complex Formation");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ========================================================================
    // 1. Load Configuration from TOML
    // ========================================================================
    let args: Vec<String> = std::env::args().collect();
    let config_path = if args.len() > 1 {
        &args[1]
    } else {
        "inputs/core_complex_2d/config.toml"
    };

    let config = SimulationConfig::from_file(config_path)
        .unwrap_or_else(|e| {
            eprintln!("ERROR loading config: {}", e);
            std::process::exit(1);
        });

    // Extract key parameters from config
    let (lx, ly, lz) = (config.domain.lx, config.domain.ly, config.domain.lz);
    let v_extension = config.boundary_conditions.extension_rate_m_per_s;

    // Get material properties for crust (material ID 0)
    let mat_crust = get_material_properties(&config, 0);
    let mu_crust = mat_crust.viscosity;  // Used for viscosity scaling

    // Gravity and Density (from config)
    let g = config.initial_conditions.gravity;
    let rho_crust = mat_crust.density;
    let _rho_mantle = get_material_properties(&config, 1).density;
    let gravity_vec = Vector3::new(0.0, 0.0, -g);

    // Timestep configuration from config
    let dt_min_sec = config.time_stepping.dt_min_years * (365.25 * 24.0 * 3600.0);
    let dt_max_sec = config.time_stepping.dt_max_years * (365.25 * 24.0 * 3600.0);
    let dt_initial_sec = config.time_stepping.dt_initial_years * (365.25 * 24.0 * 3600.0);
    let use_adaptive = config.time_stepping.use_adaptive;
    let use_cfl_constraint = config.time_stepping.use_cfl_constraint;
    let use_maxwell_constraint = config.time_stepping.use_maxwell_constraint;
    let cfl_target = config.time_stepping.cfl_target;

    // Initial timestep
    let mut dt = dt_initial_sec;
    let total_time_sec = config.simulation.total_time_years * (365.25 * 24.0 * 3600.0);
    let max_steps = 20000; // Safety limit

    // Output and checkpoint intervals from config
    let output_interval_years = config.simulation.output_interval_years;
    let quality_check_years = config.simulation.quality_check_interval_years;

    // Print configuration summary
    config.print_summary();

    // ========================================================================
    // Non-dimensionalization Setup (CRITICAL for convergence!)
    // ========================================================================
    let scales = CharacteristicScales::new(
        lx,           // Length scale: 100 km (domain width)
        v_extension,  // Velocity scale: 1 cm/yr (extension rate)
        mu_crust,     // Viscosity scale (reference viscosity)
        rho_crust,    // Density scale: 2700 kg/m³
        g,            // Gravity: 9.81 m/s²
    );

    println!("\nSolver Configuration:");
    println!("  Non-linear Solver: Picard Iteration (Fixed-Point)");
    println!("  Scaling Strategy:  Non-dimensionalization (Physical → O(1))");
    println!("  Relaxation:        α=0.5 (Conservative)");
    println!("  Linear Solver:     GMRES (Restart=200, MaxIter=8000)");
    println!("  Linear Tolerance:  Rel=5e-3, Abs=1e-8 (Dimensionless)");
    println!("  Preconditioner:    Block-Diagonal (Saddle-Point)");
    println!("    - Velocity Block: AMG (Algebraic Multigrid)");
    println!("    - Pressure Block: Scaled Identity");

    // ========================================================================
    // 2. Mesh and Tracer Initialization (from config)
    // ========================================================================

    println!("\nGenerating Mesh and Tracers...");

    // Get grid dimensions from config
    let (nx, ny, nz) = config.grid_dimensions();

    println!("  Target Resolution: dx={}m, dy={}m, dz={}m", config.domain.dx, config.domain.dy, config.domain.dz);
    println!("  Grid Dimensions:   {} x {} x {} cells", nx, ny, nz);

    let mut mesh = ImprovedMeshGenerator::generate_cube(nx, ny, nz, lx, ly, lz);
    let n_elements = mesh.num_elements();

    // Initialize Tracers from config
    let tracers_per_elem = config.tracers.tracers_per_element;
    let mut swarm = TracerSwarm::with_capacity(n_elements * tracers_per_elem);

    // Distribute tracers uniformly within each element
    let nx_tracer = (tracers_per_elem as f64).powf(1.0/3.0).ceil() as usize;
    for elem_id in 0..n_elements {
        let elem = &mesh.connectivity.tet10_elements[elem_id];

        // Get element bounding box
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;

        for &node_id in &elem.nodes[0..4] {
            let p = &mesh.geometry.nodes[node_id];
            x_min = x_min.min(p.x);
            x_max = x_max.max(p.x);
            y_min = y_min.min(p.y);
            y_max = y_max.max(p.y);
            z_min = z_min.min(p.z);
            z_max = z_max.max(p.z);
        }

        // Place tracers
        for i in 0..nx_tracer {
            for j in 0..nx_tracer {
                for k in 0..nx_tracer {
                    let p = Point3::new(
                        x_min + (i as f64 + 0.5) * (x_max - x_min) / nx_tracer as f64,
                        y_min + (j as f64 + 0.5) * (y_max - y_min) / nx_tracer as f64,
                        z_min + (k as f64 + 0.5) * (z_max - z_min) / nx_tracer as f64,
                    );
                    swarm.add_tracer(p, 0); // Material ID assigned by setup_initial_tracers
                }
            }
        }
    }

    // Setup initial conditions (material IDs and plastic strain) from config
    setup_initial_tracers(&config, &mesh, &mut swarm);

    println!("  Mesh: {} elements | Tracers: {}", n_elements, swarm.num_tracers());

    // ========================================================================
    // 3. Materials Setup (from config)
    // ========================================================================

    // Create materials for all possible material IDs (0=upper, 1=lower, 2=weak zone)
    // Even if layers are disabled, we need the material objects to avoid index errors
    let mat_upper = get_material_properties(&config, 0);
    let mat_lower = get_material_properties(&config, 1);
    let mat_weak = if config.materials.weak_zone.enabled {
        MaterialProps {
            density: mat_lower.density,
            viscosity: config.materials.weak_zone.viscosity,
            cohesion: config.materials.weak_zone.cohesion_mpa * 1e6,
            cohesion_min: config.materials.weak_zone.cohesion_min_mpa * 1e6,
            friction_angle: mat_lower.friction_angle,
            shear_modulus: mat_lower.shear_modulus,
        }
    } else {
        mat_lower  // Use lower crust properties if weak zone disabled
    };

    let materials = vec![
        // Material 0: Upper crust
        ElastoViscoPlastic::new(100e9, 0.25, mat_upper.viscosity, mat_upper.cohesion, mat_upper.friction_angle)
            .with_softening(mat_upper.cohesion_min, 0.5, 0.5),
        // Material 1: Lower crust
        ElastoViscoPlastic::new(100e9, 0.25, mat_lower.viscosity, mat_lower.cohesion, mat_lower.friction_angle)
            .with_softening(mat_lower.cohesion_min, 0.5, 0.5),
        // Material 2: Weak zone (or lower crust if disabled)
        ElastoViscoPlastic::new(100e9, 0.25, mat_weak.viscosity, mat_weak.cohesion, mat_weak.friction_angle)
            .with_softening(mat_weak.cohesion_min, 0.5, 0.5),
    ];

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

    // ========================================================================
    // 4. Time Integration
    // ========================================================================

    let dof_mgr_init = DofManager::new_mixed(mesh.num_nodes(), &mesh.connectivity.corner_nodes());
    let n_dofs_total = dof_mgr_init.total_dofs();
    let mut current_sol = vec![0.0; n_dofs_total];

    // CRITICAL: Initialize velocity with non-zero guess based on BCs
    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        let x_normalized = node.x / lx;
        current_sol[dof_mgr_init.velocity_dof(node_id, 0)] = v_extension * (2.0 * x_normalized - 1.0);
    }


    let mut element_pressures = vec![0.0; n_elements];
    let mut _prev_picard_iters = 0;  // Track previous Picard iterations for adaptive config
    
    // Initial lithostatic pressure for step 0
    let rho_avg = rho_crust; // Single-layer model
    for (elem_id, elem) in mesh.connectivity.tet10_elements.iter().enumerate() {
        let mut z_sum = 0.0;
        for &node_id in &elem.nodes { z_sum += mesh.geometry.nodes[node_id].z; }
        let depth = (lz - z_sum / 10.0).max(0.0);
        element_pressures[elem_id] = rho_avg * g * depth;
    }
    
    std::fs::create_dir_all(&config.output.output_dir).ok();

    // ========================================================================
    // Initialize Surface Processes & Mesh Quality Tools
    // ========================================================================

    // Check INITIAL mesh quality (before any deformation)
    let initial_quality = assess_mesh_quality(&mesh);
    println!("\nInitial Mesh Quality:");
    println!("  {}", initial_quality.report());

    if !initial_quality.is_acceptable() {
        println!("  ERROR: Mesh is broken BEFORE simulation starts!");
        println!("  This is a bug in the quality assessment or mesh generator.");
        return;
    }

    // Winkler foundation (TEMPORARILY DISABLED to isolate solver issues)
    let mut winkler = WinklerFoundation::new(5e6);
    winkler.initialize_reference(&mesh);
    println!("Winkler foundation DISABLED for testing");

    let diffusion = HillslopeDiffusion::new(1e-6); // 1e-6 m²/yr (DES3D value)
    println!("Surface diffusion configured (κ = 1e-6 m²/yr)");

    // Parse verbose flag (simple check)
    let args: Vec<String> = std::env::args().collect();
    let verbose = args.contains(&"--verbose".to_string());

    // ========================================================================
    // OUTPUT STEP 0: Initial Condition & Boundary Condition Check
    // ========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  STEP 0: Initial Condition Check");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  BC: Extension {:.2} cm/yr, Plane strain Y, Fixed Z-bottom, Free Z-top",
        config.boundary_conditions.extension_rate_cm_per_year);
    println!("  Tracers: {} (weak zone: {} with ε_p > 0)", swarm.num_tracers(),
        swarm.plastic_strain.iter().filter(|&&ps| ps > 0.0).count());

    // Create comprehensive IC/BC output with all fields
    {
        let grid_ic = SearchGrid::build(&mesh, [10, 10, 10]);
        let (elem_mat_ids_ic, elem_strains_ic) = swarm.get_element_properties(&mesh, &grid_ic);

        let mut viz_mesh = mesh.clone();

        // Add velocity field showing boundary conditions
        // Left: -v, Right: +v, Everything else: zero
        let mut vel_bc = vec![0.0; mesh.num_nodes() * 3];
        for &node_id in &left_nodes {
            vel_bc[node_id * 3] = -v_extension;  // x-component
        }
        for &node_id in &right_nodes {
            vel_bc[node_id * 3] = v_extension;   // x-component
        }
        viz_mesh.field_data.add_vector_field(VectorField::from_dof_vector("Velocity", &vel_bc));

        // Add nodal pressure (initial lithostatic)
        let mut p_nodal = vec![0.0; mesh.num_nodes()];
        for (elem_id, elem) in mesh.connectivity.tet10_elements.iter().enumerate() {
            let elem_pressure = element_pressures[elem_id];
            for &node_id in &elem.nodes {
                p_nodal[node_id] = elem_pressure;
            }
        }
        viz_mesh.field_data.add_field(ScalarField::new("PressureNodal", p_nodal));

        // Add element-based data
        viz_mesh.cell_data.add_field(ScalarField::new("MaterialID", elem_mat_ids_ic.iter().map(|&id| id as f64).collect()));
        viz_mesh.cell_data.add_field(ScalarField::new("PlasticStrain", elem_strains_ic.clone()));
        viz_mesh.cell_data.add_field(ScalarField::new("Pressure", element_pressures.clone()));
        viz_mesh.cell_data.add_field(ScalarField::new("StrainRate_II", vec![0.0; n_elements]));
        viz_mesh.cell_data.add_field(ScalarField::new("Stress_II", vec![0.0; n_elements]));
        viz_mesh.cell_data.add_field(ScalarField::new("Viscosity", vec![mu_crust; n_elements]));

        // Additional diagnostic fields for visualization (initial state - no yielding yet)
        viz_mesh.cell_data.add_field(ScalarField::new("IsYielding", vec![0.0; n_elements]));

        // Compute initial yield stress and cohesion for each element
        let mut initial_yield_stress = vec![0.0; n_elements];
        let mut initial_cohesion = vec![0.0; n_elements];
        for (elem_id, _elem) in mesh.connectivity.tet10_elements.iter().enumerate() {
            let mat_id = elem_mat_ids_ic[elem_id] as usize;
            let p_elem = element_pressures[elem_id];
            let c = materials[mat_id].plasticity.cohesion;
            let phi = materials[mat_id].plasticity.friction_angle;
            initial_yield_stress[elem_id] = c * phi.cos() + p_elem * phi.sin();
            initial_cohesion[elem_id] = c;
        }
        viz_mesh.cell_data.add_field(ScalarField::new("YieldStress", initial_yield_stress));
        viz_mesh.cell_data.add_field(ScalarField::new("PlasticStrainRate", vec![0.0; n_elements]));
        viz_mesh.cell_data.add_field(ScalarField::new("SoftenedCohesion", initial_cohesion));
        viz_mesh.cell_data.add_field(ScalarField::new("PlasticViscosity", vec![1e30; n_elements]));

        let ic_filename = format!("{}/step_0000.vtu", config.output.output_dir);
        VtkWriter::write_combined_vtu(&viz_mesh, &swarm, &ic_filename).unwrap();
    }

    println!("  ✓ Written: {}/step_0000.vtu", config.output.output_dir);
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("\nStarting Tectonic Evolution...");
    let start_sim = Instant::now();

    // Time-based output tracking
    let mut next_output_time = output_interval_years;  // First output at interval, not at step 0
    let mut next_quality_check_time = 0.0;

    // Adaptive timestepping state
    let mut current_time_sec = 0.0;
    let mut step = 0;

    // Calculate ramp steps for spin-up tolerance (approximate, will adapt)
    let dt_years_initial = dt / (365.25 * 24.0 * 3600.0);
    let ramp_steps = (config.boundary_conditions.ramp_duration_years / dt_years_initial).ceil() as usize;

    while current_time_sec < total_time_sec && step <= max_steps {
        let current_time_years = current_time_sec / (365.25 * 24.0 * 3600.0);
        // 1. Setup BCs for current mesh state
        let mut dof_mgr = DofManager::new_mixed(mesh.num_nodes(), &mesh.connectivity.corner_nodes());

        // Apply boundary conditions from config
        // Use current time for ramp-up (config specifies ramp duration)
        setup_boundary_conditions(&config, &mesh, &mut dof_mgr, current_time_years);

        if step < 20 && verbose {
            let ramp_fraction = current_time_years / config.boundary_conditions.ramp_duration_years;
            println!("    Apply Velocity Ramp: {:.1}% (t = {:.1} kyr)", ramp_fraction * 100.0, current_time_years / 1000.0);
        }

        // Pin one pressure DOF to remove the constant pressure null space
        let corners = mesh.connectivity.corner_nodes();
        if !corners.is_empty() {
            if let Some(p_dof) = dof_mgr.pressure_dof(corners[0]) {
                dof_mgr.set_dirichlet(p_dof, 0.0);
            }
        }

        let _n_dofs = dof_mgr.total_dofs();

        // 2. Map Material Properties (M2E)
        let grid = SearchGrid::build(&mesh, [10, 10, 10]);
        let (elem_mat_ids, elem_strains) = swarm.get_element_properties(&mesh, &grid);

        // 3. Setup Picard Iteration solver

        // Picard config: Use working parameters from picard_simple.rs
        let picard_config = PicardConfig {
            max_iterations: 30,      // Sufficient for convergence
            tolerance: 5e-4,         // Relative change tolerance
            relaxation: 0.5,         // Conservative (CRITICAL for stability)
            abs_tolerance: 1e-15,    // Absolute tolerance guard
        };

        // Linear solver: GMRES - use working parameters
        let mut linear_solver = GMRES::new()
            .with_restart(200)              // Larger restart for saddle-point
            .with_max_iterations(8000)      // Increased for later Picard iterations
            .with_tolerance(5e-3)           // Tightened to help Picard convergence
            .with_abs_tolerance(1e-8)       // Dimensionless absolute tolerance
            .with_verbose(false)            // Turn off to see Picard convergence clearly
            .with_preconditioner(false);    // Use custom block preconditioner

        // Assembler closure: Returns (K, f, _) for full saddle-point system
        let assembler = |sol: &[f64]| {
             let (k_matrix, mut f_total, _) = Assembler::assemble_stokes_vep_multimaterial_parallel(
                &mesh, &dof_mgr, &materials, &elem_mat_ids, sol, &elem_strains
            );

            // Gravity RHS (only for velocity DOFs)
            for (elem_id, _mat_id) in elem_mat_ids.iter().enumerate() {
                let rho = rho_crust; 
                let elem_nodes = &mesh.connectivity.tet10_elements[elem_id].nodes;
                let mut nodes = [Point3::origin(); 10];
                for i in 0..10 { nodes[i] = mesh.geometry.nodes[elem_nodes[i]]; }

                let f_elem = geo_simulator::mechanics::BodyForce::gravity_load(&nodes, rho, &gravity_vec);
                for i in 0..10 {
                    for comp in 0..3 {
                        let gi = dof_mgr.velocity_dof(elem_nodes[i], comp);
                        f_total[gi] += f_elem[3 * i + comp];
                    }
                }
            }

            (k_matrix, f_total, 0.0)
        };

        // 4. Setup Block-Diagonal Preconditioner Factory
        // Extract velocity DOFs for block structure
        let num_vel_dofs = dof_mgr.total_vel_dofs();

        let precond_factory = |matrix: &sprs::CsMat<f64>| -> Box<dyn geo_simulator::linalg::Preconditioner> {
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
            // Use AMG for velocity block (much better for Stokes saddle-point systems)
            let vel_precond = match AMGPreconditioner::new(&vel_block, 10, 500, 0.25) {
                Ok(amg) => amg,
                Err(e) => {
                    eprintln!("    WARNING: AMG failed for velocity block: {}. Falling back to ILU(0).", e);
                    match ILUPreconditioner::new(&vel_block) {
                        Ok(ilu) => return Box::new(BlockDiagonalPreconditioner::new(ilu, IdentityPreconditioner, num_vel_dofs)),
                        Err(e2) => {
                            eprintln!("    WARNING: ILU(0) also failed: {}. Using Jacobi.", e2);
                            return Box::new(JacobiPreconditioner::new(matrix));
                        }
                    }
                }
            };

            // For pressure: use identity (pressure block is zero in saddle-point)
            let press_precond = IdentityPreconditioner;

            Box::new(BlockDiagonalPreconditioner::new(
                vel_precond,
                press_precond,
                num_vel_dofs,
            ))
        };

        // 5. Wrap assembler with Non-dimensionalization
        // Picard needs (K, f) signature, not (K, f, energy)
        let assembler_nd = |sol_nd: &[f64]| {
            // Convert to physical
            let sol_phys = scales.dim_solution(sol_nd, &dof_mgr);

            // Assemble in physical units
            let (k_phys, f_phys, _) = assembler(&sol_phys);

            // Convert to dimensionless
            let k_nd = scales.nondim_matrix(&k_phys, &dof_mgr);
            let f_nd = scales.nondim_residual(&f_phys, &dof_mgr);

            (k_nd, f_nd)
        };

        // Convert current solution to dimensionless
        let mut current_sol_nd = scales.nondim_solution(&current_sol, &dof_mgr);

        // 6. Run Picard Solver with Non-dimensionalization
        // This converts physical units → O(1) dimensionless → solve → convert back
        // CRITICAL: Without this, matrix has 30+ orders of magnitude variation!

        if verbose {
            println!("    Running Picard iteration...");
        }

        let (sol_nd, picard_stats) = picard_solve(
            assembler_nd,
            &mut linear_solver,
            &mut current_sol_nd,
            &dof_mgr,
            &picard_config,
            Some(precond_factory),
        );

        // Convert back to physical
        let sol_new = scales.dim_solution(&sol_nd, &dof_mgr);

        // Report convergence
        if !picard_stats.converged {
            println!("  Step {}: WARNING - Picard did not converge! Iters: {}, Rel Change: {:.3e}",
                     step, picard_stats.iterations, picard_stats.relative_change);
            println!("    Last linear solve: converged={}, iterations={}, residual={:.3e}",
                     picard_stats.last_linear_stats.converged,
                     picard_stats.last_linear_stats.iterations,
                     picard_stats.last_linear_stats.residual_norm);
        } else if verbose {
            println!("    Picard converged: Iters={}, Rel Change={:.3e}, Lin Iters={}",
                     picard_stats.iterations, picard_stats.relative_change,
                     picard_stats.total_linear_iterations);
        }

        current_sol = sol_new;
        let _prev_picard_iters = picard_stats.iterations;  // Track iterations

        // 2b. Compute Adaptive Timestep (if enabled)
        if use_adaptive && step > 0 {
            let adaptive_dt_info = compute_adaptive_timestep(
                &mesh,
                &dof_mgr,
                &current_sol,
                &materials,
                cfl_target,
                dt_min_sec,
                dt_max_sec,
                use_cfl_constraint,
                use_maxwell_constraint,
            );

            dt = adaptive_dt_info.dt;

            if verbose && step < 20 {
                println!("    Adaptive dt: {:.1} yr (CFL: {:.1} yr, Maxwell: {:.2e} yr, v_max: {:.2e} m/s)",
                    dt / (365.25 * 24.0 * 3600.0),
                    adaptive_dt_info.cfl_dt / (365.25 * 24.0 * 3600.0),
                    adaptive_dt_info.maxwell_dt / (365.25 * 24.0 * 3600.0),
                    adaptive_dt_info.max_velocity);
            }
        }

        // 3. Compute Physical Invariants and Update Plastic Strain
        // Use standardized update functions from updates module
        let elem_props = compute_element_properties(
            &mesh,
            &dof_mgr_init,
            &current_sol,
            &materials,
            &elem_mat_ids,
            &elem_strains,
        );

        // Extract element properties for backward compatibility
        let mut sr_ii = vec![0.0; mesh.num_elements()];
        let mut stress_ii = vec![0.0; mesh.num_elements()];
        let mut viscosity_eff = vec![0.0; mesh.num_elements()];
        let mut element_pressures_new = vec![0.0; mesh.num_elements()];
        for (elem_id, props) in elem_props.iter().enumerate() {
            sr_ii[elem_id] = props.strain_rate_ii;
            stress_ii[elem_id] = props.stress_ii;
            viscosity_eff[elem_id] = props.effective_viscosity;
            element_pressures_new[elem_id] = props.pressure;
        }
        element_pressures = element_pressures_new;

        // Compute visualization fields
        let viz = compute_visualization_fields(
            &mesh,
            &materials,
            &elem_mat_ids,
            &elem_strains,
            &elem_props,
        );
        let is_yielding = viz.is_yielding;
        let yield_stress = viz.yield_stress;
        let plastic_strain_rate = viz.plastic_strain_rate;
        let softened_cohesion = viz.softened_cohesion;
        let plastic_viscosity = viz.plastic_viscosity;

        // 4. Advect Tracers & 5. Update Mesh Nodes
        // CRITICAL: Safety Guard
        // - Ramp-up Phase (Step < 10): Force update even if not fully converged.
        //   The system needs time to adjust to the onset of plasticity. As long as the linear solver works,
        //   we can "fail forward" to find the stable basin.
        // - Production (Step >= 10): Strict Check. If solver fails, HALT.
        if picard_stats.converged || step < ramp_steps {
            if step < ramp_steps && !picard_stats.converged {
                 println!("    ⚠ Spin-up Warning (Step {}): Forcing update to settle plasticity.", step);
            }

            // Update tracer properties (including plastic strain) using standardized function
            update_tracer_properties(
                &mesh,
                &mut swarm,
                &materials,
                &elem_mat_ids,
                &elem_props,
                dt,
            );

            // Advect tracers and update mesh nodes using standardized function
            advect_tracers_and_mesh(
                &mut mesh,
                &mut swarm,
                &dof_mgr,
                &current_sol,
                dt,
            );
        } else {
             println!("\n🛑 FATAL ERROR: Solver failed at Step {}. Halting simulation to prevent divergence.", step);
             break; // Terminate Main Loop
        }

        // 5a. Mesh Quality Check & Smoothing (Time-based)
        if current_time_years >= next_quality_check_time || step == 0 {
            let quality = assess_mesh_quality(&mesh);
            if verbose {
                println!("       [{:.2} kyr] Mesh Quality: {}", current_time_years / 1000.0, quality.report());
            }
            next_quality_check_time = current_time_years + quality_check_years;

            // Apply smoothing if mesh quality degrading
            if quality.needs_smoothing() {
                println!("       ⚠ Mesh quality degraded, applying Laplacian smoothing...");
                let (before, after) = smooth_mesh_auto(&mut mesh, 5, 0.6);
                println!("         BEFORE: {}", before.report());
                println!("         AFTER:  {}", after.report());

                if after.num_inverted > 0 {
                    println!("         ✗ WARNING: Still {} inverted elements after smoothing!", after.num_inverted);
                } else if after.min_jacobian > before.min_jacobian * 1.1 {
                    println!("         ✓ Smoothing successful (min_J improved by {:.1}%)",
                             (after.min_jacobian - before.min_jacobian) / before.min_jacobian * 100.0);
                } else {
                    println!("         ✓ Smoothing completed");
                }
            }
        }

        // 5b. Surface Diffusion (every 10 steps, ~50 kyr)
        if step % 10 == 0 && step > 0 {
            let dt_diffusion = 10.0 * dt / (365.25 * 24.0 * 3600.0); // Convert to years
            diffusion.apply_diffusion(&mut mesh, dt_diffusion);
        }

        // 6. Element pressures are updated from the solution in the invariants step

        // 7. Update Internal Strain on Tracers (Simplified: use element strain)
        // In a full implementation, we'd interpolate L_dot back to racers.
        // For now, elements carry the softened state derived from majority.

        let time_my = current_time_years / 1e6;

        // Output and VTK saving (time-based)
        // Note: step 0 is already output before the loop with proper BC visualization
        let is_final_step = current_time_sec + dt >= total_time_sec;
        let should_output = current_time_years >= next_output_time || is_final_step;

        if should_output {
            println!("⏱  {:.2} MY ({:6} steps) | Picard: {:2} | LinIter: {:5} | RelChg: {:.2e} | Mesh: {}/{} OK",
                time_my,
                step,
                picard_stats.iterations,
                picard_stats.total_linear_iterations,
                picard_stats.relative_change,
                mesh.num_elements() - assess_mesh_quality(&mesh).num_inverted,
                mesh.num_elements()
            );
            next_output_time = current_time_years + output_interval_years;
            let filename = format!("{}/step_{:04}.vtu", config.output.output_dir, step);
            
            let mut viz_mesh = mesh.clone();
            // Extract velocity part for VectorField
            let sol_ref = &current_sol;
            let dof_ref = &dof_mgr;
            let vel_only: Vec<f64> = (0..mesh.num_nodes())
                .flat_map(|n| {
                    (0..3).map(move |c| sol_ref[dof_ref.velocity_dof(n, c)])
                })
                .collect();
            viz_mesh.field_data.add_vector_field(VectorField::from_dof_vector("Velocity", &vel_only));
            
            // Extract nodal pressure for ScalarField
            let mut p_nodal = vec![0.0; mesh.num_nodes()];
            for (node_id, _) in mesh.geometry.nodes.iter().enumerate() {
                if dof_mgr.pressure_node_map()[node_id].is_some() {
                    if let Some(dof) = dof_mgr.pressure_dof(node_id) {
                         p_nodal[node_id] = current_sol[dof];
                    }
                }
            }
            viz_mesh.field_data.add_field(ScalarField::new("PressureNodal", p_nodal));
            
            // Add element-based data (CellData)
            viz_mesh.cell_data.add_field(ScalarField::new("MaterialID", elem_mat_ids.iter().map(|&id| id as f64).collect()));
            viz_mesh.cell_data.add_field(ScalarField::new("PlasticStrain", elem_strains.clone()));
            viz_mesh.cell_data.add_field(ScalarField::new("Pressure", element_pressures.clone()));
            viz_mesh.cell_data.add_field(ScalarField::new("StrainRate_II", sr_ii.clone()));
            viz_mesh.cell_data.add_field(ScalarField::new("Stress_II", stress_ii.clone()));
            viz_mesh.cell_data.add_field(ScalarField::new("Viscosity", viscosity_eff.clone()));

            // Additional diagnostic fields for visualization
            viz_mesh.cell_data.add_field(ScalarField::new("IsYielding", is_yielding.clone()));
            viz_mesh.cell_data.add_field(ScalarField::new("YieldStress", yield_stress.clone()));
            viz_mesh.cell_data.add_field(ScalarField::new("PlasticStrainRate", plastic_strain_rate.clone()));
            viz_mesh.cell_data.add_field(ScalarField::new("SoftenedCohesion", softened_cohesion.clone()));
            viz_mesh.cell_data.add_field(ScalarField::new("PlasticViscosity", plastic_viscosity.clone()));

            VtkWriter::write_combined_vtu(&viz_mesh, &swarm, &filename).unwrap();
        }

        // Advance time and step counter
        current_time_sec += dt;
        step += 1;
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Simulation Completed in {:?}", start_sim.elapsed());
    println!("  Results saved to {}/", config.output.output_dir);
    println!("═══════════════════════════════════════════════════════════════");
}

