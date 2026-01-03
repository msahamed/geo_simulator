/// Benchmark: 3D Core Complex Formation
///
use geo_simulator::{
    ImprovedMeshGenerator, DofManager, Assembler, GMRES, VectorField,
    VtkWriter, ElastoViscoPlastic, TracerSwarm, SearchGrid, ScalarField, JFNKConfig, GaussQuadrature, Tet10Basis, StrainDisplacement,
    HillslopeDiffusion, assess_mesh_quality, smooth_mesh_auto, WinklerFoundation,
    picard_solve, PicardConfig, CharacteristicScales,
    SimulationConfig,
};
use geo_simulator::ic_bc::{setup_boundary_conditions, setup_initial_tracers, get_material_properties};
use nalgebra::{Point3, Vector3};
use std::time::Instant;
use rayon::prelude::*;

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
    let mu_crust = mat_crust.viscosity;
    let c0 = mat_crust.cohesion;
    let cmin = mat_crust.cohesion_min;
    let phi = mat_crust.friction_angle;

    // Gravity and Density (from config)
    let g = config.initial_conditions.gravity;
    let rho_crust = mat_crust.density;
    let rho_mantle = get_material_properties(&config, 1).density;
    let gravity_vec = Vector3::new(0.0, 0.0, -g);

    // Timestep and duration from config
    let dt = config.time_stepping.dt_initial_years * (365.25 * 24.0 * 3600.0); // Convert years to seconds
    let n_steps = ((config.simulation.total_time_years / config.time_stepping.dt_initial_years) as usize).min(20000);

    // Output and checkpoint intervals from config
    let output_interval_years = config.simulation.output_interval_years;
    let quality_check_years = config.simulation.quality_check_interval_years;

    // Print configuration summary
    config.print_summary();

    // ========================================================================
    // Non-dimensionalization Setup (CRITICAL for convergence!)
    // ========================================================================
    let _scales = CharacteristicScales::new(
        lx,           // Length scale: 100 km (domain width)
        v_extension,  // Velocity scale: 1 cm/yr (extension rate)
        mu_crust,     // Viscosity scale: 1e24 Pa·s (reference viscosity)
        rho_crust,    // Density scale: 2700 kg/m³
        g,            // Gravity: 9.81 m/s²
    );

    println!("\nSolver Configuration:");
    println!("  Non-linear Solver: JFNK (Jacobian-Free Newton-Krylov)");
    println!("  Scaling Strategy:  Non-dimensionalization (Physical → O(1))");
    println!("  Perturbation:      Dimensionless (Scaled Space)");
    println!("  Linear Solver:     GMRES (Restart=200, MaxIter=2000)");
    println!("  Linear Tolerance:  Rel=1e-4, Abs=1e-9 (Dimensionless)");
    println!("  Preconditioner:    Block Triangular (Upper Schur)");
    println!("    - Velocity Block: ILU (Incomplete LU) - Robust but Slower Setup");
    println!("    - Pressure Block: Inverse Viscosity Approximation");

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
    // 3. Materials Setup
    // ========================================================================

    let materials = vec![
        // Single material: EVP with DES3D parameters
        // Softening: c0=44MPa → c1=4MPa over εp=0→0.5
        ElastoViscoPlastic::new(100e9, 0.25, mu_crust, c0, phi)
            .with_softening(cmin, 0.5, 0.5), // Fully softened at εp=0.5 (DES3D: pls1)
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

        let ic_filename = format!("{}/step_0000.vtu", config.output.output_dir);
        VtkWriter::write_combined_vtu(&viz_mesh, &swarm, &ic_filename).unwrap();
    }

    println!("  ✓ Written: {}/step_0000.vtu", config.output.output_dir);
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("\nStarting Tectonic Evolution...");
    let start_sim = Instant::now();

    // Time-based output tracking
    let dt_years = dt / (365.25 * 24.0 * 3600.0);
    let mut next_output_time = output_interval_years;  // First output at interval, not at step 0
    let mut next_quality_check_time = 0.0;

    // Calculate ramp steps for spin-up tolerance
    let ramp_steps = (config.boundary_conditions.ramp_duration_years / dt_years).ceil() as usize;

    for step in 0..=n_steps {
        let current_time_years = step as f64 * dt_years;
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

        // 3. Setup JFNK (Jacobian-Free Newton-Krylov) solver

        // JFNK config: Optimized for viscoplastic geodynamics
        let mut jfnk_config = JFNKConfig::conservative();
        jfnk_config.tolerance = 5e-4;       // Slightly relaxed for speed (0.05% relative error)
        jfnk_config.abs_tolerance = 1e-7;   // Relaxed absolute tolerance
        jfnk_config.max_newton_iterations = 20; // Reduced - converging in 5-6 iterations
        jfnk_config.verbose = verbose;

        // Linear solver: GMRES with balanced settings
        let mut linear_solver = GMRES::new()
            .with_restart(180)              // Slightly reduced from 200
            .with_max_iterations(1200)      // Reduced from 2000, but not too aggressive
            .with_tolerance(1e-4)           // Standard tolerance
            .with_abs_tolerance(1e-9)       // Absolute floor
            .with_verbose(verbose);

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

        let residual_evaluator = |sol: &[f64]| {
            Assembler::compute_stokes_residual_parallel(
                &mesh, &dof_mgr, &materials, &elem_mat_ids, sol, &elem_strains, &gravity_vec, rho_crust, rho_mantle
            )
        };

        // 4. Picard Warm Start (Critical for Step 0)
        // Use Picard to get past the initial non-linear "shock"
        if step == 0 {
            if verbose { println!("    Running Picard Warm Start for Step 0..."); }
            
            // Adapter for Picard assembler signature: needs to return (K, f)
            // (Note: We need a fresh assembler closure because the previous one is moved/borrowed or we can just reuse code structure)
            // Ideally we'd reuse 'assembler' but it returns a triplet. Let's make a wrapper.
            let picard_assembler = |sol: &[f64]| {
                 let (k, f, _) = (assembler)(sol);
                 (k, f)
            };

            let picard_config = PicardConfig {
                max_iterations: 10,       // Just enough to stabilize
                tolerance: 1e-2,          // Loose tolerance is fine for pre-solve
                relaxation: 0.5,          // Conservative
                abs_tolerance: 1e-9,
            };

            let (warm_sol, stats) = picard_solve(
                picard_assembler,
                &mut linear_solver,
                &mut current_sol,
                &dof_mgr,
                &picard_config,
                None::<fn(&sprs::CsMat<f64>) -> Box<dyn geo_simulator::linalg::Preconditioner>>
            );
            
            // Update solution with the warmed-up result
            current_sol = warm_sol;

            if verbose {
                println!("    Picard Warmup: Iters={}, Converged={}, ||R_rel||={:.2e}",
                    stats.iterations, stats.converged, stats.relative_change);
            }
        }
        
        // 5. Run JFNK Solver
        // NOTE: Using built-in equilibration instead of explicit nondimensionalization
        // to avoid double-scaling issues. Equilibration brings system to O(1) automatically.

        // Solve nonlinear system
        let (sol_new, jfnk_stats) = geo_simulator::jfnk_solve(
            assembler,
            residual_evaluator,
            &mut linear_solver,
            &mut current_sol,
            &dof_mgr,
            &jfnk_config,
        );

        // Report convergence
        if !jfnk_stats.converged {
            println!("  Step {}: WARNING - JFNK did not converge! Iters: {}, Residual: {:.3e}",
                     step, jfnk_stats.newton_iterations, jfnk_stats.residual_norm);
            println!("    Last linear solve: converged={}, iterations={}, residual={:.3e}",
                     jfnk_stats.last_linear_stats.converged,
                     jfnk_stats.last_linear_stats.iterations,
                     jfnk_stats.last_linear_stats.residual_norm);
        }

        current_sol = sol_new;
        let _prev_picard_iters = jfnk_stats.newton_iterations;  // Track iterations

        // 3. Compute Physical Invariants and Update Plastic Strain
        // OPTIMIZED: Parallelize with rayon for 2-3x speedup
        let elem_properties: Vec<_> = (0..mesh.num_elements()).into_par_iter().map(|elem_id| {
            let elem = &mesh.connectivity.tet10_elements[elem_id];
            let mut nodes_elem = [Point3::origin(); 10];
            for i in 0..10 { nodes_elem[i] = mesh.geometry.nodes[elem.nodes[i]]; }

            // Compute average strain rate
            let mut strain_rate = nalgebra::SMatrix::<f64, 6, 1>::zeros();
            let quad = GaussQuadrature::tet_4point();
            for qp in &quad.points {
                let b = StrainDisplacement::compute_b_at_point(qp, &nodes_elem);
                let mut v_elem = nalgebra::SMatrix::<f64, 30, 1>::zeros();
                for i in 0..10 {
                    for comp in 0..3 {
                        v_elem[3 * i + comp] = current_sol[dof_mgr_init.velocity_dof(elem.nodes[i], comp)];
                    }
                }
                strain_rate += b * v_elem;
            }
            strain_rate /= quad.points.len() as f64;

            let j2_edot = 0.5 * (
                strain_rate[0]*strain_rate[0] + strain_rate[1]*strain_rate[1] + strain_rate[2]*strain_rate[2] +
                0.5 * (strain_rate[3]*strain_rate[3] + strain_rate[4]*strain_rate[4] + strain_rate[5]*strain_rate[5])
            );
            let sr_mag = j2_edot.sqrt();

            let mat_idx = elem_mat_ids[elem_id] as usize;
            let eps_p = elem_strains[elem_id];

            // Compute average pressure from the 4 corner nodes
            let mut p_elem = 0.0;
            for i in 0..4 {
                p_elem += current_sol[dof_mgr_init.pressure_dof(elem.nodes[i]).unwrap()];
            }
            p_elem /= 4.0;

            let mu_p = materials[mat_idx].plasticity.softened_viscosity(&strain_rate, p_elem, eps_p);
            let mu_v = materials[mat_idx].viscosity;
            let mu_eff = mu_v.min(mu_p);
            let stress = 2.0 * mu_eff * sr_mag;

            (sr_mag, stress, mu_eff, p_elem)
        }).collect();

        // Extract results from parallel computation
        let mut sr_ii = vec![0.0; mesh.num_elements()];
        let mut stress_ii = vec![0.0; mesh.num_elements()];
        let mut viscosity_eff = vec![0.0; mesh.num_elements()];
        let mut element_pressures_new = vec![0.0; mesh.num_elements()];
        for (elem_id, (sr, stress, visc, p_elem)) in elem_properties.into_iter().enumerate() {
            sr_ii[elem_id] = sr;
            stress_ii[elem_id] = stress;
            viscosity_eff[elem_id] = visc;
            element_pressures_new[elem_id] = p_elem;
        }
        element_pressures = element_pressures_new;

        // 4. Advect Tracers & 5. Update Mesh Nodes
        // CRITICAL: Safety Guard
        // - Ramp-up Phase (Step < 10): Force update even if not fully converged.
        //   The system needs time to adjust to the onset of plasticity. As long as the linear solver works,
        //   we can "fail forward" to find the stable basin.
        // - Production (Step >= 10): Strict Check. If solver fails, HALT.
        if jfnk_stats.converged || step < ramp_steps {
            if step < ramp_steps && !jfnk_stats.converged {
                 println!("    ⚠ Spin-up Warning (Step {}): Forcing update to settle plasticity.", step);
            }

             // Update tracer properties (including plastic strain)
            let grid_viz = SearchGrid::build(&mesh, [10, 10, 10]);
            for i in 0..swarm.num_tracers() {
                let p_tracer = Point3::new(swarm.x[i], swarm.y[i], swarm.z[i]);
                let candidates = grid_viz.get_potential_elements(p_tracer);
                for &elem_id in candidates {
                    let elem = &mesh.connectivity.tet10_elements[elem_id];
                    let mut vertices = [Point3::origin(); 4];
                    for k in 0..4 { vertices[k] = mesh.geometry.nodes[elem.nodes[k]]; }
                    let l = Tet10Basis::cartesian_to_barycentric(&p_tracer, &vertices);
                    if l.iter().all(|&val| val >= -1e-5 && val <= 1.0 + 1e-5) {
                        // Update plastic strain if yielding significantly (Refined Accumulation)
                        // Only accumulate if effective viscosity < 90% of viscous limit
                        let mat_id = elem_mat_ids[elem_id] as usize;
                        if viscosity_eff[elem_id] < materials[mat_id].viscosity * 0.9 {
                            swarm.plastic_strain[i] += sr_ii[elem_id] * dt;
                        }
                        
                        // Update visualization fields
                        swarm.strain_rate_ii[i] = sr_ii[elem_id];
                        swarm.stress_ii[i] = stress_ii[elem_id];
                        swarm.viscosity[i] = viscosity_eff[elem_id];
                        swarm.pressure[i] = element_pressures[elem_id];
                        break;
                    }
                }
            }

            swarm.advect_rk2(&mesh, &grid, &dof_mgr, &current_sol, dt);

            for (node_id, node) in mesh.geometry.nodes.iter_mut().enumerate() {
                node.x += current_sol[dof_mgr.velocity_dof(node_id, 0)] * dt;
                node.y += current_sol[dof_mgr.velocity_dof(node_id, 1)] * dt;
                node.z += current_sol[dof_mgr.velocity_dof(node_id, 2)] * dt;
            }
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
        let should_output = current_time_years >= next_output_time || step == n_steps;

        if should_output {
            println!("⏱  {:.2} MY ({:6} steps) | Newton: {:2} | LinIter: {:5} | ||R||: {:.2e} | Mesh: {}/{} OK",
                time_my,
                step,
                jfnk_stats.newton_iterations,
                jfnk_stats.total_linear_iterations,
                jfnk_stats.residual_norm,
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

            VtkWriter::write_combined_vtu(&viz_mesh, &swarm, &filename).unwrap();
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Simulation Completed in {:?}", start_sim.elapsed());
    println!("  Results saved to {}/", config.output.output_dir);
    println!("═══════════════════════════════════════════════════════════════");
}

