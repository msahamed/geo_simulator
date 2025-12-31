/// Benchmark: 3D Core Complex Formation
///
/// **Goal:** Simulate crustal extension, strain localization, and mantle upwelling.
///
/// **Setup:**
/// - Domain: 100km x 100km x 30km
/// - Layers: 20km Crust (Mat 0), 10km Mantle (Mat 1)
/// - BCs: Extension at 1 cm/yr (v_x = +V at x=L, v_x = -V at x=0)
/// - Physics: Multi-material Visco-Elasto-Plastic (VEP) with tracer tracking.

use geo_simulator::{
    ImprovedMeshGenerator, DofManager, Assembler, BiCGSTAB, VectorField,
    VtkWriter, ElastoViscoPlastic, TracerSwarm, SearchGrid, ScalarField,
    jfnk_solve, JFNKConfig, GaussQuadrature, Tet10Basis, StrainDisplacement,
    HillslopeDiffusion, assess_mesh_quality, smooth_mesh_auto, WinklerFoundation,
};
use nalgebra::{Point3, Vector3};
use std::time::Instant;
use rayon::prelude::*;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Geodynamic Benchmark: 3D Core Complex Formation");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ========================================================================
    // 1. Problem Parameters (SI Units: meters, kg, seconds, Pascals)
    // ========================================================================
    // DES3D-matched configuration

    let (lx, ly, lz) = (100_000.0, 10_000.0, 10_000.0); // 100x10x10 km (DES3D geometry)
    let v_extension = 0.01 / (365.25 * 24.0 * 3600.0); // 1 cm/yr in m/s

    // Rheologies (DES3D target values)
    let mu_crust = 1e24;
    let mu_mantle = 1e24;
    let c0 = 44e6;        // 44 MPa (DES3D cohesion0)
    let cmin = 4e6;       // 4 MPa (DES3D cohesion1)
    let phi = 30.0_f64.to_radians();

    // Gravity and Density
    let g = 9.81;
    let rho_crust = 2700.0;
    let rho_mantle = 3300.0;
    let gravity_vec = Vector3::new(0.0, 0.0, -g);

    // Timestep for 1 Myr simulation
    let n_steps = 2000;  // 1 Myr total (2000 steps × 500 yr)
    let dt = 500.0 * (365.25 * 24.0 * 3600.0); // 500 years per step

    // Mesh quality check interval (check every N steps to reduce cost)
    let quality_check_interval = 10;

    println!("Problem Dimensions (DES3D-matched, viscosity adjusted for unpreconditioned solver):");
    println!("  Domain: 100 km × 10 km × 10 km");
    println!("  Extension Rate: 1 cm/yr");
    println!("  Rheology: EVP (μ = 1e21 Pa·s, c = 44→4 MPa)");
    println!("  Timestep: {} years", dt / (365.25 * 24.0 * 3600.0));
    println!("  Duration: {} kyr ({} steps)", n_steps as f64 * dt / (365.25 * 24.0 * 3600.0) / 1000.0, n_steps);
    println!("  Mesh quality check: Every {} steps", quality_check_interval);

    // ========================================================================
    // 2. Mesh and Tracer Initialization
    // ========================================================================

    println!("\nGenerating Mesh and Tracers...");
    // Higher resolution (e.g., 100x10x10) is better for research, but 40x4x4 is faster for experiments
    let res_x = 40;
    let res_y = 4;
    let res_z = 4;
    let mut mesh = ImprovedMeshGenerator::generate_cube(res_x, res_y, res_z, lx, ly, lz);
    let n_elements = mesh.num_elements();

    // Initialize Tracers (SoA)
    let mut swarm = TracerSwarm::with_capacity(n_elements * 10);
    let mut n_tracers = 0;

    // Distribute tracers (4 tracers per element on average)
    let nx = 100; let ny = 10; let nz = 10;
    let dx = lx / nx as f64;
    let dy = ly / ny as f64;
    let dz = lz / nz as f64;
    
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let p = Point3::new(
                    i as f64 * dx + dx/2.0,
                    j as f64 * dy + dy/2.0,
                    k as f64 * dz + dz/2.0,
                );
                
                // Single-layer model (no crust/mantle distinction for simplicity)
                let mat_id = 0;

                swarm.add_tracer(p, mat_id);
                n_tracers += 1;
            }
        }
    }

    // DES3D-style dipping weak zone
    // Geometry: Plane dipping at 60° from horizontal, rotated 15° from vertical
    let _weakzone_azimuth = 15.0_f64.to_radians(); // Reserved for 3D rotation
    let weakzone_inclination = -60.0_f64.to_radians(); // -60° = dips downward
    let weakzone_halfwidth = 1200.0; // 1.2 km
    let weakzone_depth_min = 0.5 * lz; // 5 km
    let weakzone_depth_max = 1.0 * lz; // 10 km (full depth)
    let weakzone_xcenter = 0.5 * lx; // 50 km
    let weakzone_ycenter = 0.5 * ly; // 5 km
    let weakzone_plstrain = 0.5; // Initial plastic strain

    // Plane normal (for distance calculation)
    // Inclination = -60° means the plane dips 60° from horizontal
    // Normal vector points perpendicular to the plane
    let cos_inc = weakzone_inclination.cos();
    let sin_inc = weakzone_inclination.sin();

    // Normal vector in 3D (simplified: assume dip in x-z plane, ignore azimuth for now)
    let nx = -sin_inc; // Horizontal component
    let ny = 0.0;
    let nz = cos_inc;  // Vertical component

    for i in 0..swarm.num_tracers() {
        let x = swarm.x[i];
        let y = swarm.y[i];
        let z = swarm.z[i];

        // Distance from tracer to plane center
        let dx = x - weakzone_xcenter;
        let dy = y - weakzone_ycenter;
        let dz = z;

        // Signed distance to plane
        let dist_to_plane = (dx * nx + dy * ny + dz * nz).abs();

        // Check if within weak zone bounds
        let in_depth_range = z >= weakzone_depth_min && z <= weakzone_depth_max;
        let in_lateral_range = x >= 0.3 * lx && x <= 0.7 * lx; // 30-70% of domain

        if in_depth_range && in_lateral_range && dist_to_plane < weakzone_halfwidth {
            swarm.plastic_strain[i] = weakzone_plstrain;
        }
    }
    println!("  Mesh: {} elements | Tracers: {}", n_elements, n_tracers);

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

    let n_dofs = mesh.num_nodes() * 3;
    let mut velocity = vec![0.0; n_dofs];

    // CRITICAL: Initialize velocity with non-zero guess based on BCs
    // This prevents μ_plastic → ∞ at step 0 when strain_rate = 0
    // Simple linear interpolation in x-direction: v_x = v_extension * (2*x/lx - 1)
    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        let x_normalized = node.x / lx;  // 0 to 1
        velocity[node_id * 3 + 0] = v_extension * (2.0 * x_normalized - 1.0); // -v to +v
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
    
    std::fs::create_dir_all("output/core_complex").ok();

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

    println!("\nStarting Tectonic Evolution...");
    let start_sim = Instant::now();

    for step in 0..=n_steps {
        // 1. Setup BCs for current mesh state
        let mut dof_mgr = DofManager::new(mesh.num_nodes(), 3);

        // Extension BCs
        for &node_id in &left_nodes { dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), -v_extension); }
        for &node_id in &right_nodes { dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), v_extension); }

        // Bottom: TEMPORARILY FIXED for solver testing (disable Winkler)
        for &node_id in &bottom_nodes {
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), 0.0);
        }

        // Free slip on y-boundaries
        for &node_id in &back_nodes { dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0); }
        for &node_id in &front_nodes { dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0); }

        // TOP SURFACE: FREE (no BCs) - allows topography development

        let n_dofs = dof_mgr.total_dofs();

        // 2. Map Material Properties (M2E)
        let grid = SearchGrid::build(&mesh, [10, 10, 10]);
        let (elem_mat_ids, elem_strains) = swarm.get_element_properties(&mesh, &grid);

        // 3. Setup JFNK (Jacobian-Free Newton-Krylov) solver

        // JFNK config: Conservative settings for viscoplastic systems
        let mut jfnk_config = JFNKConfig::conservative();
        // Show detailed Newton iterations every 20 steps to keep console clean
        jfnk_config.verbose = step % 20 == 0;

        // Linear solver for each Newton iteration
        // CRITICAL: Must use tight abs_tolerance for geodynamic forces (1e7 Newtons)
        let mut linear_solver = BiCGSTAB::new()
            .with_max_iterations(1000) // Lower for faster steps
            .with_tolerance(1e-3)      // Relaxed relative tol
            .with_abs_tolerance(1e13); // Relaxed absolute tol for geodynamic scales

        // Assembler closure: Returns (K, f) for current velocity
        let assembler = |v: &[f64]| {
            // Assemble K-matrix using current velocity estimate
            let (k_matrix, penalty) = Assembler::assemble_stokes_vep_multimaterial_parallel(
                &mesh, &dof_mgr, &materials, &elem_mat_ids, v, &element_pressures, &elem_strains
            );

            // Compute gravity RHS (constant, doesn't depend on v)
            let mut f = vec![0.0; n_dofs];
            for (elem_id, _mat_id) in elem_mat_ids.iter().enumerate() {
                let rho = rho_crust; // Single-layer model
                let elem_nodes = &mesh.connectivity.tet10_elements[elem_id].nodes;
                let mut nodes = [Point3::origin(); 10];
                for i in 0..10 { nodes[i] = mesh.geometry.nodes[elem_nodes[i]]; }

                let f_elem = geo_simulator::mechanics::BodyForce::gravity_load(&nodes, rho, &gravity_vec);
                for i in 0..10 {
                    for comp in 0..3 {
                        let gi = dof_mgr.global_dof(elem_nodes[i], comp);
                        f[gi] += f_elem[3 * i + comp];
                    }
                }
            }

            // Winkler foundation (DISABLED for testing)
            // let f_winkler = winkler.compute_forces(&mesh, &dof_mgr);
            // for i in 0..f.len() {
            //     f[i] += f_winkler[i];
            // }

            (k_matrix, f, penalty)
        };

        let residual_evaluator = |v: &[f64], p: f64, cd: f64| {
            Assembler::compute_stokes_residual_parallel(
                &mesh, &dof_mgr, &materials, &elem_mat_ids, v, &element_pressures, &elem_strains, &gravity_vec, rho_crust, rho_mantle, p, cd
            )
        };

        // Solve nonlinear system with clean JFNK (no normalization)
        let (v_new, jfnk_stats) = jfnk_solve(
            assembler,
            residual_evaluator,
            &mut linear_solver,
            &mut velocity,
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

        velocity = v_new;
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
                        v_elem[3 * i + comp] = velocity[dof_mgr.global_dof(elem.nodes[i], comp)];
                    }
                }
                strain_rate += b * v_elem;
            }
            strain_rate /= quad.points.len() as f64;

            // Compute invariants
            let j2_edot = 0.5 * (
                strain_rate[0]*strain_rate[0] + strain_rate[1]*strain_rate[1] + strain_rate[2]*strain_rate[2] +
                0.5 * (strain_rate[3]*strain_rate[3] + strain_rate[4]*strain_rate[4] + strain_rate[5]*strain_rate[5])
            );
            let sr_mag = j2_edot.sqrt();

            let mat_idx = elem_mat_ids[elem_id] as usize;
            let eps_p = elem_strains[elem_id];
            let p = element_pressures[elem_id];
            let mu_p = materials[mat_idx].plasticity.softened_viscosity(&strain_rate, p, eps_p);
            let mu_v = materials[mat_idx].viscosity;
            let mu_eff = mu_v.min(mu_p);
            let stress = 2.0 * mu_eff * sr_mag;

            (sr_mag, stress, mu_eff)  // Return tuple
        }).collect();

        // Extract results from parallel computation
        let mut sr_ii = vec![0.0; mesh.num_elements()];
        let mut stress_ii = vec![0.0; mesh.num_elements()];
        let mut viscosity_eff = vec![0.0; mesh.num_elements()];
        for (elem_id, (sr, stress, visc)) in elem_properties.into_iter().enumerate() {
            sr_ii[elem_id] = sr;
            stress_ii[elem_id] = stress;
            viscosity_eff[elem_id] = visc;
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

        // 4. Advect Tracers (RK2)
        swarm.advect_rk2(&mesh, &grid, &dof_mgr, &velocity, dt);

        // 5. Update Mesh Nodes (Lagrangian step)
        // This ensures the mesh deformation is visible in VTK output
        for (node_id, node) in mesh.geometry.nodes.iter_mut().enumerate() {
            node.x += velocity[dof_mgr.global_dof(node_id, 0)] * dt;
            node.y += velocity[dof_mgr.global_dof(node_id, 1)] * dt;
            node.z += velocity[dof_mgr.global_dof(node_id, 2)] * dt;
        }

        // 5a. Mesh Quality Check & Smoothing (Periodic - every N steps for performance)
        if step % quality_check_interval == 0 {
            let quality = assess_mesh_quality(&mesh);
            println!("       Mesh Quality: {}", quality.report());

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

        // 6. Update Element Pressures (Lithostatic Approximation)
        for (elem_id, _mat_id) in elem_mat_ids.iter().enumerate() {
            let mut z_sum = 0.0;
            let elem_nodes = &mesh.connectivity.tet10_elements[elem_id].nodes;
            for &node_id in elem_nodes { z_sum += mesh.geometry.nodes[node_id].z; }
            let z_center = z_sum / 10.0;
            let depth = (lz - z_center).max(0.0);
            let rho = rho_crust; // Single-layer model
            element_pressures[elem_id] = rho * g * depth;
        }

        // 7. Update Internal Strain on Tracers (Simplified: use element strain)
        // In a full implementation, we'd interpolate L_dot back to racers.
        // For now, elements carry the softened state derived from majority.

        let time_my = (step as f64 * dt) / 1e6 / (365.25 * 24.0 * 3600.0);
        println!("Step {:4} | {:.2} MY | Newton: {:2} | Lin: {:5} | ||R||: {:.2e}",
            step,
            time_my,
            jfnk_stats.newton_iterations,
            jfnk_stats.total_linear_iterations,
            jfnk_stats.residual_norm);

        if step % 20 == 0 || step == n_steps {
            let filename = format!("output/core_complex/step_{:04}.vtu", step);
            
            let mut viz_mesh = mesh.clone();
            viz_mesh.field_data.add_vector_field(VectorField::from_dof_vector("Velocity", &velocity));
            
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
    println!("  Results saved to output/core_complex/");
    println!("═══════════════════════════════════════════════════════════════");
}

