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
    ImprovedMeshGenerator, DofManager, Assembler, BiCGSTAB, ConjugateGradient, VectorField,
    VtkWriter, ElastoViscoPlastic, TracerSwarm, SearchGrid, ScalarField,
    jfnk_solve, JFNKConfig, GaussQuadrature, Tet10Basis, StrainDisplacement,
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

    let (lx, ly, lz) = (40_000.0, 10_000.0, 30_000.0);
    let v_extension = 0.01 / (365.25 * 24.0 * 3600.0); // 1 cm/yr in m/s
    
    // Rheologies
    let mu_crust = 1e23;
    let mu_mantle = 1e21; // Lower viscosity for mantle upwelling
    let c0 = 20e6;
    let cmin = 2e6;
    let phi = 30.0_f64.to_radians();
    
    // Gravity and Density
    let g = 9.81;
    let rho_crust = 2700.0;
    let rho_mantle = 3300.0;
    let gravity_vec = Vector3::new(0.0, 0.0, -g);
    
    // CRITICAL: Reduce time step for plasticity stability
    // Set to 1000 steps for 5 Myr total
    let n_steps = 1000;
    let dt = 5_000.0 * (365.25 * 24.0 * 3600.0); // 5,000 years per step -> 5 Myr total

    println!("Problem Dimensions:");
    println!("  Crust: 20 km | Mantle: 10 km");
    println!("  Extension Rate: 1 cm/yr");
    println!("  Time scale: {} years/step", dt / (365.25 * 24.0 * 3600.0));
    println!("  Total Duration: 5 Million Years");

    // ========================================================================
    // 2. Mesh and Tracer Initialization
    // ========================================================================

    println!("\nGenerating Mesh and Tracers...");
    // Resolution restored now that we're using CG (not DirectSolver)
    let res_x = 16;
    let res_y = 4;
    let res_z = 8;
    let mut mesh = ImprovedMeshGenerator::generate_cube(res_x, res_y, res_z, lx, ly, lz);
    let n_elements = mesh.num_elements();
    
    // Initialize Tracers (SoA)
    // 4 tracers per element initially
    let mut swarm = TracerSwarm::with_capacity(n_elements * 10);
    let mut n_tracers = 0;
    
    // Distribute tracers in layers
    let nx = 20; let ny = 10; let nz = 15;
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
                
                // Initial material ID (will be overwritten by perturbation)
                let mat_id = if p.z > 10_000.0 { 0 } else { 1 }; 
                
                swarm.add_tracer(p, mat_id);
                n_tracers += 1;
            }
        }
    }

    // Apply perturbation and localized seeding
    let interface_z = 10000.0;
    for i in 0..swarm.num_tracers() {
        // 1. Interface Perturbation
        let perturbation = 1000.0 * (-( (swarm.x[i] - lx/2.0).powi(2) ) / (2.0 * 5000.0_f64.powi(2))).exp();
        if swarm.z[i] < interface_z + perturbation { swarm.material_id[i] = 1; } // Mantle
        else { swarm.material_id[i] = 0; } // Crust

        // 2. Localized Plastic Strain Seed (The "Patch")
        // Add a small Gaussian patch of initial damage at the domain center
        let dist_sq = (swarm.x[i] - lx/2.0).powi(2) + (swarm.z[i] - 10000.0).powi(2);
        let radius: f64 = 2000.0;
        let seed = 0.1 * (-(dist_sq) / (2.0 * radius.powi(2))).exp();
        if seed > 0.01 {
            swarm.plastic_strain[i] = seed;
        }
    }
    println!("  Mesh: {} elements | Tracers: {}", n_elements, n_tracers);

    // ========================================================================
    // 3. Materials Setup
    // ========================================================================

    let materials = vec![
        // Crust: Stronger background
        ElastoViscoPlastic::new(100e9, 0.25, mu_crust, c0, phi).with_softening(cmin, 10.0, 0.1),
        // Mantle: Weaker
        ElastoViscoPlastic::new(100e9, 0.25, mu_mantle, c0, phi).with_softening(cmin, 10.0, 0.1),
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
    let rho_avg = (rho_crust + rho_mantle) / 2.0;
    for (elem_id, elem) in mesh.connectivity.tet10_elements.iter().enumerate() {
        let mut z_sum = 0.0;
        for &node_id in &elem.nodes { z_sum += mesh.geometry.nodes[node_id].z; }
        let depth = (lz - z_sum / 10.0).max(0.0);
        element_pressures[elem_id] = rho_avg * g * depth;
    }
    
    std::fs::create_dir_all("output/core_complex").ok();
    
    println!("\nStarting Tectonic Evolution...");
    let start_sim = Instant::now();

    for step in 0..=n_steps {
        // 1. Setup BCs for current mesh state
        let mut dof_mgr = DofManager::new(mesh.num_nodes(), 3);
        for &node_id in &left_nodes { dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), -v_extension); }
        for &node_id in &right_nodes { dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), v_extension); }
        for &node_id in &bottom_nodes { dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), 0.0); }
        // Free slip on y-boundaries
        for &node_id in &back_nodes { dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0); }
        for &node_id in &front_nodes { dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0); }
        let n_dofs = dof_mgr.total_dofs();

        // 2. Map Material Properties (M2E)
        let grid = SearchGrid::build(&mesh, [10, 10, 10]);
        let (elem_mat_ids, elem_strains) = swarm.get_element_properties(&mesh, &grid);

        // 3. Setup JFNK (Jacobian-Free Newton-Krylov) solver

        // JFNK config: Conservative settings for viscoplastic systems
        // JFNK config: Conservative settings for viscoplastic systems
        let mut jfnk_config = JFNKConfig::conservative();
        // Show detailed Newton iterations every 0.1 MY (20 steps) to keep console clean
        jfnk_config.verbose = step % 20 == 0; 

        // Linear solver for each Newton iteration
        let mut linear_solver = ConjugateGradient::new()
            .with_max_iterations(10000)
            .with_tolerance(1e-8)
            .with_abs_tolerance(1e7); // Expert recommended for geodynamic scales

        // residual_fn: Fast parallel closure for JFNK inner iterations
        let residual_fn = |v: &[f64]| {
            Assembler::compute_stokes_residual_parallel(
                &mesh, &dof_mgr, &materials, &elem_mat_ids, v, &element_pressures, &elem_strains,
                &gravity_vec, rho_crust, rho_mantle
            )
        };

        // precond_fn: Standard parallel assembly for JFNK Jacobi preconditioning
        let precond_fn = |v: &[f64]| {
            Assembler::assemble_stokes_vep_multimaterial_parallel(
                &mesh, &dof_mgr, &materials, &elem_mat_ids, v, &element_pressures, &elem_strains
            )
        };

        // Solve nonlinear system with optimized JFNK
        let (v_new, jfnk_stats) = jfnk_solve(
            residual_fn,
            precond_fn,
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

        // 6. Update Element Pressures (Lithostatic Approximation)
        for (elem_id, &mat_id) in elem_mat_ids.iter().enumerate() {
            let mut z_sum = 0.0;
            let elem_nodes = &mesh.connectivity.tet10_elements[elem_id].nodes;
            for &node_id in elem_nodes { z_sum += mesh.geometry.nodes[node_id].z; }
            let z_center = z_sum / 10.0;
            let depth = (lz - z_center).max(0.0);
            let rho = if mat_id == 0 { rho_crust } else { rho_mantle };
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

