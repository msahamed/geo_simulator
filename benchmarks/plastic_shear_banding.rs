/// Benchmark: 3D Plastic Shear Banding
/// 
/// **Goal:** Verify the formation of localized plastic shear bands in a 3D domain
/// with strain softening and a weak seed.
///
/// **Setup:**
/// - Domain: 100m x 100m x 100m cube
/// - BCs: Pure shear (v_x = +V at z=H, v_x = -V at z=0)
/// - Material: Drucker-Prager with cohesion softening
/// - Weak Seed: A small region with 50% reduced cohesion
/// - Multi-step: Perform several iterations, updating accumulated strain
///
/// **Physics:**
/// Non-linear Stokes flow with viscoplastic regularization.
/// Softening leads to strain localization.

use geo_simulator::{
    ImprovedMeshGenerator, Mesh, DofManager, Assembler, BiCGSTAB, VectorField,
    VtkWriter, ElastoViscoPlastic, PlasticityState, ScalarField,
    jfnk_solve, JFNKConfig,
};
use nalgebra::Point3;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Plasticity Validation: 3D Shear Banding Benchmark");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ========================================================================
    // 1. Problem Parameters
    // ========================================================================

    let L = 100.0;      
    let V = 1.0;        // Boundary velocity (top/bottom)
    let mu_visc = 1e23; // Background viscosity
    let C0 = 20e6;      // Initial cohesion (20 MPa)
    let Cmin = 5e6;     // Final cohesion (5 MPa)
    let phi = 20.0_f64.to_radians(); 
    let strain_ref = 0.5; // Reference strain for softening
    
    let n_steps = 10;   // Number of pseudo-time steps
    let dt = 0.1;       // Time increment

    println!("Physical Parameters:");
    println!("  Domain:           {}m x {}m x {}m", L, L, L);
    println!("  BC Velocity:      {} m/s", V);
    println!("  Viscosity:        {:.1e} Pa·s", mu_visc);
    println!("  Cohesion:         {:.1} -> {:.1} MPa", C0/1e6, Cmin/1e6);
    println!("  Friction Angle:   {:.1}°", phi.to_degrees());
    println!();

    // ========================================================================
    // 2. Mesh and Material Setup
    // ========================================================================

    println!("Generating mesh (10x10x10)...");
    let mut mesh = ImprovedMeshGenerator::generate_cube(10, 10, 10, L, L, L);
    let n_elements = mesh.connectivity.tet10_elements.len();
    println!("  Elements: {}", n_elements);

    // Initialize plasticity state
    mesh.plasticity_state = Some(PlasticityState::new(n_elements));

    // Introduce a weak seed in the center
    // Reduce cohesion for elements near center
    let center = Point3::new(L/2.0, L/2.0, L/2.0);
    let seed_radius = 15.0;
    let mut n_seed = 0;
    
    // We'll handle the seed by modifying the material properties locally OR 
    // just starting with some initial plastic strain. 
    // Let's use initial plastic strain in the seed region.
    if let Some(ps) = mesh.plasticity_state.as_mut() {
        for (elem_id, _elem) in mesh.connectivity.tet10_elements.iter().enumerate() {
            // Get element centroid (approximate from base nodes)
            let nodes = &mesh.connectivity.tet10_elements[elem_id].nodes;
            let mut centroid = Point3::origin();
            for i in 0..4 {
                let p = mesh.geometry.nodes[nodes[i]];
                centroid += nalgebra::Vector3::new(p.x, p.y, p.z);
            }
            centroid = Point3::from(centroid.coords / 4.0);
            
            if (centroid - center).norm() < seed_radius {
                ps.set(elem_id, 0.2); // Start with 20% strain (half-softened)
                n_seed += 1;
            }
        }
    }
    println!("  Weak seed initialized in {} elements", n_seed);

    let _material = ElastoViscoPlastic::new(100e9, 0.25, mu_visc, C0, phi)
        .plasticity.with_softening(Cmin, 20.0, strain_ref);
    // Note: The conversion above is slightly messy due to struct nesting, let's fix it:
    let mut evp = ElastoViscoPlastic::new(100e9, 0.25, mu_visc, C0, phi);
    evp.plasticity = evp.plasticity.with_softening(Cmin, 20.0, strain_ref);

    let mut dof_mgr = DofManager::new(mesh.num_nodes(), 3);
    let n_dofs = dof_mgr.total_dofs();

    // BCs: top side v_x = +V, bottom side v_x = -V
    let tol = 1e-6;
    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        if node.z.abs() < tol {
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), -V);
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0);
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), 0.0);
        }
        if (node.z - L).abs() < tol {
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), V);
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0);
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), 0.0);
        }
    }

    // ========================================================================
    // 3. Simulation Loop
    // ========================================================================

    let mut velocity = vec![0.0; n_dofs];
    let element_pressures = vec![0.0; n_elements]; // Assume zero pressure for this basic test

    // Initialize velocity with linear profile to match BCs
    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        let z_normalized = node.z / L;  // 0 to 1
        velocity[node_id * 3 + 0] = V * (2.0 * z_normalized - 1.0); // -V to +V
    }

    println!("\nStarting Pseudo-Time Integration ({} steps)...", n_steps);
    for step in 1..=n_steps {
        println!("  Step {}:", step);

        // Setup JFNK solver
        let mut jfnk_config = JFNKConfig::conservative();
        jfnk_config.verbose = true;  // Show Newton iterations

        // Linear solver for each Newton iteration
        let mut linear_solver = BiCGSTAB::new()
            .with_max_iterations(10000)
            .with_tolerance(1e-8)
            .with_abs_tolerance(1e7);

        // Assembler closure
        let assembler = |v: &[f64]| {
            let K = Assembler::assemble_stokes_vep_parallel(&mesh, &dof_mgr, &evp, v, &element_pressures);
            let f = vec![0.0; n_dofs];
            (K, f)
        };

        // Solve nonlinear system with JFNK
        let (v_new, jfnk_stats) = jfnk_solve(
            assembler,
            &mut linear_solver,
            &mut velocity,
            &dof_mgr,
            &jfnk_config,
        );

        if !jfnk_stats.converged {
            println!("    WARNING: JFNK did not converge! Iters: {}, Residual: {:.3e}",
                     jfnk_stats.newton_iterations, jfnk_stats.residual_norm);
        } else {
            println!("    JFNK converged in {} iterations, residual: {:.3e}",
                     jfnk_stats.newton_iterations, jfnk_stats.residual_norm);
        }

        velocity = v_new;

        // B. Update Accumulated Strain (Pseudo-Evolution)
        // ε_acc += |ε̇| * dt
        if let Some(ps) = mesh.plasticity_state.as_mut() {
            for (elem_id, elem) in mesh.connectivity.tet10_elements.iter().enumerate() {
                // Compute strain rate magnitude for the element
                let mut nodes = [Point3::origin(); 10];
                for (i, &nid) in elem.nodes.iter().enumerate() {
                    nodes[i] = mesh.geometry.nodes[nid];
                }
                
                // Use center strain rate (barycentric [0.25, 0.25, 0.25, 0.25])
                let qp_center = [0.25, 0.25, 0.25, 0.25];
                let B = geo_simulator::StrainDisplacement::compute_b_at_point(&qp_center, &nodes);
                
                let mut v_elem = nalgebra::SMatrix::<f64, 30, 1>::zeros();
                for i in 0..10 {
                    for comp in 0..3 {
                        v_elem[3*i + comp] = velocity[dof_mgr.global_dof(elem.nodes[i], comp)];
                    }
                }
                
                let edot = B * v_elem;
                
                // √J₂(ε̇)
                let j2_edot = 0.5 * (
                    edot[0] * edot[0] + edot[1] * edot[1] + edot[2] * edot[2] +
                    0.5 * (edot[3] * edot[3] + edot[4] * edot[4] + edot[5] * edot[5])
                );
                let strain_mag = j2_edot.sqrt();
                
                ps.add(elem_id, strain_mag * dt);
            }
        }
        
        println!("    Max velocity: {:.3} m/s", velocity.iter().map(|&x| x.abs()).fold(0.0/0.0, f64::max));
    }

    // ========================================================================
    // 4. Export Results
    // ========================================================================

    println!("\nExporting final results...");
    mesh.field_data.add_vector_field(VectorField::from_dof_vector("Velocity", &velocity));
    
    if let Some(ps) = &mesh.plasticity_state {
        mesh.field_data.add_field(ScalarField {
            name: "PlasticStrain".to_string(),
            data: ps.accumulated_strain.clone(),
        });
    }

    std::fs::create_dir_all("output/plastic_shear_banding").ok();
    VtkWriter::write_vtu(&mesh, "output/plastic_shear_banding/solution.vtu").expect("VTK export failed");
    println!("  Saved to: output/plastic_shear_banding/solution.vtu");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Benchmark Complete!");
    println!("═══════════════════════════════════════════════════════════════");
}
