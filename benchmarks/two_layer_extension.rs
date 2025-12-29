/// Benchmark: 2-Layer Extension with Tracer Advection
///
/// **Goal:** Validate tracer advection and marker-to-element (M2E) mapping
/// for two-material systems under small deformation.
///
/// **Setup:**
/// - Domain: 40 km × 30 km × 30 km
/// - Two materials: Weak upper layer (0-15 km) + Strong lower layer (15-30 km)
/// - Extension: 1 cm/yr for 10,000 years (10 steps × 1000 years)
/// - Total extension: 100 m (0.25% strain - minimal distortion)
/// - 3000 tracers tracking material properties
///
/// **Validation Criteria:**
/// 1. JFNK converges in <30 iterations per step
/// 2. Tracers remain in correct material layers
/// 3. M2E mapping gives smooth material distribution
/// 4. Velocity field matches kinematic BCs (linear in x)
/// 5. No mesh quality degradation (all Jacobians > 0)
///
/// **Success:** Clean convergence through all 10 steps with correct material tracking

use geo_simulator::{
    ImprovedMeshGenerator, DofManager, Assembler, BiCGSTAB, VectorField,
    VtkWriter, ElastoViscoPlastic, PlasticityState, ScalarField, TracerSwarm, SearchGrid,
    jfnk_solve, JFNKConfig, Tet10Basis,
};
use nalgebra::Point3;
use std::time::Instant;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Validation: 2-Layer Extension with Tracer Advection");
    println!("═══════════════════════════════════════════════════════════════\n");

    // ========================================================================
    // 1. Physical Parameters
    // ========================================================================

    let lx = 40e3;  // 40 km
    let ly = 30e3;  // 30 km
    let lz = 30e3;  // 30 km
    let interface_z = 15e3;  // Interface at 15 km depth

    let v_extension = 1e-2 / (365.25 * 24.0 * 3600.0);  // 1 cm/yr in m/s
    let dt = 1000.0 * 365.25 * 24.0 * 3600.0;  // 1000 years in seconds
    let n_steps = 10;

    // Material properties
    let mu_upper = 1e21;  // Weak upper layer (Pa·s)
    let mu_lower = 1e23;  // Strong lower layer (Pa·s) - 100× stronger
    let rho_upper = 2700.0;  // Upper density (kg/m³)
    let rho_lower = 3300.0;  // Lower density (kg/m³)
    let c0 = 20e6;     // Cohesion (Pa)
    let phi = 30.0_f64.to_radians();  // Friction angle
    let cmin = 5e6;    // Softened cohesion (Pa)

    let g = 9.81;
    let gravity_vec = nalgebra::Vector3::new(0.0, 0.0, -g);

    println!("Physical Parameters:");
    println!("  Domain:           {:.0} km × {:.0} km × {:.0} km", lx/1e3, ly/1e3, lz/1e3);
    println!("  Interface depth:  {:.0} km", interface_z/1e3);
    println!("  Extension rate:   {:.2e} m/s ({:.1} cm/yr)", v_extension, v_extension * 365.25 * 24.0 * 3600.0 * 100.0);
    println!("  Timestep:         {:.0} years", dt / (365.25 * 24.0 * 3600.0));
    println!("  Total steps:      {}", n_steps);
    println!("  Total extension:  {:.1} m ({:.3}% strain)", v_extension * dt * n_steps as f64, 100.0 * v_extension * dt * n_steps as f64 / lx);
    println!("\nMaterial Properties:");
    println!("  Upper viscosity:  {:.1e} Pa·s", mu_upper);
    println!("  Lower viscosity:  {:.1e} Pa·s", mu_lower);
    println!("  Viscosity ratio:  {:.0}×", mu_lower / mu_upper);
    println!("  Density contrast: {:.0} kg/m³", rho_lower - rho_upper);
    println!();

    // ========================================================================
    // 2. Mesh Generation
    // ========================================================================

    println!("Generating mesh (6×6×6 for fast testing)...");
    let mut mesh = ImprovedMeshGenerator::generate_cube(6, 6, 6, lx, ly, lz);
    let n_elements = mesh.connectivity.tet10_elements.len();
    println!("  Elements: {}", n_elements);

    // Initialize plasticity state
    mesh.plasticity_state = Some(PlasticityState::new(n_elements));

    // ========================================================================
    // 3. Tracer Initialization
    // ========================================================================

    println!("\nInitializing tracers...");
    let n_tracers = 3000;
    let mut swarm = TracerSwarm::with_capacity(n_tracers);

    // Distribute tracers uniformly in 3D
    let mut rng_seed = 12345u64;
    for _ in 0..n_tracers {
        // Simple LCG random number generator
        rng_seed = (1103515245u64.wrapping_mul(rng_seed).wrapping_add(12345)) % (1u64 << 31);
        let rand_x = (rng_seed as f64) / ((1u64 << 31) as f64);

        rng_seed = (1103515245u64.wrapping_mul(rng_seed).wrapping_add(12345)) % (1u64 << 31);
        let rand_y = (rng_seed as f64) / ((1u64 << 31) as f64);

        rng_seed = (1103515245u64.wrapping_mul(rng_seed).wrapping_add(12345)) % (1u64 << 31);
        let rand_z = (rng_seed as f64) / ((1u64 << 31) as f64);

        let x = rand_x * lx;
        let y = rand_y * ly;
        let z = rand_z * lz;

        // Assign material ID based on z-coordinate
        let mat_id = if z < interface_z { 0 } else { 1 };

        swarm.add_tracer(Point3::new(x, y, z), mat_id);
    }

    let n_upper = swarm.material_id.iter().filter(|&&id| id == 0).count();
    let n_lower = n_tracers - n_upper;
    println!("  Tracers: {} total ({} upper, {} lower)", n_tracers, n_upper, n_lower);

    // ========================================================================
    // 4. Material Models
    // ========================================================================

    let materials = vec![
        // Upper: Weak with plasticity
        ElastoViscoPlastic::new(100e9, 0.25, mu_upper, c0, phi).with_softening(cmin, 10.0, 0.1),
        // Lower: Strong with plasticity
        ElastoViscoPlastic::new(100e9, 0.25, mu_lower, c0, phi).with_softening(cmin, 10.0, 0.1),
    ];

    // ========================================================================
    // 5. Boundary Conditions
    // ========================================================================

    let tol = 1e-6;
    let mut left_nodes = Vec::new();
    let mut right_nodes = Vec::new();
    let mut bottom_nodes = Vec::new();
    let mut back_nodes = Vec::new();
    let mut front_nodes = Vec::new();

    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        if node.x.abs() < tol { left_nodes.push(node_id); }
        if (node.x - lx).abs() < tol { right_nodes.push(node_id); }
        if node.z.abs() < tol { bottom_nodes.push(node_id); }
        if node.y.abs() < tol { back_nodes.push(node_id); }
        if (node.y - ly).abs() < tol { front_nodes.push(node_id); }
    }

    println!("\nBoundary nodes:");
    println!("  Left (x=0):     {} nodes (v_x = -{:.2e} m/s)", left_nodes.len(), v_extension);
    println!("  Right (x=Lx):   {} nodes (v_x = +{:.2e} m/s)", right_nodes.len(), v_extension);
    println!("  Bottom (z=0):   {} nodes (fixed)", bottom_nodes.len());
    println!("  Front/Back:     {} nodes (free slip)", back_nodes.len() + front_nodes.len());

    // ========================================================================
    // 6. Time Integration
    // ========================================================================

    let n_dofs = mesh.num_nodes() * 3;
    let mut velocity = vec![0.0; n_dofs];

    // Initialize velocity with linear profile matching BCs
    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        let x_normalized = node.x / lx;  // 0 to 1
        velocity[node_id * 3 + 0] = v_extension * (2.0 * x_normalized - 1.0); // -v to +v
    }

    let mut element_pressures = vec![0.0; n_elements];

    // Initial lithostatic pressure
    let rho_avg = (rho_upper + rho_lower) / 2.0;
    for (elem_id, elem) in mesh.connectivity.tet10_elements.iter().enumerate() {
        let mut z_sum = 0.0;
        for &node_id in &elem.nodes { z_sum += mesh.geometry.nodes[node_id].z; }
        let depth = (lz - z_sum / 10.0).max(0.0);
        element_pressures[elem_id] = rho_avg * g * depth;
    }

    std::fs::create_dir_all("output/two_layer_extension").ok();

    println!("\nStarting Time Integration ({} steps)...", n_steps);
    let start_sim = Instant::now();

    for step in 0..=n_steps {
        // 1. Setup BCs
        let mut dof_mgr = DofManager::new(mesh.num_nodes(), 3);
        for &node_id in &left_nodes { dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), -v_extension); }
        for &node_id in &right_nodes { dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), v_extension); }
        for &node_id in &bottom_nodes { dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), 0.0); }
        for &node_id in &back_nodes { dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0); }
        for &node_id in &front_nodes { dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0); }

        // 2. Map tracers to elements (M2E)
        let grid = SearchGrid::build(&mesh, [10, 10, 10]);
        let (elem_mat_ids, elem_strains) = swarm.get_element_properties(&mesh, &grid);

        // 3. JFNK solve
        let mut jfnk_config = JFNKConfig::conservative();
        jfnk_config.verbose = true;

        let mut linear_solver = BiCGSTAB::new()
            .with_max_iterations(10000)
            .with_tolerance(1e-8)
            .with_abs_tolerance(1e7);

        let assembler = |v: &[f64]| {
            let k_matrix = Assembler::assemble_stokes_vep_multimaterial_parallel(
                &mesh, &dof_mgr, &materials, &elem_mat_ids, v, &element_pressures, &elem_strains
            );

            let mut f = vec![0.0; n_dofs];
            for (elem_id, &mat_id) in elem_mat_ids.iter().enumerate() {
                let rho = if mat_id == 0 { rho_upper } else { rho_lower };
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

            (k_matrix, f)
        };

        let (v_new, jfnk_stats) = jfnk_solve(
            assembler,
            &mut linear_solver,
            &mut velocity,
            &dof_mgr,
            &jfnk_config,
        );

        if !jfnk_stats.converged {
            println!("  ⚠️  Step {}: JFNK did not converge! Iters: {}, Residual: {:.3e}",
                     step, jfnk_stats.newton_iterations, jfnk_stats.residual_norm);
        } else {
            println!("Step {:3} | {:.3} kyr | Newton: {:2} | Lin: {:5} | ||R||: {:.2e}",
                     step,
                     step as f64 * dt / (1000.0 * 365.25 * 24.0 * 3600.0),
                     jfnk_stats.newton_iterations,
                     jfnk_stats.total_linear_iterations,
                     jfnk_stats.residual_norm);
        }

        velocity = v_new;

        // 4. Update tracers
        if step < n_steps {
            // Advect tracers (Euler for simplicity)
            for i in 0..swarm.num_tracers() {
                let p_tracer = Point3::new(swarm.x[i], swarm.y[i], swarm.z[i]);
                let candidates = grid.get_potential_elements(p_tracer);

                for &elem_id in candidates {
                    let elem = &mesh.connectivity.tet10_elements[elem_id];
                    let mut vertices = [Point3::origin(); 4];
                    for k in 0..4 { vertices[k] = mesh.geometry.nodes[elem.nodes[k]]; }

                    let l = Tet10Basis::cartesian_to_barycentric(&p_tracer, &vertices);
                    if l.iter().all(|&val| val >= -1e-5 && val <= 1.0 + 1e-5) {
                        // Get nodal velocities for this element
                        let mut vx_nodes = [0.0; 10];
                        let mut vy_nodes = [0.0; 10];
                        let mut vz_nodes = [0.0; 10];
                        for j in 0..10 {
                            let node_id = elem.nodes[j];
                            vx_nodes[j] = velocity[node_id * 3 + 0];
                            vy_nodes[j] = velocity[node_id * 3 + 1];
                            vz_nodes[j] = velocity[node_id * 3 + 2];
                        }

                        // Interpolate velocity at tracer position
                        let vx = Tet10Basis::evaluate_at_point(&l, &vx_nodes);
                        let vy = Tet10Basis::evaluate_at_point(&l, &vy_nodes);
                        let vz = Tet10Basis::evaluate_at_point(&l, &vz_nodes);

                        // Advect
                        swarm.x[i] += vx * dt;
                        swarm.y[i] += vy * dt;
                        swarm.z[i] += vz * dt;
                        break;
                    }
                }
            }

            // Update mesh node positions (Lagrangian)
            for node_id in 0..mesh.num_nodes() {
                mesh.geometry.nodes[node_id].x += velocity[node_id * 3 + 0] * dt;
                mesh.geometry.nodes[node_id].y += velocity[node_id * 3 + 1] * dt;
                mesh.geometry.nodes[node_id].z += velocity[node_id * 3 + 2] * dt;
            }
        }

        // 5. Export every 2 steps
        if step % 2 == 0 {
            let mut viz_mesh = mesh.clone();
            viz_mesh.field_data.add_vector_field(VectorField::from_dof_vector("Velocity", &velocity));
            viz_mesh.cell_data.add_field(ScalarField::new("MaterialID", elem_mat_ids.iter().map(|&x| x as f64).collect()));
            viz_mesh.cell_data.add_field(ScalarField::new("PlasticStrain", elem_strains.to_vec()));

            let filename = format!("output/two_layer_extension/step_{:04}.vtu", step);
            VtkWriter::write_combined_vtu(&viz_mesh, &swarm, &filename).unwrap();
        }
    }

    let elapsed = start_sim.elapsed().as_secs_f64();
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Benchmark Complete!");
    println!("  Total time: {:.1} seconds ({:.2} s/step)", elapsed, elapsed / (n_steps as f64));
    println!("═══════════════════════════════════════════════════════════════");
}
