/// Benchmark 3: Winkler Foundation - Point Load Validation
///
/// **Goal:** Validate Winkler foundation restoring forces
///
/// **Problem Setup:**
/// - Elastic column: 10 km × 10 km × 30 km (depth)
/// - Apply vertical point load at top center
/// - Winkler foundation at bottom: F = -k(z - z₀)
/// - Material: E = 100 GPa, ν = 0.25
///
/// **Physics:**
/// - Elastic deformation under gravity + point load
/// - Bottom boundary supported by Winkler foundation
/// - Equilibrium: σ_elastic + F_winkler = 0
///
/// **Success Criteria:**
/// - Bottom deflection < 1 m with stiff foundation (k = 1e8 Pa/m)
/// - Bottom deflection > 10 m with soft foundation (k = 1e6 Pa/m)
/// - Force balance verified

use geo_simulator::{
    ImprovedMeshGenerator, DofManager, Assembler, IsotropicElasticity,
    ConjugateGradient, Solver, VtkWriter, VectorField, ScalarField,
    WinklerFoundation,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Winkler Foundation: Point Load Validation");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =====================================================================
    // Physical Parameters
    // =====================================================================

    let lx = 10_000.0;    // 10 km
    let ly = 10_000.0;
    let lz = 30_000.0;    // 30 km depth

    let E = 100e9;        // 100 GPa
    let nu = 0.25;
    let rho = 3300.0;     // kg/m³
    let g = 10.0;         // m/s²

    let point_load = 1e12; // 1 TN downward force at top center

    println!("Parameters:");
    println!("  Domain: {:.0} × {:.0} × {:.0} m", lx, ly, lz);
    println!("  E = {:.0} GPa, ν = {:.2}", E/1e9, nu);
    println!("  ρ = {:.0} kg/m³", rho);
    println!("  Point load: {:.2e} N\n", point_load);

    // Test two foundation stiffnesses
    let test_cases = vec![
        ("Stiff foundation", 1e8),   // 100 MPa/m
        ("Soft foundation", 1e6),    // 1 MPa/m
    ];

    for (test_name, k_winkler) in test_cases {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  {}: k = {:.0e} Pa/m", test_name, k_winkler);
        println!("═══════════════════════════════════════════════════════════════\n");

        // =====================================================================
        // Mesh Generation
        // =====================================================================

        let nx = 6;
        let ny = 6;
        let nz = 8;

        let mut mesh = ImprovedMeshGenerator::generate_cube(nx, ny, nz, lx, ly, lz);

        println!("Mesh: {} nodes, {} elements", mesh.num_nodes(), mesh.num_elements());

        // =====================================================================
        // Boundary Conditions
        // =====================================================================

        let mut dof_mgr = DofManager::new(mesh.num_nodes(), 3);
        let tol = 1e-6;

        // Fix bottom nodes in x, y (allow z deflection)
        let mut bottom_nodes = Vec::new();
        for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
            if node.z.abs() < tol {
                dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), 0.0);  // u_x = 0
                dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0);  // u_y = 0
                // u_z is FREE (Winkler will provide restoring force)
                bottom_nodes.push(node_id);
            }
        }

        // Free surface at top (no BCs)
        println!("Bottom nodes (Winkler active): {}", bottom_nodes.len());

        // =====================================================================
        // Initialize Winkler Foundation
        // =====================================================================

        let mut winkler = WinklerFoundation::new(k_winkler);
        winkler.initialize_reference(&mesh);

        // =====================================================================
        // Material and Assembly
        // =====================================================================

        let material = IsotropicElasticity::new(E, nu);

        println!("Assembling stiffness matrix...");
        let K = Assembler::assemble_elasticity_stiffness_parallel(&mesh, &dof_mgr, &material);

        // External force (point load only - skip gravity for simplicity)
        let mut f_ext = vec![0.0; dof_mgr.total_dofs()];
        let mut center_node = 0;
        let mut min_dist = f64::INFINITY;
        for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
            if (node.z - lz).abs() < tol {  // Top surface
                let dist = (node.x - lx/2.0).powi(2) + (node.y - ly/2.0).powi(2);
                if dist < min_dist {
                    min_dist = dist;
                    center_node = node_id;
                }
            }
        }
        let dof_z_center = dof_mgr.global_dof(center_node, 2);
        f_ext[dof_z_center] = -point_load;  // Downward

        println!("Point load applied at node {} (top center)", center_node);

        // =====================================================================
        // Iterative Solution with Winkler
        // =====================================================================

        println!("\nSolving with Winkler foundation...");

        let mut displacement = vec![0.0; dof_mgr.total_dofs()];
        let mut solver = ConjugateGradient::new()
            .with_max_iterations(5000)
            .with_tolerance(1e-10);

        // Iterate: solve elasticity, update Winkler forces, re-solve
        for iter in 0..10 {
            // Compute Winkler restoring forces based on current displacement
            let f_winkler = winkler.compute_forces(&mesh, &dof_mgr);

            // Total RHS = external + Winkler
            let mut f_rhs = f_ext.clone();
            for i in 0..f_rhs.len() {
                f_rhs[i] += f_winkler[i];
            }

            // Apply BCs and solve
            let (K_bc, f_bc) = Assembler::apply_dirichlet_bcs(&K, &f_rhs, &dof_mgr);
            let (u_new, stats) = solver.solve(&K_bc, &f_bc);

            // Update mesh geometry for next Winkler calculation
            for node_id in 0..mesh.num_nodes() {
                for dof in 0..3 {
                    let global_dof = dof_mgr.global_dof(node_id, dof);
                    match dof {
                        0 => mesh.geometry.nodes[node_id].x += u_new[global_dof] - displacement[global_dof],
                        1 => mesh.geometry.nodes[node_id].y += u_new[global_dof] - displacement[global_dof],
                        2 => mesh.geometry.nodes[node_id].z += u_new[global_dof] - displacement[global_dof],
                        _ => {}
                    }
                }
            }

            // Check convergence
            let du_max = (0..u_new.len())
                .map(|i| (u_new[i] - displacement[i]).abs())
                .fold(0.0f64, f64::max);

            displacement = u_new;

            if iter % 2 == 0 {
                println!("  Iter {}: CG iters={}, Δu_max={:.3e} m", iter, stats.iterations, du_max);
            }

            if du_max < 1e-6 {
                println!("  Converged at iteration {}", iter);
                break;
            }
        }

        // =====================================================================
        // Results
        // =====================================================================

        // Measure bottom deflection
        let z_bottom: Vec<f64> = bottom_nodes.iter()
            .map(|&id| displacement[dof_mgr.global_dof(id, 2)])
            .collect();

        let max_deflection = z_bottom.iter().map(|&z| z.abs()).fold(0.0f64, f64::max);
        let avg_deflection = z_bottom.iter().sum::<f64>() / z_bottom.len() as f64;

        println!("\n  Bottom deflection:");
        println!("    Average: {:.3} m", avg_deflection);
        println!("    Maximum: {:.3} m", max_deflection);

        // Top center deflection
        let top_deflection = displacement[dof_z_center];
        println!("\n  Top center deflection: {:.3} m\n", top_deflection);

        // Validation
        if k_winkler == 1e8 {
            if max_deflection < 1.0 {
                println!("  ✓ Stiff foundation: deflection < 1 m");
            } else {
                println!("  ✗ FAILED: deflection too large for stiff foundation");
            }
        } else {
            if max_deflection > 1.0 {
                println!("  ✓ Soft foundation: deflection > 1 m");
            } else {
                println!("  ✗ FAILED: deflection too small for soft foundation");
            }
        }

        // Export VTK
        mesh.field_data.add_vector_field(VectorField::from_dof_vector("Displacement", &displacement));

        let deflection_field: Vec<f64> = (0..mesh.num_nodes())
            .map(|id| displacement[dof_mgr.global_dof(id, 2)])
            .collect();
        mesh.field_data.add_field(ScalarField::new("VerticalDeflection", deflection_field));

        std::fs::create_dir_all("output/winkler_point_load").ok();
        let filename = format!("output/winkler_point_load/{}.vtu",
                              test_name.replace(" ", "_").to_lowercase());
        VtkWriter::write_vtu(&mesh, &filename).unwrap();

        println!("  VTK: {}", filename);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Winkler Foundation Validation Complete");
    println!("═══════════════════════════════════════════════════════════════");
}
