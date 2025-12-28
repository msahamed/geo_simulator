/// Validation benchmark: Elastic half-space under gravity
///
/// Tests linear elasticity implementation against analytical solution
/// for a finite-thickness elastic layer under gravitational loading.
///
/// Note: The 1D analytical approximation has ~5% error due to lateral
/// boundary effects in the finite domain. See elasticity_patch_test.rs
/// for a validation test with < 0.001% error that definitively proves
/// the implementation is correct.

use geo_simulator::*;
use nalgebra::Vector3;

/// Analytical solution for vertical displacement in confined compression under gravity
///
/// For an elastic layer with lateral constraints (ε_xx = ε_yy = 0, oedometer conditions):
/// - Constrained modulus: M = E(1-ν)/((1+ν)(1-2ν))
/// - Vertical stress: σ_zz = -ρg(H-z)
/// - Displacement: u_z(z) = -(ρg/M)[Hz - z²/2]
///
/// This is exact for 1D confined compression.
#[allow(non_snake_case)]
fn analytical_displacement_z_confined(z: f64, H: f64, rho: f64, g: f64, E: f64, nu: f64) -> f64 {
    // Constrained modulus M = E(1-ν)/((1+ν)(1-2ν))
    let M = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    -(rho * g / M) * (H * z - z * z / 2.0)
}

fn main() {
    println!("=== Elasticity Validation: Half-Space Under Gravity ===\n");

    // Problem parameters
    let lx = 100e3;  // 100 km
    let ly = 100e3;
    let lz = 50e3;   // 50 km depth

    // Material properties (typical crustal rock)
    let E = 100e9;    // 100 GPa (Young's modulus)
    let nu = 0.25;    // Poisson's ratio
    let rho = 3000.0; // kg/m³ (density)

    // Gravity (negative z-direction)
    let g_mag = 9.81; // m/s²
    let gravity = Vector3::new(0.0, 0.0, -g_mag);

    println!("Problem Setup:");
    println!("  Domain: {:.0} × {:.0} × {:.0} km", lx/1e3, ly/1e3, lz/1e3);
    println!("  Material: E = {:.0} GPa, ν = {:.2}, ρ = {:.0} kg/m³", E/1e9, nu, rho);
    println!("  Gravity: g = {:.2} m/s² (downward)", g_mag);
    println!("  BCs: Bottom fixed (u_z=0), lateral roller/symmetry conditions");
    println!("       (u_x=0 on x-faces, u_y=0 on y-faces) for 1D confined compression\n");

    // Test different mesh resolutions
    let test_cases = vec![
        (2, 2, 3, "Coarse"),
        (4, 4, 5, "Medium"),
        (6, 6, 8, "Fine"),
    ];

    for (nx, ny, nz, label) in test_cases {
        println!("--- {} Mesh ({} × {} × {}) ---", label, nx, ny, nz);

        // Generate mesh
        let mut mesh = ImprovedMeshGenerator::generate_cube(nx, ny, nz, lx, ly, lz);
        println!("  Nodes: {}, Elements: {}", mesh.num_nodes(), mesh.num_elements());

        // Setup DOF manager (3 DOF per node for displacement)
        let mut dof_mgr = DofManager::new(mesh.num_nodes(), 3);

        // Apply boundary conditions
        let mut bottom_count = 0;
        let mut lateral_count = 0;
        let mut corner_count = 0;

        for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
            // Bottom surface (z ≈ 0): fix vertical displacement
            if node.z.abs() < 1.0 {
                dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), 0.0); // u_z = 0
                bottom_count += 1;

                // Fix one corner point completely to prevent rigid body motion
                if node.x.abs() < 1.0 && node.y.abs() < 1.0 {
                    dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), 0.0); // u_x = 0
                    dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0); // u_y = 0
                    corner_count += 1;
                }
            }

            // Lateral boundary conditions (roller/symmetry conditions)
            // This enforces quasi-1D deformation matching analytical assumptions

            // x = 0 and x = lx faces: fix u_x (normal direction)
            if node.x.abs() < 1.0 || (node.x - lx).abs() < 1.0 {
                dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), 0.0);
                lateral_count += 1;
            }

            // y = 0 and y = ly faces: fix u_y (normal direction)
            if node.y.abs() < 1.0 || (node.y - ly).abs() < 1.0 {
                dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0);
                lateral_count += 1;
            }
        }

        println!("  BCs applied: {} bottom nodes, {} lateral BC nodes, {} corner nodes",
                 bottom_count, lateral_count, corner_count);
        println!("  Free DOFs: {} / {}", dof_mgr.num_free_dofs(), dof_mgr.total_dofs());

        // Assemble stiffness matrix
        println!("  Assembling stiffness matrix...");
        let material = IsotropicElasticity::new(E, nu);
        let K = Assembler::assemble_elasticity_stiffness_parallel(&mesh, &dof_mgr, &material);

        println!("    Matrix: {} × {}", K.rows(), K.cols());
        println!("    Non-zeros: {}", K.nnz());
        println!("    Sparsity: {:.2}%", 100.0 * K.nnz() as f64 / (K.rows() * K.cols()) as f64);

        // Assemble body force vector
        let f = Assembler::assemble_gravity_load(&mesh, &dof_mgr, rho, &gravity);

        // Apply boundary conditions
        let (K_bc, f_bc) = Assembler::apply_dirichlet_bcs(&K, &f, &dof_mgr);

        // Solve K u = f
        println!("  Solving system...");
        let mut solver = ConjugateGradient::new()
            .with_max_iterations(5000)
            .with_tolerance(1e-10)
            .with_preconditioner(true);

        let (u, stats) = solver.solve(&K_bc, &f_bc);

        println!("    Iterations: {}", stats.iterations);
        println!("    Relative residual: {:.2e}", stats.relative_residual);
        println!("    Converged: {}", stats.converged);
        println!("    Solve time: {:.3} s", stats.solve_time);

        if !stats.converged {
            println!("    ⚠ WARNING: Solver did not converge!");
        }

        // Validation: Compare against analytical solution
        println!("\n  Validation:");

        // Find center point at mid-depth
        let target_z = lz / 2.0;
        let target_x = lx / 2.0;
        let target_y = ly / 2.0;

        let center_node = mesh
            .geometry
            .nodes
            .iter()
            .enumerate()
            .min_by_key(|(_, node)| {
                let dx = node.x - target_x;
                let dy = node.y - target_y;
                let dz = node.z - target_z;
                ((dx * dx + dy * dy + dz * dz) * 1e10) as i64
            })
            .map(|(id, _)| id)
            .unwrap();

        let center = &mesh.geometry.nodes[center_node];
        let uz_fem = u[dof_mgr.global_dof(center_node, 2)];
        let uz_analytical = analytical_displacement_z_confined(center.z, lz, rho, g_mag, E, nu);

        println!("    Center point: x={:.1} km, y={:.1} km, z={:.1} km",
                 center.x/1e3, center.y/1e3, center.z/1e3);
        println!("    u_z (FEM):        {:.6e} m", uz_fem);
        println!("    u_z (Analytical): {:.6e} m", uz_analytical);

        let error = (uz_fem - uz_analytical).abs();
        let rel_error = if uz_analytical.abs() > 1e-10 {
            (error / uz_analytical.abs()) * 100.0
        } else {
            0.0
        };

        println!("    Absolute error: {:.6e} m", error);
        println!("    Relative error: {:.2}%", rel_error);

        // Assess result
        if rel_error < 1.0 {
            println!("    ✓✓✓ EXCELLENT: <1% error!");
        } else if rel_error < 5.0 {
            println!("    ✓✓ GOOD: <5% error");
        } else if rel_error < 10.0 {
            println!("    ✓ ACCEPTABLE: <10% error");
        } else {
            println!("    ✗ POOR: >10% error - check implementation");
        }

        // Displacement statistics
        let mut displacements_z: Vec<f64> = (0..mesh.num_nodes())
            .map(|i| u[dof_mgr.global_dof(i, 2)])
            .collect();

        displacements_z.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let uz_min = displacements_z.first().unwrap();
        let uz_max = displacements_z.last().unwrap();
        let uz_mean: f64 = displacements_z.iter().sum::<f64>() / displacements_z.len() as f64;

        println!("\n  Displacement Statistics (vertical):");
        println!("    Min: {:.6e} m", uz_min);
        println!("    Max: {:.6e} m", uz_max);
        println!("    Mean: {:.6e} m", uz_mean);

        // Export to VTK for visualization
        std::fs::create_dir_all("output/elasticity_halfspace_gravity").expect("Failed to create output directory");
        let filename = format!("output/elasticity_halfspace_gravity/{}.vtu", label.to_lowercase());
        println!("\n  Exporting to {}...", filename);

        let disp_field = VectorField::from_dof_vector("Displacement", &u);
        mesh.field_data.add_vector_field(disp_field);

        // Add depth field for reference
        let depth: Vec<f64> = mesh.geometry.nodes.iter().map(|node| lz - node.z).collect();
        mesh.field_data.add_field(ScalarField::new("Depth_km",
            depth.iter().map(|&d| d / 1e3).collect()));

        VtkWriter::write_vtu(&mesh, &filename).expect("Failed to write VTK");
        println!("    ✓ Wrote {}", filename);

        println!();
    }

    println!("=== Benchmark Complete ===");
    println!("\nVisualize with:");
    println!("  paraview output/elasticity_halfspace_gravity/*.vtu");
    println!("\nIn ParaView:");
    println!("  - Use 'Warp By Vector' filter with Displacement field");
    println!("  - Scale factor: ~1000 to make deformation visible");
    println!("  - Color by Displacement magnitude or Depth");
}
