/// Convergence test for Maxwell stress relaxation
///
/// Tests temporal and spatial convergence to identify error sources

use geo_simulator::{
    ImprovedMeshGenerator, DofManager, MaxwellViscoelasticity,
    ConjugateGradient, Solver, StressHistory,
    update_stresses_maxwell, Assembler,
};

fn test_decay_rate(mesh_size: usize, n_steps: usize) -> (f64, f64) {
    let E = 100e9;
    let nu = 0.25;
    let mu = 1e19;
    let material = MaxwellViscoelasticity::new(E, nu, mu);
    let tau_M = material.relaxation_time();

    let epsilon_0 = 0.001;
    let sigma_0 = E * epsilon_0;

    let mut mesh = ImprovedMeshGenerator::generate_cube(mesh_size, mesh_size, mesh_size, 1.0, 1.0, 1.0);
    let n_nodes = mesh.num_nodes();
    let n_elems = mesh.connectivity.tet10_elements.len();

    mesh.stress_history = Some(StressHistory::new(n_elems));

    let mut dof_mgr = DofManager::new(n_nodes, 3);
    let extension = epsilon_0 * 1.0;

    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        if node.z.abs() < 1e-6 {
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), 0.0);
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0);
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), 0.0);
        }
        if (node.z - 1.0).abs() < 1e-6 {
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), extension);
        }
    }

    let mut solver = ConjugateGradient::new()
        .with_max_iterations(1000)
        .with_tolerance(1e-10);

    // Initial elastic solution
    let (K_init, _) = Assembler::assemble_maxwell_viscoelastic_parallel(&mesh, &dof_mgr, &material, 1e-20);
    let f_init = vec![0.0; dof_mgr.total_dofs()];
    let (K_bc, f_bc) = Assembler::apply_dirichlet_bcs(&K_init, &f_init, &dof_mgr);
    let (u_0, _) = solver.solve(&K_bc, &f_bc);

    let initial_stresses = update_stresses_maxwell(&mesh, &dof_mgr, &material, &vec![0.0; u_0.len()], &u_0, 1e-20);
    mesh.stress_history.as_mut().unwrap().update_all(initial_stresses.clone());

    let sigma_0_fem = initial_stresses.iter().map(|s| s[2]).sum::<f64>() / (n_elems as f64);

    // Check initial stress accuracy
    // CRITICAL: BCs allow lateral contraction, so this is uniaxial STRESS, not strain!
    // For uniaxial stress: σ_zz = E * ε_zz (NOT (λ+2G)*ε)
    let E = 100e9;
    let sigma_0_analytical = E * 0.001;  // Uniaxial stress formula
    let init_error = ((sigma_0_fem - sigma_0_analytical) / sigma_0_analytical * 100.0).abs();

    if mesh_size == 3 && n_steps == 100 {
        println!("Initial stress check:");
        println!("  FEM:        {:.6e} Pa", sigma_0_fem);
        println!("  Analytical: {:.6e} Pa", sigma_0_analytical);
        println!("  Error:      {:.3}%\n", init_error);
    }

    // Time step to t = tau_M
    let dt = tau_M / (n_steps as f64);

    for _ in 0..n_steps {
        let new_stresses = update_stresses_maxwell(&mesh, &dof_mgr, &material, &u_0, &u_0, dt);
        mesh.stress_history.as_mut().unwrap().update_all(new_stresses);
    }

    let sigma_1tau_fem = mesh.stress_history.as_ref().unwrap().stress.iter().map(|s| s[2]).sum::<f64>() / (n_elems as f64);

    let ratio_fem = sigma_1tau_fem / sigma_0_fem;

    // Analytical: σ(τ_M)/σ(0) = (1/3 + 2/3*exp(-1)) / 1 = 1/3 + 2/3*0.368 = 0.578
    let ratio_analytical = 1.0/3.0 + 2.0/3.0 * (-1.0_f64).exp();

    (ratio_fem, ratio_analytical)
}

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  Maxwell Viscoelasticity: Convergence Test");
    println!("═══════════════════════════════════════════════════════\n");

    println!("Analytical decay ratio at t=τ_M: {:.6}", 1.0/3.0 + 2.0/3.0 * (-1.0_f64).exp());
    println!();

    // Test temporal convergence (fixed mesh 5×5×5)
    println!("Temporal Convergence (5×5×5 mesh):");
    println!("  Steps  |  dt/τ_M  |  Ratio  | Error(%)");
    println!("  -------|----------|---------|----------");

    for &n_steps in &[50, 100, 200, 500, 1000, 2000] {
        let (ratio_fem, ratio_analytical) = test_decay_rate(5, n_steps);
        let error = ((ratio_fem - ratio_analytical) / ratio_analytical * 100.0).abs();
        let dt_over_tau = 1.0 / (n_steps as f64);
        println!("  {:5}  |  {:.5}  | {:.5} | {:6.3}%", n_steps, dt_over_tau, ratio_fem, error);
    }

    println!();

    // Test spatial convergence (fixed dt = 0.001*tau_M, 1000 steps)
    println!("Spatial Convergence (dt = 0.001*τ_M, 1000 steps):");
    println!("  Mesh   | Elements |  Ratio  | Error(%)");
    println!("  -------|----------|---------|----------");

    for &mesh_size in &[3, 4, 5, 6, 7, 8] {
        let (ratio_fem, ratio_analytical) = test_decay_rate(mesh_size, 1000);
        let n_elems = mesh_size * mesh_size * mesh_size * 6;
        let error = ((ratio_fem - ratio_analytical) / ratio_analytical * 100.0).abs();
        println!("  {}×{}×{}  |   {:4}   | {:.5} | {:6.3}%", mesh_size, mesh_size, mesh_size, n_elems, ratio_fem, error);
    }

    println!("\n═══════════════════════════════════════════════════════");
}
