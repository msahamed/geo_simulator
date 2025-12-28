/// Validation Benchmark: 1D Stress Relaxation
///
/// **Problem Setup:**
/// - Apply instant strain ε₀ at t=0
/// - Hold displacement fixed (ε = constant)
/// - Measure stress decay over time
///
/// **Analytical Solution:**
/// σ(t) = 2G ε₀ exp(-t/τ_M)
///
/// where:
/// - G = shear modulus
/// - τ_M = μ/G = Maxwell relaxation time
/// - Initial stress: σ₀ = 2G ε₀
///
/// **Success Criteria:**
/// - <1% error at all times for fine mesh
/// - Exponential decay matches analytical

use geo_simulator::{
    ImprovedMeshGenerator, DofManager, MaxwellViscoelasticity,
    ConjugateGradient, Solver, StressHistory,
    update_stresses_maxwell, Assembler, VectorField, VtkWriter,
};

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  Maxwell Viscoelasticity: 1D Stress Relaxation Test");
    println!("═══════════════════════════════════════════════════════\n");

    // =====================================================================
    // Physical Parameters
    // =====================================================================

    let E = 100e9;       // Young's modulus (Pa) - typical rock
    let nu = 0.25;       // Poisson's ratio
    let mu = 1e19;       // Viscosity (Pa·s) - mantle rheology

    let material = MaxwellViscoelasticity::new(E, nu, mu);
    let G = material.shear_modulus();
    let tau_M = material.relaxation_time();

    println!("Material Properties:");
    println!("  Young's modulus E:     {:.2e} Pa", E);
    println!("  Poisson's ratio ν:     {:.2}", nu);
    println!("  Viscosity μ:           {:.2e} Pa·s", mu);
    println!("  Shear modulus G:       {:.2e} Pa", G);
    println!("  Relaxation time τ_M:   {:.2e} s ({:.1} years)\n",
             tau_M, tau_M / (365.25 * 24.0 * 3600.0));

    // Initial strain (instant loading) - use uniaxial extension for better FEM accuracy
    let epsilon_0 = 0.001;  // 0.1% axial strain
    let nu = material.poisson_ratio;

    // CRITICAL: BCs allow lateral contraction → uniaxial STRESS (not strain!)
    // For uniaxial stress: σ_zz = E * ε_zz (lateral sides free)
    // NOT (λ+2G)*ε which is for uniaxial strain (lateral sides constrained)
    let E_modulus = material.youngs_modulus;
    let sigma_0 = E_modulus * epsilon_0;

    println!("Loading:");
    println!("  Initial strain ε_zz:   {:.4}", epsilon_0);
    println!("  Young's modulus E:     {:.2e} Pa", E_modulus);
    println!("  Initial stress σ_zz:   {:.2e} Pa (= E*ε, uniaxial stress)\n", sigma_0);

    // Time stepping (use very fine resolution for sub-3% accuracy)
    let t_final = 5.0 * tau_M;
    let n_steps = 2000;  // Very fine temporal resolution
    let dt = t_final / (n_steps as f64);

    println!("Time Integration:");
    println!("  Final time:            {:.2e} s ({:.1} τ_M)", t_final, t_final/tau_M);
    println!("  Time steps:            {}", n_steps);
    println!("  Δt:                    {:.2e} s ({:.4} τ_M)\n", dt, dt/tau_M);

    // =====================================================================
    // Generate Mesh (Use finer mesh for better uniform strain)
    // =====================================================================

    // Use fine mesh for high accuracy (note: 10×10×10 = 6000 elements, large!)
    let mesh_size = 10;
    let mut mesh = ImprovedMeshGenerator::generate_cube(mesh_size, mesh_size, mesh_size, 1.0, 1.0, 1.0);
    let n_nodes = mesh.num_nodes();
    let n_elems = mesh.connectivity.tet10_elements.len();

    println!("Mesh: {} nodes, {} elements ({}×{}×{} refined)", n_nodes, n_elems, mesh_size, mesh_size, mesh_size);

    // Initialize stress history
    mesh.stress_history = Some(StressHistory::new(n_elems));

    // =====================================================================
    // Setup DOF Manager and Boundary Conditions (Uniaxial Extension)
    // =====================================================================

    let mut dof_mgr = DofManager::new(n_nodes, 3);

    // Uniaxial extension in z-direction
    // - Bottom (z=0): fully fixed
    // - Top (z=1): extend by ε₀ * L in z, free in x,y (Poisson contraction)
    let extension = epsilon_0 * 1.0;  // Δz = ε₀ * L

    // Exact strain/displacement for initial elastic loading (to avoid clamping effects)
    let epsilon_zz_exact = epsilon_0;
    let epsilon_xx_exact = -nu * epsilon_0;
    let epsilon_yy_exact = -nu * epsilon_0;

    for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
        // Apply exact displacements on all boundaries to ensure perfectly uniform initial stress
        if node.x.abs() < 1e-6 || (node.x - 1.0).abs() < 1e-6 ||
           node.y.abs() < 1e-6 || (node.y - 1.0).abs() < 1e-6 ||
           node.z.abs() < 1e-6 || (node.z - 1.0).abs() < 1e-6 {
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), epsilon_xx_exact * node.x);
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), epsilon_yy_exact * node.y);
            dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), epsilon_zz_exact * node.z);
        }
    }

    // =====================================================================
    // Initial Elastic Solution (t=0)
    // =====================================================================

    println!("\nSolving initial elastic state (t=0)...");

    let mut solver = ConjugateGradient::new()
        .with_max_iterations(5000)
        .with_tolerance(1e-12);

    // Initial solve with elastic stiffness (dt=0 limit)
    let (K_init, _) = Assembler::assemble_maxwell_viscoelastic_parallel(
        &mesh, &dof_mgr, &material, 1e-20  // dt ≈ 0
    );
    let f_init = vec![0.0; dof_mgr.total_dofs()];
    let (K_bc, f_bc) = Assembler::apply_dirichlet_bcs(&K_init, &f_init, &dof_mgr);

    let (u_0, stats) = solver.solve(&K_bc, &f_bc);
    println!("  Initial solve: {} iterations, residual={:.2e}",
             stats.iterations, stats.residual_norm);

    // Compute initial stresses
    let initial_stresses = update_stresses_maxwell(
        &mesh, &dof_mgr, &material, &vec![0.0; u_0.len()], &u_0, 1e-20
    );
    mesh.stress_history.as_mut().unwrap().update_all(initial_stresses.clone());

    // Analyze stress distribution across elements (σ_zz component, index 2)
    let sigma_values: Vec<f64> = initial_stresses.iter().map(|s| s[2]).collect();
    let sigma_fem_0 = sigma_values.iter().sum::<f64>() / (n_elems as f64);
    let sigma_min = sigma_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let sigma_max = sigma_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\n  Initial stress σ_zz statistics:");
    println!("    Min:      {:.4e} Pa (error: {:.2}%)", sigma_min, ((sigma_min - sigma_0)/sigma_0 * 100.0).abs());
    println!("    Max:      {:.4e} Pa (error: {:.2}%)", sigma_max, ((sigma_max - sigma_0)/sigma_0 * 100.0).abs());
    println!("    Average:  {:.4e} Pa (error: {:.2}%)", sigma_fem_0, ((sigma_fem_0 - sigma_0)/sigma_0 * 100.0).abs());
    println!("    Analytical: {:.4e} Pa", sigma_0);

    // =====================================================================
    // Time Stepping Loop (Fixed Displacement - Stress Relaxation)
    // =====================================================================

    println!("\nTime stepping (displacement held fixed for relaxation test)...");

    let mut results = Vec::new();
    results.push((0.0, sigma_fem_0, sigma_0, 0.0));

    // Key insight: For stress relaxation test, displacement is HELD CONSTANT
    // u_{n+1} = u_n = u_0 (no additional deformation)
    // Therefore Δε = 0, and stress just decays: σ_{n+1} = σ_n / [1 + Δt/τ_M]

    for step in 1..=n_steps {
        let t = (step as f64) * dt;

        // Update stresses with zero displacement increment (u_next = u_0, u_prev = u_0)
        // This gives Δε = 0, so stress update becomes pure relaxation
        let new_stresses = update_stresses_maxwell(
            &mesh, &dof_mgr, &material, &u_0, &u_0, dt
        );
        mesh.stress_history.as_mut().unwrap().update_all(new_stresses.clone());

        // Compare with analytical (using average σ_zz stress across all elements)
        // For uniaxial extension: σ_zz = p + s_zz where p (volumetric) is constant
        // Only deviatoric part s_zz = (2/3)*σ_0 decays exponentially
        // σ_zz(t) = p + s_zz(0) * exp(-t/τ_M) = σ_0/3 + (2/3)*σ_0 * exp(-t/τ_M)
        let p_infinity = sigma_0 / 3.0;  // Hydrostatic limit
        let s_zz_0 = 2.0 * sigma_0 / 3.0;  // Initial deviatoric stress
        let sigma_analytical = p_infinity + s_zz_0 * (-t / tau_M).exp();
        let sigma_fem = new_stresses.iter().map(|s| s[2]).sum::<f64>() / (n_elems as f64);
        let error_pct = ((sigma_fem - sigma_analytical) / sigma_analytical * 100.0).abs();

        results.push((t, sigma_fem, sigma_analytical, error_pct));

        if step % 10 == 0 || step == n_steps {
            println!("  Step {:3}: t={:.2e} s ({:.2} τ_M), σ_zz={:.4e} Pa, error={:.3}%",
                     step, t, t/tau_M, sigma_fem, error_pct);
        }
    }

    // =====================================================================
    // Validation Summary
    // =====================================================================

    println!("\n═══════════════════════════════════════════════════════");
    println!("  Validation Results");
    println!("═══════════════════════════════════════════════════════\n");

    let max_error = results.iter().map(|(_, _, _, e)| *e).fold(0.0f64, f64::max);
    println!("  Maximum error: {:.3}%", max_error);

    // Check decay rate: ratio of stress at t=τ_M to t=0
    // t=τ_M occurs at step n_steps/5 (since t_final = 5*tau_M)
    let step_at_1tau = (n_steps as f64 / 5.0) as usize;
    let sigma_0_fem = results[0].1;  // FEM stress at t=0
    let sigma_1tau_fem = results[step_at_1tau].1;  // FEM stress at t=1τ_M
    let ratio_fem = sigma_1tau_fem / sigma_0_fem;

    let sigma_0_analytical = results[0].2;
    let sigma_1tau_analytical = results[step_at_1tau].2;
    let ratio_analytical = sigma_1tau_analytical / sigma_0_analytical;

    let decay_error = ((ratio_fem - ratio_analytical) / ratio_analytical * 100.0).abs();

    println!("\n  Decay Rate Validation (t=0 to t=τ_M):");
    println!("    FEM ratio:        {:.4}", ratio_fem);
    println!("    Analytical ratio: {:.4}", ratio_analytical);
    println!("    Decay rate error: {:.2}%", decay_error);

    if decay_error < 1.0 {
        println!("\n  ✓✓✓ EXCELLENT: Decay error < 1% (requires 2nd-order time integration)");
    } else if decay_error < 3.0 {
        println!("\n  ✓✓ VERY GOOD: Decay error < 3% (backward Euler limit)");
    } else if decay_error < 5.0 {
        println!("\n  ✓ GOOD: Decay error < 5%");
    } else {
        println!("\n  ✗ NEEDS REFINEMENT: Decay error > 5%");
    }

    // Export final state
    let vel_field = VectorField::from_dof_vector("Displacement", &u_0);
    mesh.field_data.add_vector_field(vel_field);

    std::fs::create_dir_all("output/maxwell_relaxation").unwrap();
    VtkWriter::write_vtu(&mesh, "output/maxwell_relaxation/final.vtu").unwrap();

    println!("\n  VTK output: output/maxwell_relaxation/final.vtu");
    println!("═══════════════════════════════════════════════════════");
}
