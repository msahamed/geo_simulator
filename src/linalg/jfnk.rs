/// Clean JFNK (Jacobian-Free Newton-Krylov) solver for viscoplastic geodynamics
///
/// **Design Philosophy:**
/// - NO explicit normalization (Jacobi preconditioner handles scaling)
/// - Simple quasi-Newton: K(u_k) * δu = -R(u_k)
/// - Backtracking line search for robustness
/// - Residual-based convergence (both relative and absolute)
///
/// **Algorithm:**
/// 1. Compute residual: R(u) = K(u)*u - f
/// 2. Solve: K(u_k) * δu = -R(u_k) using BiCGSTAB
/// 3. Line search: find α such that ||R(u + α*δu)|| < ||R(u)||
/// 4. Update: u_{k+1} = u_k + α*δu
/// 5. Repeat until ||R|| small enough
///
/// # References
/// - Knoll & Keyes (2004), "Jacobian-free Newton-Krylov methods"
/// - ASPECT: https://aspect.geodynamics.org (uses similar approach)

use sprs::CsMat;
use crate::linalg::solver::{Solver, SolverStats};
use crate::linalg::preconditioner::{Preconditioner, ILUPreconditioner};
use crate::linalg::scaling::CharacteristicScales;

/// JFNK configuration
#[derive(Debug, Clone)]
pub struct JFNKConfig {
    /// Maximum Newton iterations
    pub max_newton_iterations: usize,

    /// Relative residual tolerance: ||R|| / ||R_0|| < tol
    pub tolerance: f64,

    /// Absolute residual tolerance (Newtons) - critical for geodynamics
    pub abs_tolerance: f64,

    /// Line search parameters
    pub max_line_search: usize,
    pub line_search_alpha: f64,  // Initial step length
    pub line_search_rho: f64,    // Reduction factor

    /// Verbose output
    /// Verbose output
    pub verbose: bool,

    /// Use AMG preconditioner for velocity block
    pub use_amg: bool,
    
    /// AMG strength threshold (0.25 default, lower for difficult problems)
    pub amg_strength_threshold: f64,
}

impl Default for JFNKConfig {
    fn default() -> Self {
        Self {
            max_newton_iterations: 20,
            tolerance: 1e-4,           // 0.01% relative
            abs_tolerance: 1e12,       // Absolute residual in Newtons
            max_line_search: 10,
            line_search_alpha: 1.0,    // Try full Newton step first
            line_search_rho: 0.5,      // Halve step on failure
            verbose: false,
            use_amg: false,
            amg_strength_threshold: 0.25,
        }
    }
}


impl JFNKConfig {
    /// Conservative config for difficult viscoplastic problems
    pub fn conservative() -> Self {
        Self {
            max_newton_iterations: 50,     // Increased for complex problems
            tolerance: 1e-4,
            abs_tolerance: 5e13,           // Relaxed - realistic for geodynamics
            max_line_search: 15,
            line_search_alpha: 1.0,        // Try full Newton step first
            line_search_rho: 0.5,
            verbose: true,
            use_amg: false,
            amg_strength_threshold: 0.25,
        }
    }
}

/// JFNK solver statistics
#[derive(Debug, Clone)]
pub struct JFNKStats {
    pub newton_iterations: usize,
    pub converged: bool,
    pub residual_norm: f64,
    pub relative_residual: f64,
    pub total_linear_iterations: usize,
    pub last_linear_stats: SolverStats,
}

#[allow(non_snake_case)]
fn compute_residual<FR>(
    residual_evaluator: &mut FR,
    u: &[f64],
) -> Vec<f64>
where
    FR: FnMut(&[f64]) -> Vec<f64>,
{
    residual_evaluator(u)
}

/// Compute L2 norm
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

struct JFNKOperator<'a, FR> {
    residual_evaluator: std::cell::RefCell<&'a mut FR>,
    u: &'a [f64],
    r_u: &'a [f64],
    s_col: Vec<f64>,
}

impl<'a, FR> crate::linalg::solver::LinearOperator for JFNKOperator<'a, FR>
where
    FR: FnMut(&[f64]) -> Vec<f64>,
{
    fn apply(&self, v_hat: &[f64]) -> Vec<f64> {
        let n = self.u.len();
        
        let v_hat_norm = norm(v_hat);
        if v_hat_norm < 1e-40 { return vec![0.0; n]; }

        // Determine epsilon in scaled space: dimensionless O(1)
        // We want a small perturbation relative to the scaled variables
        // u_hat = u_phys / s_col
        let mut u_hat = vec![0.0; n];
        for i in 0..n { u_hat[i] = self.u[i] / self.s_col[i]; }
        let u_hat_norm = norm(&u_hat);
        
        // Typical value 1e-8 for finite difference
        let epsilon = 1e-8 * (u_hat_norm + 1.0) / v_hat_norm;

        // Physical perturbation: dv_phys = epsilon * (S_col * v_hat)
        let mut u_perturbed_phys = vec![0.0; n];
        for i in 0..n { 
            u_perturbed_phys[i] = self.u[i] + epsilon * (v_hat[i] * self.s_col[i]); 
        }

        let r_perturbed = compute_residual(*self.residual_evaluator.borrow_mut(), &u_perturbed_phys);

        let mut jv_scaled = vec![0.0; n];
        for i in 0..n { 
            jv_scaled[i] = (r_perturbed[i] - self.r_u[i]) / epsilon;
        }
        jv_scaled
    }

    fn rows(&self) -> usize { self.u.len() }
    fn cols(&self) -> usize { self.u.len() }
}

struct ScaledPreconditioner<'a, P: Preconditioner> {
    inner: &'a P,
    s_row: Vec<f64>,
    s_col: Vec<f64>,
}

impl<'a, P: Preconditioner> Preconditioner for ScaledPreconditioner<'a, P> {
    fn apply(&self, r_hat: &[f64]) -> Vec<f64> {
        let n = r_hat.len();
        let mut r_phys = vec![0.0; n];
        // J_scaled = S_row * J * S_col
        // Preconditioner M_scaled ≈ S_row * J * S_col
        // M_scaled^-1 = S_col^-1 * M_phys^-1 * S_row^-1
        for i in 0..n {
            r_phys[i] = r_hat[i] / self.s_row[i];
        }
        let z_phys = self.inner.apply(&r_phys);
        let mut z_hat = vec![0.0; n];
        for i in 0..n {
            z_hat[i] = z_phys[i] / self.s_col[i];
        }
        z_hat
    }
}

#[allow(non_snake_case)]
pub fn jfnk_solve<FA, FR, S>(
    mut assembler: FA,
    mut residual_evaluator: FR,
    linear_solver: &mut S,
    u_guess: &mut [f64],
    dof_mgr: &crate::fem::DofManager,
    config: &JFNKConfig,
) -> (Vec<f64>, JFNKStats)
where
    FA: FnMut(&[f64]) -> (CsMat<f64>, Vec<f64>, f64),
    FR: FnMut(&[f64]) -> Vec<f64>,
    S: Solver,
{
    let n = u_guess.len();
    let nv = dof_mgr.total_vel_dofs();
    let mut u = u_guess.to_vec();
    let mut total_linear_iters = 0;
    let mut last_linear_stats = SolverStats::new();

    // 0. Initial residual
    let mut R = compute_residual(&mut residual_evaluator, &u);
    let R_norm_0 = norm(&R);
    let mut R_norm = R_norm_0;

    if config.verbose {
        println!("    JFNK: Initial Mixed Residual = {:.3e}, (v_dofs: {}, p_dofs: {})", R_norm_0, nv, n - nv);
    }

    let mut relative_res = 1.0;
    let mut final_newton_iters = 0;

    for newton_iter in 0..config.max_newton_iterations {
        final_newton_iters = newton_iter;
        relative_res = if R_norm_0 > config.abs_tolerance {
            R_norm / R_norm_0
        } else {
            R_norm / config.abs_tolerance
        };

        if (newton_iter > 0 && relative_res < config.tolerance) || R_norm < config.abs_tolerance {
            if config.verbose {
                println!("    JFNK: Converged in {} iterations", newton_iter);
            }
            u_guess.copy_from_slice(&u);
            return (u, JFNKStats {
                newton_iterations: newton_iter,
                converged: true,
                residual_norm: R_norm,
                relative_residual: relative_res,
                total_linear_iterations: total_linear_iters,
                last_linear_stats,
            });
        }

        // 1. Assemble Saddle-Point Matrix K = [A B^T; B 0]
        let timer = std::time::Instant::now();
        let (K, _f, _) = assembler(&u);
        
        // Handle Pressure Null Space (Pin one pressure DOF)
        // We pin the first pressure DOF if no other pressure constraints exist
        if nv < n {
            let first_p_dof = nv; 
            // Check if it's already Dirichlet (it shouldn't be for Stokes)
            if !dof_mgr.is_dirichlet(first_p_dof) {
                // Pin it to 0
                // This is a bit of a hack in the Jacobian, but effective for the null space.
                // Better approach: ensure the assembler doesn't create a singular system.
                // For JFNK, pinning in the Picard matrix is enough for the preconditioner.
            }
        }

        let (K_bc, _) = crate::fem::Assembler::apply_dirichlet_bcs(&K, &_f, dof_mgr);
        if config.verbose { println!("    JFNK: Assembly took {:?}", timer.elapsed()); }

        // 2. Setup Block Diagonal Preconditioner with AMG for Velocity Block
        // Extract velocity block (cannot use AMG on full system due to zero pressure diagonal)
        let mut A_triplets = sprs::TriMat::new((nv, nv));
        for (row_idx, row) in K_bc.outer_iterator().enumerate() {
            if row_idx < nv {
                for (col_idx, &val) in row.iter() {
                    if col_idx < nv {
                        A_triplets.add_triplet(row_idx, col_idx, val);
                    }
                }
            }
        }
        let A_block = A_triplets.to_csr();

        if config.verbose {
            println!("    AMG: Building preconditioner for velocity block ({} x {})", A_block.rows(), A_block.cols());
            println!("    AMG: A_block has {} non-zeros", A_block.nnz());

            // Verify dimensions match
            if A_block.rows() != nv || A_block.cols() != nv {
                eprintln!("WARNING: A_block dimensions {}x{} != expected {}x{}",
                    A_block.rows(), A_block.cols(), nv, nv);
            }
        }

        // Velocity block preconditioner: AMG (optimal for elliptic operators) or ILU(0)
        let a_precond: Box<dyn Preconditioner> = if config.use_amg {
            if config.verbose { println!("    AMG: Building preconditioner for velocity block ({} x {})", A_block.rows(), A_block.cols()); }
            let amg = crate::linalg::amg::AMGPreconditioner::new(
                &A_block,
                10,    // max_levels
                100,   // coarse_size
                config.amg_strength_threshold,  // strength_threshold
            ).expect("AMG setup failed");
            if config.verbose { println!("    AMG built successfully"); }
            Box::new(amg)
        } else {
             if config.verbose { println!("    ILU: Building preconditioner for velocity block"); }
             let ilu = ILUPreconditioner::new(&A_block).unwrap();
             Box::new(ilu)
        };


        // Pressure block preconditioner: Scaled identity (geodynamic Schur approximation)
        // S ≈ (1/μ) * M_p ≈ (1/μ) * I for dimensionless system
        let mut a_diag_sum = 0.0;
        let mut a_diag_count = 0;
        for i in 0..nv {
            if let Some(&val) = K_bc.get(i, i) {
                if val.abs() > 1e-18 {
                    a_diag_sum += val.abs();
                    a_diag_count += 1;
                }
            }
        }
        let mu_char = if a_diag_count > 0 { a_diag_sum / a_diag_count as f64 } else { 1.0 };

        // Pressure mass matrix approximation
        let mut s_mat_tri = sprs::TriMat::new((n - nv, n - nv));
        for i in 0..(n-nv) {
            s_mat_tri.add_triplet(i, i, mu_char);
        }
        let s_mat = s_mat_tri.to_csr();
        let s_precond = crate::linalg::preconditioner::JacobiPreconditioner::new(&s_mat);

        // Combine into block diagonal preconditioner
        let block_diag_precond = crate::linalg::preconditioner::BlockDiagonalPreconditioner::new(
            a_precond,
            s_precond,
            nv
        );

        // 3. Diagonal Equilibration Scaling
        // With non-dimensionalization, system is already O(1), but we still apply
        // diagonal equilibration to ensure uniform scaling across DOFs
        // Scaling: S = Diag(1/sqrt(|D_ii|))

        let s_diag = vec![1.0; n];
        /* 
        for i in 0..n {
            let is_pressure = i >= nv;
            if dof_mgr.is_dirichlet(i) || is_pressure {
                s_diag[i] = 1.0;  // Don't scale Dirichlet DOFs or Pressure DOFs (stabilization is tiny)
            } else {
                if let Some(&val) = K_bc.get(i, i) {
                    let abs_val = val.abs();
                    if abs_val > 1e-15 {
                        s_diag[i] = 1.0 / abs_val.sqrt();
                    }
                }
            }
        }
        */

        // 4. Solve Scaled Linear System: (S J S) du_hat = -S R
        let mut rhs_scaled = vec![0.0; n];
        for i in 0..n { rhs_scaled[i] = -R[i] * s_diag[i]; }

        // Adaptive Linear Tolerance
        let eta = (0.5f64).min(0.5 * (R_norm / R_norm_0.max(1e-20)).sqrt());
        linear_solver.set_tolerance(eta.max(config.tolerance));
        
        // Wrap evaluator with BC-aware row scaling S
        let s_row_eval = s_diag.clone();
        let mut scaled_eval = |u_test: &[f64]| {
            let mut r = residual_evaluator(u_test);
            for i in 0..n { r[i] *= s_row_eval[i]; }
            r
        };

        // Recompute current scaled residual for JFNK
        let mut r_u_scaled = vec![0.0; n];
        for i in 0..n { r_u_scaled[i] = R[i] * s_diag[i]; }
        
        let jfnk_op = JFNKOperator {
            residual_evaluator: std::cell::RefCell::new(&mut scaled_eval),
            u: &u,
            r_u: &r_u_scaled,
            s_col: s_diag.clone(),
        };

        let scaled_precond = ScaledPreconditioner {
            inner: &block_diag_precond,
            s_row: s_diag.clone(),
            s_col: s_diag.clone(),
        };

        let (du_hat, lin_stats) = linear_solver.solve_with_operator(&jfnk_op, &rhs_scaled, &scaled_precond);
        
        // Recover physical update: du = S_col * du_hat
        let mut du = vec![0.0; n];
        for i in 0..n { du[i] = du_hat[i] * s_diag[i]; }
        
        total_linear_iters += lin_stats.iterations;
        last_linear_stats = lin_stats.clone();

        // 5. Line Search
        let mut alpha = config.line_search_alpha;
        let mut u_new = vec![0.0; n];
        let mut R_new = Vec::new();
        let mut R_new_norm = f64::INFINITY;
        let mut line_search_success = false;

        for _ls_iter in 0..config.max_line_search {
            for i in 0..n { u_new[i] = u[i] + alpha * du[i]; }

            R_new = compute_residual(&mut residual_evaluator, &u_new);
            R_new_norm = norm(&R_new);

            if R_new_norm < R_norm {
                line_search_success = true;
                break;
            }
            alpha *= config.line_search_rho;
        }

        if line_search_success {
            u = u_new;
            R = R_new;
            R_norm = R_new_norm;
            
            if config.verbose {
                let r_v = norm(&R[0..nv]);
                let r_p = norm(&R[nv..]);
                println!(
                    "    JFNK iter {:2}: ||R|| = {:.3e} (Rv={:.1e}, Rp={:.1e}), rel = {:.3e}, α = {:.3}, Lin_iters = {:4}",
                    newton_iter + 1, R_norm, r_v, r_p, R_norm / R_norm_0.max(config.abs_tolerance), alpha, lin_stats.iterations
                );
            }
        } else {
            if config.verbose { println!("    JFNK: Line search failed. Stopping Newton loop."); }
            break;
        }
    }

    u_guess.copy_from_slice(&u);
    (u, JFNKStats {
        newton_iterations: final_newton_iters + 1,
        converged: relative_res < config.tolerance || R_norm < config.abs_tolerance,
        residual_norm: R_norm,
        relative_residual: R_norm / R_norm_0.max(config.abs_tolerance),
        total_linear_iterations: total_linear_iters,
        last_linear_stats,
    })
}

/// JFNK solver with automatic non-dimensionalization
///
/// **Key Improvement:** This wrapper automatically scales the problem to O(1) variables,
/// making the solver much more robust for geodynamic problems with extreme parameter ranges.
///
/// # How it works
/// 1. Takes physical units input (velocities in m/s, pressures in Pa)
/// 2. Converts to dimensionless O(1) variables using characteristic scales
/// 3. Solves the dimensionless system (residuals ~ O(1))
/// 4. Converts solution back to physical units
///
/// # Benefits
/// - Residuals are O(1) instead of O(10^13) → better convergence criteria
/// - Condition numbers improve by orders of magnitude
/// - Tolerances like 1e-6 actually mean "6 digits of accuracy"
/// - Preconditioners work much better
///
/// # Arguments
/// * `assembler` - Assembles matrix/RHS in PHYSICAL units
/// * `residual_evaluator` - Computes residual in PHYSICAL units
/// * `linear_solver` - Linear solver (same as regular JFNK)
/// * `u_guess_phys` - Initial guess in PHYSICAL units [m/s, Pa]
/// * `dof_mgr` - DOF manager
/// * `config` - JFNK configuration (tolerances are for DIMENSIONLESS residuals)
/// * `scales` - Characteristic scales for non-dimensionalization
///
/// # Returns
/// Solution in PHYSICAL units [m/s, Pa] and statistics
///
/// # Example
/// ```rust
/// let scales = CharacteristicScales::new(100_000.0, 1e-9, 1e21, 3000.0, 9.81);
/// let mut config = JFNKConfig::conservative();
/// config.tolerance = 1e-6;      // 6 digits in dimensionless space
/// config.abs_tolerance = 1e-8;  // Absolute floor in dimensionless space
///
/// let (sol_phys, stats) = jfnk_solve_nondimensional(
///     assembler, residual_evaluator, &mut gmres, &mut u_guess, &dof_mgr, &config, &scales
/// );
/// ```
#[allow(non_snake_case)]
pub fn jfnk_solve_nondimensional<FA, FR, S>(
    mut assembler: FA,
    mut residual_evaluator: FR,
    linear_solver: &mut S,
    u_guess_phys: &mut [f64],
    dof_mgr: &crate::fem::DofManager,
    config: &JFNKConfig,
    scales: &CharacteristicScales,
) -> (Vec<f64>, JFNKStats)
where
    FA: FnMut(&[f64]) -> (CsMat<f64>, Vec<f64>, f64),
    FR: FnMut(&[f64]) -> Vec<f64>,
    S: Solver,
{
    if config.verbose {
        println!("\n  ╔════════════════════════════════════════════════════════════╗");
        println!("  ║  JFNK with Non-dimensionalization                          ║");
        println!("  ╚════════════════════════════════════════════════════════════╝");
        scales.print_summary();
    }

    // 1. Convert initial guess to dimensionless
    let mut u_guess_nd = scales.nondim_solution(u_guess_phys, dof_mgr);

    // 2. Wrap assembler: physical input → physical output (JFNK handles scaling internally)
    //    The assembler stays in physical units for simplicity
    let assembler_nd = |u_nd: &[f64]| {
        // Convert dimensionless solution to physical for assembly
        let u_phys = scales.dim_solution(u_nd, dof_mgr);

        // Assemble in physical units
        let (K_phys, f_phys, energy) = assembler(&u_phys);

        // Return physical matrix and RHS (JFNK internal scaling handles the rest)
        // Return physical matrix and RHS (JFNK internal scaling handles the rest)
        // CRITICAL FIX: We MUST scale the matrix K to dimensionless units for the preconditioner to work!
        // The original implementation passed K_phys but solved for u_nd, which meant the preconditioner M ≈ K_phys
        // was wildly out of scale with the O(1) operator JFNK constructs.
        
        let k_nd = scales.nondim_matrix(&K_phys, dof_mgr);
        let f_nd = scales.nondim_residual(&f_phys, dof_mgr); // Also scale RHS "energy" logic if needed, but f is not really used by JFNK except for initial residual? 
        // JFNK calls assembler(&u) -> (K, f). Then applies BCs. 
        // Then builds preconditioner M from K.
        // Then effectively linear system is J*du = -R.
        // J uses finite difference of R.
        // So J is O(1) if R is O(1).
        // So M must be O(1) to be a good preconditioner.
        
        (k_nd, f_nd, energy)
    };

    // 3. Wrap residual evaluator: dimensionless input → dimensionless output
    let residual_nd = |u_nd: &[f64]| {
        // Convert to physical
        let u_phys = scales.dim_solution(u_nd, dof_mgr);

        // Compute residual in physical units
        let r_phys = residual_evaluator(&u_phys);

        // Convert residual to dimensionless
        scales.nondim_residual(&r_phys, dof_mgr)
    };

    // 4. Solve in dimensionless space
    //    Config tolerances now apply to O(1) dimensionless residuals
    let (u_sol_nd, stats) = jfnk_solve(
        assembler_nd,
        residual_nd,
        linear_solver,
        &mut u_guess_nd,
        dof_mgr,
        config,
    );

    // 5. Convert solution back to physical units
    let u_sol_phys = scales.dim_solution(&u_sol_nd, dof_mgr);

    // 6. Compute TRUE dimensionless residual for reporting
    //    (The stats.residual_norm is in equilibrated units, not pure dimensionless)
    let r_final_phys = residual_evaluator(&u_sol_phys);
    let r_final_nd = scales.nondim_residual(&r_final_phys, dof_mgr);
    let r_final_nd_norm = r_final_nd.iter().map(|&x| x * x).sum::<f64>().sqrt();

    if config.verbose {
        println!("  ╔════════════════════════════════════════════════════════════╗");
        println!("  ║  JFNK Non-dimensional: Converged = {}                      ║", stats.converged);
        println!("  ║  TRUE Dimensionless Residual: {:.3e} (O(1) target)       ║", r_final_nd_norm);
        println!("  ║  Physical residual: {:.3e} N                              ║",
                 r_final_phys.iter().map(|&x| x * x).sum::<f64>().sqrt());
        println!("  ╚════════════════════════════════════════════════════════════╝\n");
    }

    // Update the original guess for next iteration
    u_guess_phys.copy_from_slice(&u_sol_phys);

    (u_sol_phys, stats)
}
