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
use crate::linalg::{Solver, SolverStats};
use crate::linalg::preconditioner::ILUPreconditioner;

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
    pub verbose: bool,
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
    penalty: f64,
    char_diag: f64,
) -> Vec<f64>
where
    FR: FnMut(&[f64], f64, f64) -> Vec<f64>,
{
    residual_evaluator(u, penalty, char_diag)
}

/// Compute L2 norm
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

struct ScaledJFNKOperator<'a, FR> {
    residual_evaluator: std::cell::RefCell<&'a mut FR>,
    u: &'a [f64],
    r_u: &'a [f64],
    s: &'a [f64],
    penalty: f64,
    char_diag: f64,
}

impl<'a, FR> crate::linalg::solver::LinearOperator for ScaledJFNKOperator<'a, FR>
where
    FR: FnMut(&[f64], f64, f64) -> Vec<f64>,
{
    fn apply(&self, v_scaled: &[f64]) -> Vec<f64> {
        let n = self.u.len();
        
        // 1. Un-scale input vector: v = S * v_scaled
        let mut v = vec![0.0; n];
        for i in 0..n {
            v[i] = v_scaled[i] * self.s[i];
        }
        
        let v_norm = norm(&v);
        // Safety check for tiny vectors
        if v_norm < 1e-40 {
            return vec![0.0; n];
        }

        // 2. Compute epsilon: scale-aware finite difference step
        // For geodynamics, u_norm is often ~1e-9. 1.0 + u_norm is dominated by 1.0,
        // making epsilon too large. We use a base scale of 1e-6 or u_norm.
        let u_norm = norm(self.u);
        let epsilon = 1e-8 * (u_norm + 1e-6) / v_norm;

        // 3. u_perturbed = u + eps * v
        let mut u_perturbed = vec![0.0; n];
        for i in 0..n {
            u_perturbed[i] = self.u[i] + epsilon * v[i];
        }

        // 4. R_perturbed = R(u_perturbed)
        let r_perturbed = compute_residual(*self.residual_evaluator.borrow_mut(), &u_perturbed, self.penalty, self.char_diag);

        // 5. Jv = (R_perturbed - R_u) / eps
        let mut jv = vec![0.0; n];
        for i in 0..n {
            jv[i] = (r_perturbed[i] - self.r_u[i]) / epsilon;
        }

        // 6. Scale output: result = S * Jv
        let mut result = vec![0.0; n];
        for i in 0..n {
            result[i] = jv[i] * self.s[i];
        }

        result
    }

    fn rows(&self) -> usize { self.u.len() }
    fn cols(&self) -> usize { self.u.len() }
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
    FR: FnMut(&[f64], f64, f64) -> Vec<f64>,
    S: Solver,
{
    let n = u_guess.len();
    let mut u = u_guess.to_vec();
    let mut total_linear_iters = 0;
    let mut last_linear_stats = SolverStats::new();

    let mut ilu_precond: Option<ILUPreconditioner> = None;

    // 0. Compute initial residual and penalty (from initial matrix state)
    let (K_initial, f_initial, mut penalty) = assembler(&u);
    let (K_bc_initial, f_bc_initial) = crate::fem::Assembler::apply_dirichlet_bcs(&K_initial, &f_initial, dof_mgr);
    
    // Find characteristic diagonal scale for BC synchronization
    let mut char_diag = 1.0;
    for (i, row) in K_bc_initial.outer_iterator().enumerate() {
        if let Some(&val) = row.get(i) {
            if val.abs() > char_diag { char_diag = val.abs(); }
        }
    }
    
    // Compute initial R_norm correctly from K_bc and f_bc using initial char_diag sync
    let mut R = vec![0.0; n];
    for (row_idx, row) in K_bc_initial.outer_iterator().enumerate() {
        let mut ku = 0.0;
        for (col_idx, &val) in row.iter() {
            ku += val * u[col_idx];
        }
        R[row_idx] = ku - f_bc_initial[row_idx];
    }
    
    // Also apply char_diag scaling to initial R manually (it was already done in compute_residual for later ones)
    for i in 0..n {
        if dof_mgr.is_dirichlet(i) {
             R[i] *= char_diag;
        }
    }
    
    let R_norm_0 = norm(&R);
    let mut R_norm = R_norm_0;

    if config.verbose {
        println!("    JFNK: Initial residual = {:.3e}", R_norm_0);
    }

    let mut relative_res = 1.0;
    let mut final_newton_iters = 0;
    for newton_iter in 0..config.max_newton_iterations {
        final_newton_iters = newton_iter + 1;
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
            return (
                u,
                JFNKStats {
                    newton_iterations: newton_iter,
                    converged: true,
                    residual_norm: R_norm,
                    relative_residual: relative_res,
                    total_linear_iterations: total_linear_iters,
                    last_linear_stats,
                },
            );
        }

        // 1. Assemble current tangent stiffness K(u_k)
        let timer = std::time::Instant::now();
        let (K, _f, p_new) = assembler(&u);
        penalty = p_new;
        let (K_bc, _) = crate::fem::Assembler::apply_dirichlet_bcs(&K, &_f, dof_mgr);
        if config.verbose { println!("    JFNK: Assembly took {:?}", timer.elapsed()); }

        // Find characteristic diagonal scale for BC synchronization
        let mut char_diag = 1.0;
        for (i, row) in K_bc.outer_iterator().enumerate() {
            if let Some(&val) = row.get(i) {
                if val.abs() > char_diag { char_diag = val.abs(); }
            }
        }

        // Update R to be consistent with the new penalty AND BC scaling
        R = compute_residual(&mut residual_evaluator, &u, penalty, char_diag);
        R_norm = norm(&R);

        // 2. Setup/Update ILU(0) preconditioner
        let timer = std::time::Instant::now();
        let setup_result = match &mut ilu_precond {
            Some(p) => p.update(&K_bc),
            None => {
                match ILUPreconditioner::new(&K_bc) {
                    Ok(p) => {
                        ilu_precond = Some(p);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
        };
        if config.verbose { println!("    JFNK: ILU(0) setup took {:?}", timer.elapsed()); }

        if let Err(e) = setup_result {
            if config.verbose {
                println!("    JFNK: ILU(0) setup failed: {}", e);
            }
            u_guess.copy_from_slice(&u);
            return (
                u,
                JFNKStats {
                    newton_iterations: newton_iter + 1,
                    converged: false,
                    residual_norm: R_norm,
                    relative_residual: relative_res,
                    total_linear_iterations: total_linear_iters,
                    last_linear_stats,
                },
            );
        }

        // 3. RHS = -R(u_k)
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            rhs[i] = -R[i];
        }

        // 4. Symmetric Diagonal Scaling (Jacobi Scaling)
        // Transform the system K * du = rhs into (S*K*S) * (S^-1*du) = S*rhs
        // where S = D^-1/2. This makes the diagonal of the scaled matrix exactly 1.0.
        let mut s = vec![1.0; n];
        for i in 0..n {
            if let Some(&val) = K_bc.get(i, i) {
                if val.abs() > 1e-30 {
                    s[i] = 1.0 / val.abs().sqrt();
                }
            }
        }

        // Scale RHS: rhs_scaled = S * rhs
        let rhs_scaled: Vec<f64> = rhs.iter().zip(s.iter()).map(|(&bi, &si)| bi * si).collect();

        // Scale Matrix: K_scaled = S * K * S
        // For CSR, we iterate over rows and multiply each entry K[i,j] by s[i] * s[j]
        let mut K_scaled = K_bc.clone();
        for (i, mut row) in K_scaled.outer_iterator_mut().enumerate() {
            let si = s[i];
            for (j, val) in row.iter_mut() {
                *val *= si * s[j];
            }
        }

        // 5. Update/Setup Preconditioner for the SCALED matrix
        let setup_result = match &mut ilu_precond {
            Some(p) => p.update(&K_scaled),
            None => {
                match ILUPreconditioner::new(&K_scaled) {
                    Ok(p) => {
                        ilu_precond = Some(p);
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
        };

        if let Err(e) = setup_result {
            if config.verbose { println!("    JFNK: ILU(0) setup failed: {}", e); }
            u_guess.copy_from_slice(&u);
            return (u, JFNKStats { 
                newton_iterations: newton_iter + 1, converged: false, residual_norm: R_norm, 
                relative_residual: relative_res, total_linear_iterations: total_linear_iters, last_linear_stats 
            });
        }

        // 6. Solve system J_scaled * du_scaled = rhs_scaled
        // Use true JFNK operator for the solve, but Picard matrix for the preconditioner
        let jfnk_op = ScaledJFNKOperator {
            residual_evaluator: std::cell::RefCell::new(&mut residual_evaluator),
            u: &u,
            r_u: &R,
            s: &s,
            penalty,
            char_diag,
        };

        // Normalize linear solver absolute tolerance for the scaled system
        let original_abs_tol = linear_solver.abs_tolerance();
        let rhs_norm_scaled = norm(&rhs_scaled);
        let rhs_norm_orig = rhs.iter().map(|&x| x*x).sum::<f64>().sqrt();
        let scale_factor = if rhs_norm_orig > 1e-20 { rhs_norm_scaled / rhs_norm_orig } else { 1.0 };
        linear_solver.set_abs_tolerance(original_abs_tol * scale_factor);

        let (du_scaled, lin_stats) = linear_solver.solve_with_operator(&jfnk_op, &rhs_scaled, ilu_precond.as_ref().unwrap());
        
        // Restore solver state
        linear_solver.set_abs_tolerance(original_abs_tol);
        
        total_linear_iters += lin_stats.iterations;
        last_linear_stats = lin_stats.clone();

        // Un-scale solution: delta_u = S * du_scaled
        let delta_u: Vec<f64> = du_scaled.iter().zip(s.iter()).map(|(&ui, &si)| ui * si).collect();

        // RESILIENCE: Even if linear solver didn't reach target tol, try the Newton step
        // unless it completely failed (NaNs) or made zero progress.
        if !lin_stats.converged && lin_stats.iterations == 0 {
            if config.verbose {
                println!("    JFNK: Linear solver stalled at Newton iter {}", newton_iter);
            }
            u_guess.copy_from_slice(&u);
            return (
                u,
                JFNKStats {
                    newton_iterations: newton_iter + 1,
                    converged: false,
                    residual_norm: R_norm,
                    relative_residual: relative_res,
                    total_linear_iterations: total_linear_iters,
                    last_linear_stats,
                },
            );
        }

        // 7. Line search: find α such that ||R(u + α*δu)|| < ||R(u)||
        let mut alpha = config.line_search_alpha;
        let mut u_new = vec![0.0; n];
        let mut R_new = Vec::new();
        let mut R_new_norm = f64::INFINITY;
        let mut line_search_success = false;

        for ls_iter in 0..config.max_line_search {
            // u_trial = u + α*δu
            for i in 0..n {
                u_new[i] = u[i] + alpha * delta_u[i];
            }

            // Compute R(u_trial)
            R_new = compute_residual(&mut residual_evaluator, &u_new, penalty, char_diag);
            R_new_norm = norm(&R_new);

            // Accept if residual decreased
            if R_new_norm < R_norm {
                line_search_success = true;
                break;
            }

            // Reduce step length
            alpha *= config.line_search_rho;

            if config.verbose && ls_iter == config.max_line_search - 1 {
                println!("    JFNK: Line search failed (α = {:.3e})", alpha);
            }
        }

        // Compute current relative residual for reporting
        let current_rel_res = R_norm / R_norm_0.max(config.abs_tolerance);

        // Update solution only if line search succeeded
        if line_search_success {
            u = u_new;
            R = R_new;
            R_norm = R_new_norm;
            
            if config.verbose {
                println!(
                    "    JFNK iter {:2}: ||R|| = {:.3e}, rel = {:.3e}, α = {:.3}, BiCG_iters = {:4}",
                    newton_iter + 1,
                    R_norm,
                    current_rel_res,
                    alpha,
                    lin_stats.iterations
                );
            }
        } else {
            if config.verbose {
                println!("    JFNK: WARNING - Line search failed to reduce residual. Stopping Newton loop.");
            }
            break;
        }
    }

    // Newton loop finished without meeting convergence criteria
    if config.verbose {
        if final_newton_iters >= config.max_newton_iterations {
            println!("    JFNK: Max iterations reached (residual = {:.3e})", R_norm);
        } else {
            // Must have broken out early (e.g. line search failure)
            println!("    JFNK: Solver stopped early (residual = {:.3e})", R_norm);
        }
    }

    u_guess.copy_from_slice(&u);
    (
        u,
        JFNKStats {
            newton_iterations: final_newton_iters,
            converged: relative_res < config.tolerance || R_norm < config.abs_tolerance,
            residual_norm: R_norm,
            relative_residual: R_norm / R_norm_0.max(config.abs_tolerance),
            total_linear_iterations: total_linear_iters,
            last_linear_stats,
        },
    )
}
