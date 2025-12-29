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

/// Compute residual: R(u) = K(u)*u - f (with BCs applied)
fn compute_residual<F>(
    assembler: &mut F,
    u: &[f64],
    dof_mgr: &crate::fem::DofManager,
) -> Vec<f64>
where
    F: FnMut(&[f64]) -> (CsMat<f64>, Vec<f64>),
{
    let (K, f) = assembler(u);
    let (K_bc, f_bc) = crate::fem::Assembler::apply_dirichlet_bcs(&K, &f, dof_mgr);

    let n = u.len();
    let mut residual = vec![0.0; n];

    // R = K*u - f
    for (row_idx, row) in K_bc.outer_iterator().enumerate() {
        let mut ku = 0.0;
        for (col_idx, &val) in row.iter() {
            ku += val * u[col_idx];
        }
        residual[row_idx] = ku - f_bc[row_idx];
    }

    residual
}

/// Compute L2 norm
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// JFNK solver - quasi-Newton with line search
///
/// # Example
/// ```rust
/// use geo_simulator::*;
///
/// let config = JFNKConfig::default();
/// let mut linear_solver = BiCGSTAB::new()
///     .with_max_iterations(10000)
///     .with_tolerance(1e-8)
///     .with_abs_tolerance(1e7);
///
/// let mut velocity = vec![0.0; n_dofs];
///
/// let assembler = |v: &[f64]| -> (CsMat<f64>, Vec<f64>) {
///     let K = assemble_stokes_vep(..., v, ...);
///     let f = compute_rhs(...);
///     (K, f)
/// };
///
/// let (v_solution, stats) = jfnk_solve(
///     assembler,
///     &mut linear_solver,
///     &mut velocity,
///     &dof_mgr,
///     &config,
/// );
/// ```
#[allow(non_snake_case)]
pub fn jfnk_solve<F, S>(
    mut assembler: F,
    linear_solver: &mut S,
    u_guess: &mut [f64],
    dof_mgr: &crate::fem::DofManager,
    config: &JFNKConfig,
) -> (Vec<f64>, JFNKStats)
where
    F: FnMut(&[f64]) -> (CsMat<f64>, Vec<f64>),
    S: Solver,
{
    let n = u_guess.len();
    let mut u = u_guess.to_vec();
    let mut total_linear_iters = 0;
    let mut last_linear_stats = SolverStats::new();

    // Compute initial residual
    let mut R = compute_residual(&mut assembler, &u, dof_mgr);
    let R_norm_0 = norm(&R);
    let mut R_norm = R_norm_0;

    if config.verbose {
        println!("    JFNK: Initial residual = {:.3e}", R_norm_0);
    }

    for newton_iter in 0..config.max_newton_iterations {
        // Check convergence (both relative AND absolute)
        let relative_res = if R_norm_0 > config.abs_tolerance {
            R_norm / R_norm_0
        } else {
            R_norm / config.abs_tolerance
        };

        if relative_res < config.tolerance || R_norm < config.abs_tolerance {
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

        // Assemble current tangent stiffness K(u_k)
        let (K, _f) = assembler(&u);
        let (K_bc, _) = crate::fem::Assembler::apply_dirichlet_bcs(&K, &vec![0.0; n], dof_mgr);

        // RHS = -R(u_k)
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            rhs[i] = -R[i];
        }

        // Solve K(u_k) * δu = -R(u_k) for Newton step
        let (delta_u, lin_stats) = linear_solver.solve(&K_bc, &rhs);
        total_linear_iters += lin_stats.iterations;
        last_linear_stats = lin_stats.clone();

        if !lin_stats.converged {
            if config.verbose {
                println!("    JFNK: Linear solver failed at Newton iter {}", newton_iter);
                println!("          BiCGSTAB: {} iters, residual = {:.3e}",
                         lin_stats.iterations, lin_stats.residual_norm);
            }
            // Return current solution even if not converged
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

        // Line search: find α such that ||R(u + α*δu)|| < ||R(u)||
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
            R_new = compute_residual(&mut assembler, &u_new, dof_mgr);
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

        // Update solution
        u = u_new;
        R = R_new;
        R_norm = R_new_norm;

        if config.verbose {
            println!(
                "    JFNK iter {:2}: ||R|| = {:.3e}, rel = {:.3e}, α = {:.3}, BiCG_iters = {:4}",
                newton_iter + 1,
                R_norm,
                relative_res,
                alpha,
                lin_stats.iterations
            );
        }

        // Check if line search completely failed
        if !line_search_success && config.verbose {
            println!("    JFNK: WARNING - Line search did not reduce residual");
        }
    }

    // Max iterations reached
    if config.verbose {
        println!("    JFNK: Max iterations reached (residual = {:.3e})", R_norm);
    }

    u_guess.copy_from_slice(&u);
    (
        u,
        JFNKStats {
            newton_iterations: config.max_newton_iterations,
            converged: false,
            residual_norm: R_norm,
            relative_residual: R_norm / R_norm_0.max(config.abs_tolerance),
            total_linear_iterations: total_linear_iters,
            last_linear_stats,
        },
    )
}
