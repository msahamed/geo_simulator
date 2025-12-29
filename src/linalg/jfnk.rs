/// Jacobian-Free Newton-Krylov (JFNK) solver for nonlinear viscoplastic systems
///
/// **Algorithm:**
/// Newton iteration solves: J(u) * δu = -R(u)
/// where:
///   - R(u) = K(u)*u - f  (residual)
///   - J = dR/du (Jacobian)
///   - u_{k+1} = u_k + α*δu (with line search)
///
/// **Jacobian-Free:** Instead of computing J explicitly, approximate J*v:
///   J*v ≈ [R(u + ε*v) - R(u)] / ε
///
/// where ε = sqrt(machine_eps) * ||u|| / ||v|| ≈ 1e-8
///
/// **Benefits:**
/// - Quadratic convergence (vs linear for Picard)
/// - No need to form/store Jacobian matrix
/// - Works with existing Krylov solvers (CG, GMRES)
///
/// # References
/// - Knoll & Keyes (2004), "Jacobian-free Newton-Krylov methods"
/// - Elman et al. (2014), "Finite Elements and Fast Iterative Solvers"
/// - Glerum et al. (2018), "Nonlinear viscoplasticity in ASPECT"

use sprs::CsMat;
use std::cell::RefCell;
use crate::linalg::{Solver, SolverStats, LinearOperator};

/// Configuration for JFNK solver
#[derive(Debug, Clone)]
pub struct JFNKConfig {
    /// Maximum Newton iterations
    pub max_newton_iterations: usize,

    /// Convergence tolerance on ||R(u)|| / ||R(u_0)||
    pub tolerance: f64,

    /// Absolute residual tolerance (for cases where initial residual is small)
    pub abs_tolerance: f64,

    /// Finite difference parameter for Jacobian-vector products
    /// ε = epsilon_fd * ||u|| / ||v||
    pub epsilon_fd: f64,

    /// Line search parameters
    pub max_line_search: usize,
    pub line_search_alpha: f64,  // Initial step length
    pub line_search_rho: f64,    // Reduction factor (0.5 = backtracking)

    /// Verbose output
    pub verbose: bool,
}

impl Default for JFNKConfig {
    fn default() -> Self {
        Self {
            max_newton_iterations: 20,
            tolerance: 1e-4,           // 0.01% relative residual
            abs_tolerance: 1e6,        // Absolute residual (Newtons)
            epsilon_fd: 1e-8,          // sqrt(machine_epsilon)
            max_line_search: 10,
            line_search_alpha: 1.0,    // Full Newton step initially
            line_search_rho: 0.5,      // Halve step length on failure
            verbose: false,
        }
    }
}

impl JFNKConfig {
    /// Conservative config for difficult problems
    pub fn conservative() -> Self {
        Self {
            max_newton_iterations: 30,
            tolerance: 1e-4,
            abs_tolerance: 1e7,
            epsilon_fd: 1e-7, // Standard JFNK epsilon for stable derivatives
            max_line_search: 15,
            line_search_alpha: 1.0,    // Expert suggestion: start with full step
            line_search_rho: 0.5,
            verbose: true,
        }
    }
}

/// Statistics from JFNK solve
#[derive(Debug, Clone)]
pub struct JFNKStats {
    /// Number of Newton iterations
    pub newton_iterations: usize,

    /// Did it converge?
    pub converged: bool,

    /// Final residual norm
    pub residual_norm: f64,

    /// Final relative residual ||R|| / ||R_0||
    pub relative_residual: f64,

    /// Total linear solver iterations (sum over all Newton steps)
    pub total_linear_iterations: usize,

    /// Linear solver stats from last Newton iteration
    pub last_linear_stats: SolverStats,
}

/// Jacobian-vector product: J*v ≈ [R(u + ε*v) - R(u)] / ε
fn jacobian_vector_product<F>(
    residual_fn: &mut F,
    u: &[f64],
    v: &[f64],
    r_u: &[f64],
    epsilon: f64,
) -> Vec<f64>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let n = u.len();

    // u_perturbed = u + ε*v
    let mut u_perturbed = vec![0.0; n];
    for i in 0..n {
        u_perturbed[i] = u[i] + epsilon * v[i];
    }

    // R(u + ε*v)
    let r_perturbed = residual_fn(&u_perturbed);

    // J*v ≈ [R(u + ε*v) - R(u)] / ε
    let mut jv = vec![0.0; n];
    for i in 0..n {
        jv[i] = (r_perturbed[i] - r_u[i]) / epsilon;
    }

    jv
}

/// Matrix-free Jacobian operator for JFNK
struct JFNKJacobian<'a, F> {
    residual_fn: &'a RefCell<F>,
    dof_mgr: &'a crate::fem::DofManager,
    u: &'a [f64],
    r_u: &'a [f64],
    epsilon_fd: f64,
    char_diag: f64,
}

impl<'a, F> LinearOperator for JFNKJacobian<'a, F>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    fn apply(&self, v: &[f64]) -> Vec<f64> {
        let n = v.len();
        
        // Stabilize: Ensure v has zeros at Dirichlet nodes for the perturbation
        // This makes the JFNK operator consistent with the zeroed columns in the preconditioner.
        let mut v_free = v.to_vec();
        for i in 0..n {
            if self.dof_mgr.is_dirichlet(i) {
                v_free[i] = 0.0;
            }
        }

        let v_norm = norm(&v_free);
        if v_norm < 1e-18 {
            // If v is purely Dirichlet, the operator acts as char_diag * I
            let mut jv = vec![0.0; n];
            for i in 0..n {
                if self.dof_mgr.is_dirichlet(i) {
                    jv[i] = v[i] * self.char_diag;
                }
            }
            return jv;
        }

        let u_norm = norm(self.u);
        let epsilon = self.epsilon_fd * (1.0 + u_norm) / v_norm;

        let mut residual_fn = self.residual_fn.borrow_mut();
        let mut jv = jacobian_vector_product(&mut *residual_fn, self.u, &v_free, self.r_u, epsilon);

        // Apply Dirichlet BC semantics (system acts as scaled identity for constrained DOFs)
        for i in 0..jv.len() {
            if self.dof_mgr.is_dirichlet(i) {
                jv[i] = v[i] * self.char_diag;
            }
        }

        jv
    }

    fn rows(&self) -> usize { self.u.len() }
    fn cols(&self) -> usize { self.u.len() }
}

/// Compute L2 norm
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// JFNK solver for nonlinear systems
///
/// # Example
/// ```rust
/// use geo_simulator::*;
///
/// let config = JFNKConfig::default();
/// let mut linear_solver = ConjugateGradient::new()
///     .with_max_iterations(10000)
///     .with_tolerance(1e-8);
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
pub fn jfnk_solve<RF, PF, S>(
    mut residual_fn: RF,
    mut precond_fn: PF,
    linear_solver: &mut S,
    u_guess: &mut [f64],
    dof_mgr: &crate::fem::DofManager,
    config: &JFNKConfig,
) -> (Vec<f64>, JFNKStats)
where
    RF: FnMut(&[f64]) -> Vec<f64>,
    PF: FnMut(&[f64]) -> CsMat<f64>,
    S: Solver,
{
    let n = u_guess.len();
    let mut u = u_guess.to_vec();
    let mut total_linear_iters = 0;
    let mut last_linear_stats = SolverStats::new();

    // Compute initial residual
    let mut r_vec = residual_fn(&u);
    let r_norm_0 = norm(&r_vec);
    let mut r_norm = r_norm_0;

    if config.verbose {
        println!("    JFNK: Initial filtered residual = {:.3e}", r_norm_0);
    }

    for newton_iter in 0..config.max_newton_iterations {
        // Check convergence
        let relative_res = if r_norm_0 > config.abs_tolerance {
            r_norm / r_norm_0
        } else {
            r_norm / config.abs_tolerance
        };

        if relative_res < config.tolerance && r_norm < config.abs_tolerance {
            if config.verbose {
                println!("    JFNK: Converged in {} iterations", newton_iter);
            }

            u_guess.copy_from_slice(&u);
            return (
                u,
                JFNKStats {
                    newton_iterations: newton_iter,
                    converged: true,
                    residual_norm: r_norm,
                    relative_residual: relative_res,
                    total_linear_iterations: total_linear_iters,
                    last_linear_stats,
                },
            );
        }

        // True JFNK: Create matrix-free Jacobian operator
        let residual_fn_ref = RefCell::new(residual_fn);

        // Use precond_fn for Jacobi preconditioning (expensive assembly only once per Newton step)
        let k_mat = precond_fn(&u);
        let k_bc = crate::fem::Assembler::apply_dirichlet_bcs_only_matrix(&k_mat, dof_mgr);
        
        // Find characteristic scale for BC scaling
        let mut char_diag = 1.0;
        for (i, row) in k_bc.outer_iterator().enumerate() {
            if dof_mgr.is_dirichlet(i) {
                for (j, &val) in row.iter() {
                    if i == j {
                        char_diag = val;
                        break;
                    }
                }
                break;
            }
        }
        
        if config.verbose {
            println!("    JFNK: BC scaling (char_diag) = {:.3e}", char_diag);
        }

        let jacobian = JFNKJacobian {
            residual_fn: &residual_fn_ref,
            dof_mgr,
            u: &u,
            r_u: &r_vec,
            epsilon_fd: config.epsilon_fd,
            char_diag,
        };

        let precond = crate::linalg::preconditioner::JacobiPreconditioner::new(&k_bc);

        // RHS = -R(u)
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            rhs[i] = -r_vec[i];
        }

        // Solve J * δu = -R for Newton step δu
        let (delta_u, lin_stats) = linear_solver.solve_with_operator(&jacobian, &rhs, &precond);
        
        // Recover closures
        residual_fn = residual_fn_ref.into_inner();
        
        total_linear_iters += lin_stats.iterations;
        last_linear_stats = lin_stats.clone();

        if !lin_stats.converged {
            if config.verbose {
                println!("    JFNK: Linear solver failed at Newton iter {}", newton_iter);
            }
            // Return current solution even if not converged
            u_guess.copy_from_slice(&u);
            return (
                u,
                JFNKStats {
                    newton_iterations: newton_iter + 1,
                    converged: false,
                    residual_norm: r_norm,
                    relative_residual: relative_res,
                    total_linear_iterations: total_linear_iters,
                    last_linear_stats,
                },
            );
        }

        // Line search: find α such that ||R(u + α*δu)|| < ||R(u)||
        let mut alpha = config.line_search_alpha;
        let mut u_new = vec![0.0; n];

        for _ls_iter in 0..config.max_line_search {
            // u_new = u + α*δu
            for i in 0..n {
                u_new[i] = u[i] + alpha * delta_u[i];
            }

            // Compute R(u_new)
            let r_new = residual_fn(&u_new);
            let r_new_norm_val = norm(&r_new);

            // Accept if residual decreased
            if r_new_norm_val < r_norm {
                r_vec = r_new;
                r_norm = r_new_norm_val;
                break;
            }

            // Reduce step length
            alpha *= config.line_search_rho;
        }

        // Update solution
        u = u_new;
        // r_norm is already updated in the loop above

        if config.verbose {
            println!(
                "    JFNK iter {:2}: ||R|| = {:.3e}, rel = {:.3e}, α = {:.3}, linear_iters = {:4}, linear_res = {:.2e}",
                newton_iter + 1,
                r_norm,
                relative_res,
                alpha,
                lin_stats.iterations,
                lin_stats.residual_norm
            );
        }
    }

    // Max iterations reached
    if config.verbose {
        println!("    JFNK: Max iterations reached (residual = {:.3e})", r_norm);
    }

    u_guess.copy_from_slice(&u);
    (
        u,
        JFNKStats {
            newton_iterations: config.max_newton_iterations,
            converged: false,
            residual_norm: r_norm,
            relative_residual: r_norm / r_norm_0.max(config.abs_tolerance),
            total_linear_iterations: total_linear_iters,
            last_linear_stats,
        },
    )
}
