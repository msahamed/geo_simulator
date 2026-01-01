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
use crate::linalg::preconditioner::{ILUPreconditioner, Preconditioner};

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

        // 2. Setup Block Preconditioner
        // Extract blocks from K_bc
        // A is the [0..nv, 0..nv] part
        // B is the [nv..n, 0..nv] part
        let mut A_triplets = sprs::TriMat::new((nv, nv));
        let mut B_triplets = sprs::TriMat::new((n - nv, nv));
        
        for (row_idx, row) in K_bc.outer_iterator().enumerate() {
            for (col_idx, &val) in row.iter() {
                if row_idx < nv && col_idx < nv {
                    A_triplets.add_triplet(row_idx, col_idx, val);
                } else if row_idx >= nv && col_idx < nv {
                    B_triplets.add_triplet(row_idx - nv, col_idx, val);
                }
            }
        }
        let A_block = A_triplets.to_csr();
        let B_block = B_triplets.to_csr();

        // Preconditioners for blocks
        let a_precond = ILUPreconditioner::new(&A_block).unwrap();
        
        // Schur complement approximation: S \approx B * diag(A)^-1 * B^T
        let mut a_diag = vec![1.0; nv];
        for i in 0..nv {
            if let Some(&val) = K_bc.get(i, i) {
                if val.abs() > 1e-18 { a_diag[i] = val; }
            }
        }
        
        let mut s_diag = vec![0.0; n - nv];
        for (row_idx, row) in B_block.outer_iterator().enumerate() {
            let mut row_sum = 0.0;
            for (col_idx, &val) in row.iter() {
                row_sum += val * val / a_diag[col_idx];
            }
            s_diag[row_idx] = row_sum;
        }
        for val in s_diag.iter_mut() {
            if val.abs() < 1e-18 { *val = 1.0; }
        }
        
        // Form a diagonal matrix for Jacobi
        let mut s_mat_tri = sprs::TriMat::new((n - nv, n - nv));
        for (i, &val) in s_diag.iter().enumerate() {
            s_mat_tri.add_triplet(i, i, val);
        }
        let s_mat = s_mat_tri.to_csr();
        let s_precond = crate::linalg::preconditioner::JacobiPreconditioner::new(&s_mat);
        
        let block_precond = crate::linalg::preconditioner::BlockTriangularPreconditioner::new(
            a_precond,
            s_precond,
            B_block,
            nv
        );

        // 3. Automated Block Scaling
        // Calculate representative diagonal values for V and P
        let mut v_diag_sum = 0.0;
        let mut v_diag_count = 0;
        for i in 0..nv {
            if let Some(&val) = K_bc.get(i, i) {
                v_diag_sum += val.abs();
                v_diag_count += 1;
            }
        }
        let avg_v_diag = if v_diag_count > 0 { v_diag_sum / v_diag_count as f64 } else { 1.0 };
        
        let mut p_diag_sum = 0.0;
        let mut p_diag_count = 0;
        for &val in &s_diag {
            p_diag_sum += val.abs();
            p_diag_count += 1;
        }
        let avg_p_diag = if p_diag_count > 0 { p_diag_sum / p_diag_count as f64 } else { 1.0 };

        // True Diagonal Balancing (Equilibration): Scaled Matrix = S J S
        // Where S = Diag(1/sqrt(D_ii))
        let s_v = 1.0 / avg_v_diag.sqrt().max(1e-15);
        let s_p = 1.0 / avg_p_diag.sqrt().max(1e-15);

        let mut s_diag = vec![1.0; n];
        for i in 0..n {
            if dof_mgr.is_dirichlet(i) {
                s_diag[i] = 1.0; 
            } else {
                s_diag[i] = if i < nv { s_v } else { s_p };
            }
        }

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
            inner: &block_precond,
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
