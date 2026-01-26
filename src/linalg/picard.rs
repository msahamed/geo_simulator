/// Picard (fixed-point) iteration for nonlinear viscoplastic problems
///
/// **Problem**: For viscoplastic materials, the stiffness matrix K depends on
/// the solution u through the effective viscosity:
///
/// ```text
/// μ_eff(u) = min(μ_viscous, μ_plastic(ε̇(u)))
/// K(u) · u = f
/// ```
///
/// **Solution**: Iterate until K and u are consistent:
///
/// ```text
/// Loop k = 1, 2, 3, ...
///   1. Compute strain rate: ε̇^k = B · u^{k-1}
///   2. Update viscosity: μ_eff^k = f(ε̇^k)
///   3. Assemble: K^k = K(μ_eff^k)
///   4. Solve: K^k · u^k = f
///   5. Check: ||u^k - u^{k-1}|| < tol → converged
/// ```
///
/// **Under-relaxation** (for stability):
/// ```text
/// u^k = α · u^k_new + (1-α) · u^{k-1}
/// ```
/// where α ∈ (0, 1] (typically 0.5-0.7 for viscoplastic)
///
/// # References
/// - Moresi et al. (2003), "A Lagrangian integration point FEM"
/// - Popov & Sobolev (2008), "SLIM3D: A tool for 3D thermomechanical modeling"
/// - Glerum et al. (2018), "Nonlinear viscoplasticity in ASPECT"

use sprs::CsMat;
use crate::linalg::{Solver, SolverStats, AndersonAccelerator};

/// Configuration for Picard iteration
#[derive(Debug, Clone)]
pub struct PicardConfig {
    /// Maximum number of nonlinear iterations
    pub max_iterations: usize,

    /// Convergence tolerance on relative velocity change: ||Δu||/||u|| < tol
    pub tolerance: f64,

    /// Under-relaxation factor α ∈ (0, 1]
    /// - α = 1.0: Full Newton step (fast but may diverge)
    /// - α = 0.5: Conservative (slower but stable)
    /// - Typical for viscoplastic: 0.5-0.7
    pub relaxation: f64,

    /// Absolute tolerance for very small velocities (m/s)
    /// Prevents division by zero in relative tolerance
    pub abs_tolerance: f64,

    /// Enable Anderson Acceleration
    /// Dramatically improves convergence for oscillating problems (2-10x speedup)
    pub use_anderson: bool,

    /// Anderson acceleration depth (number of previous iterates to store)
    /// Typical values: 3-5. Higher = more memory but better acceleration
    pub anderson_depth: usize,

    /// Anderson regularization parameter (prevents ill-conditioning)
    /// Typical values: 1e-10 to 1e-6
    pub anderson_beta: f64,

    /// Enable adaptive damping (auto-reduce relaxation on oscillations)
    /// Works independently and complements Anderson acceleration
    pub use_adaptive_damping: bool,

    /// Minimum relaxation parameter (lower bound for adaptive damping)
    pub alpha_min: f64,

    /// Maximum relaxation parameter (upper bound for adaptive damping)
    pub alpha_max: f64,

    /// Damping reduction factor when oscillation detected
    pub damping_reduction: f64,

    /// Damping increase factor when converging well
    pub damping_increase: f64,
}

impl Default for PicardConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            tolerance: 1e-3,  // 0.1% relative change
            relaxation: 0.7,   // Moderate under-relaxation
            abs_tolerance: 1e-15,  // ~1 nm/s
            use_anderson: false,  // Disabled by default for backwards compatibility
            anderson_depth: 3,
            anderson_beta: 1e-8,
            use_adaptive_damping: false,  // Disabled by default
            alpha_min: 0.1,
            alpha_max: 0.9,
            damping_reduction: 0.7,  // Reduce by 30% on oscillation
            damping_increase: 1.05,   // Increase by 5% when converging well
        }
    }
}

impl PicardConfig {
    /// Create conservative config (slower, very stable)
    pub fn conservative() -> Self {
        Self {
            max_iterations: 50,  // Increased for difficult viscoplastic cases
            tolerance: 1e-4,
            relaxation: 0.5,
            abs_tolerance: 1e-15,
            use_anderson: false,
            anderson_depth: 3,
            anderson_beta: 1e-8,
            use_adaptive_damping: false,
            alpha_min: 0.1,
            alpha_max: 0.9,
            damping_reduction: 0.7,
            damping_increase: 1.05,
        }
    }

    /// Create aggressive config (faster, may diverge)
    pub fn aggressive() -> Self {
        Self {
            max_iterations: 15,
            tolerance: 1e-2,
            relaxation: 0.8,
            abs_tolerance: 1e-15,
            use_anderson: false,
            anderson_depth: 3,
            anderson_beta: 1e-8,
            use_adaptive_damping: false,
            alpha_min: 0.1,
            alpha_max: 0.9,
            damping_reduction: 0.7,
            damping_increase: 1.05,
        }
    }

    /// Create Anderson-accelerated config (recommended for oscillating problems)
    pub fn with_anderson() -> Self {
        Self {
            max_iterations: 30,
            tolerance: 1e-3,
            relaxation: 0.5,  // Still use relaxation with Anderson
            abs_tolerance: 1e-15,
            use_anderson: true,
            anderson_depth: 5,  // Deeper history for better acceleration
            anderson_beta: 1e-8,
            use_adaptive_damping: true,  // Also use adaptive damping as safety net
            alpha_min: 0.1,
            alpha_max: 0.7,
            damping_reduction: 0.7,
            damping_increase: 1.02,  // Conservative increase
        }
    }

    /// Create config with both Anderson and Adaptive Damping (most robust)
    pub fn robust() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-3,
            relaxation: 0.5,
            abs_tolerance: 1e-15,
            use_anderson: true,
            anderson_depth: 5,
            anderson_beta: 1e-8,
            use_adaptive_damping: true,
            alpha_min: 0.05,  // Can go very conservative
            alpha_max: 0.7,
            damping_reduction: 0.6,  // Aggressive reduction
            damping_increase: 1.01,  // Very conservative increase
        }
    }
}

/// Statistics from Picard iteration
#[derive(Debug, Clone)]
pub struct PicardStats {
    /// Number of nonlinear iterations performed
    pub iterations: usize,

    /// Did it converge?
    pub converged: bool,

    /// Final relative change ||Δu||/||u||
    pub relative_change: f64,

    /// Total linear solver iterations (sum over all Picard steps)
    pub total_linear_iterations: usize,

    /// Linear solver stats from last Picard iteration
    pub last_linear_stats: SolverStats,
}

/// Picard iteration solver for nonlinear problems
///
/// # Type Parameters
/// - `F`: Assembler function type
/// - `S`: Linear solver type
///
/// # Example
/// ```rust
/// use geo_simulator::*;
///
/// // Setup problem...
/// let config = PicardConfig::default();
/// let mut linear_solver = ConjugateGradient::new()
///     .with_max_iterations(5000)
///     .with_tolerance(1e-9);
///
/// // Initial guess
/// let mut velocity = vec![0.0; n_dofs];
///
/// // Assembler function (captures mesh, materials, etc.)
/// let assembler = |v: &[f64]| -> (CsMat<f64>, Vec<f64>) {
///     let K = assemble_stokes_vep(..., v, ...);
///     let f = compute_rhs(...);
///     (K, f)
/// };
///
///// Solve nonlinear system
/// let (v_solution, stats) = picard_solve(
///     assembler,
///     &mut linear_solver,
///     &mut velocity,
///     &dof_mgr,
///     &config,
///     None::<fn(&CsMat<f64>) -> Box<dyn crate::linalg::preconditioner::Preconditioner>>,
/// );
/// ```
pub fn picard_solve<F, S, P>(
    mut assembler: F,
    linear_solver: &mut S,
    velocity_guess: &mut [f64],
    dof_mgr: &crate::fem::DofManager,
    config: &PicardConfig,
    mut preconditioner_factory: Option<P>,
) -> (Vec<f64>, PicardStats)
where
    F: FnMut(&[f64]) -> (CsMat<f64>, Vec<f64>),
    S: Solver,
    P: FnMut(&CsMat<f64>) -> Box<dyn crate::linalg::preconditioner::Preconditioner>,
{
    let n_dofs = velocity_guess.len();
    let mut velocity_prev = velocity_guess.to_vec();
    let mut total_linear_iters = 0;
    let mut last_linear_stats = SolverStats::new();

    // Initialize Anderson accelerator if enabled
    let mut anderson = if config.use_anderson {
        Some(AndersonAccelerator::new(config.anderson_depth, config.anderson_beta)
            .with_verbose(false))
    } else {
        None
    };

    // Adaptive damping state
    let mut current_alpha = config.relaxation;
    let mut residual_history: Vec<f64> = Vec::new();

    for picard_iter in 0..config.max_iterations {
        // 1. Assemble K and f using current velocity
        let (k_mat, f) = assembler(&velocity_prev);

        // 2. Apply boundary conditions
        let (k_bc, f_bc) = crate::fem::Assembler::apply_dirichlet_bcs(&k_mat, &f, dof_mgr);

        // 3. Solve linear system
        let (velocity_new, lin_stats) = if let Some(factory) = &mut preconditioner_factory {
            let precond = factory(&k_bc);
            // Pass reference to Box, so P = Box<dyn Preconditioner> (which is Sized)
            linear_solver.solve_with_operator(&k_bc, &f_bc, &precond)
        } else {
            linear_solver.solve(&k_bc, &f_bc)
        };
        total_linear_iters += lin_stats.iterations;
        last_linear_stats = lin_stats;

        if !last_linear_stats.converged {
            // Linear solver failed - cannot continue Picard
            return (
                velocity_new,
                PicardStats {
                    iterations: picard_iter + 1,
                    converged: false,
                    relative_change: f64::INFINITY,
                    total_linear_iterations: total_linear_iters,
                    last_linear_stats,
                },
            );
        }

        // Check for NaN/Inf in solution (indicates numerical breakdown)
        if velocity_new.iter().any(|&v| !v.is_finite()) {
            eprintln!("    WARNING: Linear solver produced NaN/Inf values!");
            return (
                velocity_new,
                PicardStats {
                    iterations: picard_iter + 1,
                    converged: false,
                    relative_change: f64::INFINITY,
                    total_linear_iterations: total_linear_iters,
                    last_linear_stats,
                },
            );
        }

        // 4. Under-relaxation (with optional Anderson Acceleration and adaptive damping)
        let alpha = current_alpha;  // Use adaptive alpha if damping is enabled

        let velocity_relaxed = if let Some(ref mut aa) = anderson {
            // Anderson Acceleration:
            // Treat Picard as fixed-point: x^{k+1} = F(x^k)
            // where F includes both linear solve AND relaxation

            // Apply standard relaxation first (with adaptive alpha)
            let mut velocity_relaxed_std = vec![0.0; n_dofs];
            for i in 0..n_dofs {
                velocity_relaxed_std[i] = alpha * velocity_new[i]
                    + (1.0 - alpha) * velocity_prev[i];
            }

            // Use Anderson to accelerate the relaxed update
            // x^k = velocity_prev, F(x^k) = velocity_relaxed_std
            aa.accelerate(&velocity_prev, &velocity_relaxed_std)
        } else {
            // Standard relaxation (no Anderson)
            let mut velocity_relaxed = vec![0.0; n_dofs];
            for i in 0..n_dofs {
                velocity_relaxed[i] = alpha * velocity_new[i]
                    + (1.0 - alpha) * velocity_prev[i];
            }
            velocity_relaxed
        };

        // 5. Compute convergence criterion: ||Δu||/||u||
        let mut delta_norm_sq = 0.0;
        let mut u_norm_sq = 0.0;

        for i in 0..n_dofs {
            let delta = velocity_relaxed[i] - velocity_prev[i];
            delta_norm_sq += delta * delta;
            u_norm_sq += velocity_relaxed[i] * velocity_relaxed[i];
        }

        let delta_norm = delta_norm_sq.sqrt();
        let u_norm = u_norm_sq.sqrt();

        // Relative change (with absolute tolerance guard)
        let relative_change = if u_norm > config.abs_tolerance {
            delta_norm / u_norm
        } else {
            // Velocity very small - use absolute criterion
            delta_norm / config.abs_tolerance
        };

        // Adaptive Damping: Adjust relaxation based on convergence behavior
        if config.use_adaptive_damping && picard_iter > 0 {
            residual_history.push(relative_change);

            // Detect oscillation: residual increasing after decreasing
            if residual_history.len() >= 2 {
                let n = residual_history.len();
                let r_curr = residual_history[n - 1];
                let r_prev = residual_history[n - 2];

                // Check for multiple oscillation patterns
                let is_oscillating = if n >= 3 {
                    let r_prev2 = residual_history[n - 3];
                    // Pattern 1: Residual increased
                    let increased = r_curr > r_prev * 1.2;
                    // Pattern 2: Zig-zag pattern (residual went down then up)
                    let zigzag = r_prev < r_prev2 && r_curr > r_prev;
                    increased || zigzag
                } else {
                    // Simple increase check
                    r_curr > r_prev * 1.2
                };

                if is_oscillating {
                    let old_alpha = current_alpha;
                    current_alpha *= config.damping_reduction;
                    current_alpha = current_alpha.max(config.alpha_min);

                    if (old_alpha - current_alpha).abs() > 1e-10 {
                        println!("    [Adaptive] Oscillation detected, reducing α: {:.3} → {:.3}",
                                 old_alpha, current_alpha);
                    }
                } else if r_curr < r_prev * 0.9 {
                    // Converging well - slowly increase alpha
                    let old_alpha = current_alpha;
                    current_alpha *= config.damping_increase;
                    current_alpha = current_alpha.min(config.alpha_max);

                    if picard_iter % 5 == 0 && (current_alpha - old_alpha).abs() > 1e-10 {
                        println!("    [Adaptive] Good convergence, increasing α: {:.3} → {:.3}",
                                 old_alpha, current_alpha);
                    }
                }
            }
        }

        // 6. Check convergence
        let converged = relative_change < config.tolerance;

        // Print progress
        println!("    Picard iter {:2}: rel_change = {:.3e}, converged = {}",
                 picard_iter + 1, relative_change, converged);

        if converged {
            // Copy relaxed solution back to guess array
            velocity_guess.copy_from_slice(&velocity_relaxed);

            return (
                velocity_relaxed,
                PicardStats {
                    iterations: picard_iter + 1,
                    converged: true,
                    relative_change,
                    total_linear_iterations: total_linear_iters,
                    last_linear_stats,
                },
            );
        }

        // 7. Update for next iteration
        velocity_prev = velocity_relaxed;
    }

    // Max iterations reached without convergence
    velocity_guess.copy_from_slice(&velocity_prev);

    (
        velocity_prev,
        PicardStats {
            iterations: config.max_iterations,
            converged: false,
            relative_change: f64::INFINITY,
            total_linear_iterations: total_linear_iters,
            last_linear_stats,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImprovedMeshGenerator, DofManager, Assembler, ElastoViscoPlastic, ConjugateGradient};
    use nalgebra::{Point3, Vector3};
    use sprs::TriMat;

    #[test]
    fn test_picard_linear_problem() {
        // For linear elasticity, Picard should converge in 1 iteration
        let mut mesh = ImprovedMeshGenerator::generate_cube(2, 2, 2, 1.0, 1.0, 1.0);
        let n_nodes = mesh.num_nodes();
        let mut dof_mgr = DofManager::new(n_nodes, 3);

        // Fix bottom, pull top
        for (node_id, node) in mesh.geometry.nodes.iter().enumerate() {
            if node.z.abs() < 1e-6 {
                dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 0), 0.0);
                dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 1), 0.0);
                dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), 0.0);
            }
            if (node.z - 1.0).abs() < 1e-6 {
                dof_mgr.set_dirichlet(dof_mgr.global_dof(node_id, 2), 0.01);
            }
        }

        let material = crate::mechanics::IsotropicElasticity::new(100e9, 0.25);

        let n_dofs = dof_mgr.total_dofs();
        let mut velocity = vec![0.0; n_dofs];

        let assembler = |_v: &[f64]| -> (CsMat<f64>, Vec<f64>) {
            let k_mat = Assembler::assemble_elasticity_stiffness_parallel(&mesh, &dof_mgr, &material);
            let f = vec![0.0; n_dofs];
            (k_mat, f)
        };

        let mut linear_solver = ConjugateGradient::new()
            .with_max_iterations(1000)
            .with_tolerance(1e-10);

        let mut config = PicardConfig::default();
        config.relaxation = 1.0; // For linear problem, should converge in 1 iteration

        let (_v, stats) = picard_solve(
            assembler, 
            &mut linear_solver, 
            &mut velocity, 
            &dof_mgr, 
            &config,
            None::<fn(&CsMat<f64>) -> Box<dyn crate::linalg::preconditioner::Preconditioner>>
        );

        // Linear problem with relaxation 1.0 → should converge in 2 iterations
        // (1st to get close, 2nd to confirm Δu < tol)
        assert!(stats.converged, "Picard should converge for linear problem");
        assert!(
            stats.iterations <= 2,
            "Linear problem with alpha=1.0 should converge in <=2 iterations, got {}",
            stats.iterations
        );
    }

    #[test]
    fn test_picard_config() {
        let default_cfg = PicardConfig::default();
        assert_eq!(default_cfg.max_iterations, 20);
        assert_eq!(default_cfg.relaxation, 0.7);

        let cons_cfg = PicardConfig::conservative();
        assert_eq!(cons_cfg.relaxation, 0.5);

        let agg_cfg = PicardConfig::aggressive();
        assert_eq!(agg_cfg.relaxation, 0.8);
    }
}
