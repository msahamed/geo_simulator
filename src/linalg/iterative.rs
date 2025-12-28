use sprs::CsMat;
use std::time::Instant;
use super::solver::{Solver, SolverStats, SolverUtils};
use super::preconditioner::{Preconditioner, JacobiPreconditioner, IdentityPreconditioner};

/// Conjugate Gradient solver for symmetric positive definite systems
///
/// Solves Ax = b where A is SPD
/// Uses optional preconditioning for better convergence
pub struct ConjugateGradient {
    /// Maximum iterations
    max_iterations: usize,

    /// Convergence tolerance (relative residual)
    tolerance: f64,

    /// Whether to use Jacobi preconditioning
    use_preconditioner: bool,

    /// Solver name
    name: String,
}

impl ConjugateGradient {
    /// Create new CG solver with defaults
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            use_preconditioner: true,
            name: "Conjugate Gradient".to_string(),
        }
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Enable/disable preconditioning
    pub fn with_preconditioner(mut self, use_precond: bool) -> Self {
        self.use_preconditioner = use_precond;
        self
    }

    /// Preconditioned Conjugate Gradient algorithm
    #[allow(non_snake_case)]
    fn solve_with_precond<P: Preconditioner>(
        &self,
        A: &CsMat<f64>,
        b: &[f64],
        precond: &P,
    ) -> (Vec<f64>, SolverStats) {
        let n = b.len();
        let start = Instant::now();

        // Initial guess: x = 0
        let mut x = vec![0.0; n];

        // Initial residual: r = b - Ax = b
        let mut r = b.to_vec();

        // Check if already converged
        let b_norm = SolverUtils::norm(b);
        if b_norm < 1e-14 {
            let stats = SolverStats {
                iterations: 0,
                residual_norm: 0.0,
                relative_residual: 0.0,
                converged: true,
                solve_time: start.elapsed().as_secs_f64(),
            };
            return (x, stats);
        }

        // Apply preconditioner: z = M^{-1} r
        let mut z = precond.apply(&r);

        // p = z
        let mut p = z.clone();

        // rz = r^T z
        let mut rz: f64 = r.iter().zip(z.iter()).map(|(&ri, &zi)| ri * zi).sum();

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            // Ap = A * p (manually compute to get dense vector)
            let mut ap = vec![0.0; n];
            for (row_idx, row) in A.outer_iterator().enumerate() {
                let mut sum = 0.0;
                for (col_idx, &val) in row.iter() {
                    sum += val * p[col_idx];
                }
                ap[row_idx] = sum;
            }

            // alpha = (r^T z) / (p^T A p)
            let pap: f64 = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).sum();

            if pap.abs() < 1e-14 {
                break; // Avoid division by zero
            }

            let alpha = rz / pap;

            // x = x + alpha * p
            for i in 0..n {
                x[i] += alpha * p[i];
            }

            // r = r - alpha * Ap
            for i in 0..n {
                r[i] -= alpha * ap[i];
            }

            // Check convergence
            let r_norm = SolverUtils::norm(&r);
            let relative_res = r_norm / b_norm;

            if relative_res < self.tolerance {
                converged = true;
                iteration += 1;
                break;
            }

            // z = M^{-1} r
            z = precond.apply(&r);

            // beta = (r_new^T z_new) / (r_old^T z_old)
            let rz_new: f64 = r.iter().zip(z.iter()).map(|(&ri, &zi)| ri * zi).sum();
            let beta = rz_new / rz;
            rz = rz_new;

            // p = z + beta * p
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }

            iteration += 1;
        }

        let solve_time = start.elapsed().as_secs_f64();
        let residual_norm = SolverUtils::residual_norm(A, &x, b);
        let relative_residual = SolverUtils::relative_residual(A, &x, b);

        let stats = SolverStats {
            iterations: iteration,
            residual_norm,
            relative_residual,
            converged,
            solve_time,
        };

        (x, stats)
    }
}

impl Default for ConjugateGradient {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for ConjugateGradient {
    #[allow(non_snake_case)]
    fn solve(&mut self, A: &CsMat<f64>, b: &[f64]) -> (Vec<f64>, SolverStats) {
        if self.use_preconditioner {
            let precond = JacobiPreconditioner::new(A);
            self.solve_with_precond(A, b, &precond)
        } else {
            let precond = IdentityPreconditioner;
            self.solve_with_precond(A, b, &precond)
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;
    use approx::assert_relative_eq;

    #[test]
    fn test_cg_simple() {
        // Solve [2 1; 1 2] x = [3; 3]
        // Solution: x = [1; 1]
        let mut triplets = TriMat::new((2, 2));
        triplets.add_triplet(0, 0, 2.0);
        triplets.add_triplet(0, 1, 1.0);
        triplets.add_triplet(1, 0, 1.0);
        triplets.add_triplet(1, 1, 2.0);
        let A = triplets.to_csr();

        let b = vec![3.0, 3.0];

        let mut solver = ConjugateGradient::new();
        let (x, stats) = solver.solve(&A, &b);

        assert_relative_eq!(x[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-6);
        assert!(stats.converged);
        assert!(stats.iterations <= 10);
    }

    #[test]
    fn test_cg_diagonal() {
        // Diagonal matrix: should converge in 1 iteration with preconditioning
        let n = 10;
        let mut triplets = TriMat::new((n, n));
        for i in 0..n {
            triplets.add_triplet(i, i, (i + 1) as f64);
        }
        let A = triplets.to_csr();

        let b: Vec<f64> = (1..=n).map(|i| (i * i) as f64).collect();

        let mut solver = ConjugateGradient::new().with_preconditioner(true);
        let (x, stats) = solver.solve(&A, &b);

        // x[i] = b[i] / A[i][i] = (i+1)^2 / (i+1) = i+1
        for i in 0..n {
            assert_relative_eq!(x[i], (i + 1) as f64, epsilon = 1e-6);
        }
        assert!(stats.converged);
        assert!(stats.iterations <= 5);
    }

    #[test]
    fn test_cg_with_without_precond() {
        // Create a moderately ill-conditioned SPD matrix
        let n = 20;
        let mut triplets = TriMat::new((n, n));

        // Tridiagonal matrix
        for i in 0..n {
            triplets.add_triplet(i, i, 4.0);
            if i > 0 {
                triplets.add_triplet(i, i - 1, -1.0);
            }
            if i < n - 1 {
                triplets.add_triplet(i, i + 1, -1.0);
            }
        }
        let A = triplets.to_csr();

        let b = vec![1.0; n];

        // Solve without preconditioner
        let mut solver_no_precond = ConjugateGradient::new().with_preconditioner(false);
        let (_, stats_no_precond) = solver_no_precond.solve(&A, &b);

        // Solve with preconditioner
        let mut solver_precond = ConjugateGradient::new().with_preconditioner(true);
        let (_, stats_precond) = solver_precond.solve(&A, &b);

        // Both should converge
        assert!(stats_no_precond.converged);
        assert!(stats_precond.converged);

        // Preconditioned should converge faster or equal
        assert!(stats_precond.iterations <= stats_no_precond.iterations);
    }
}
