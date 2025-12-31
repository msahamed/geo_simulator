use sprs::CsMat;
use std::time::Instant;
use super::solver::{Solver, SolverStats, SolverUtils};
use super::preconditioner::{Preconditioner, JacobiPreconditioner, IdentityPreconditioner};

/// Conjugate Gradient solver for symmetric positive definite systems
///
/// Solves Ax = b where A is SPD
/// Uses optional preconditioning for better convergence
pub struct ConjugateGradient {
    /// Maximum allowed iterations
    pub max_iterations: usize,
    /// Relative residual tolerance: ||r|| / ||b||
    pub tolerance: f64,
    /// Absolute residual tolerance: ||r||
    pub abs_tolerance: f64,
    /// Whether to use preconditioning
    pub use_preconditioner: bool,
    /// Solver name
    name: String,
}

impl ConjugateGradient {
    /// Create a new Conjugate Gradient solver with default settings
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            abs_tolerance: 1e-10,
            use_preconditioner: true,
            name: "Conjugate Gradient".to_string(),
        }
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set relative tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set absolute tolerance
    pub fn with_abs_tolerance(mut self, abs_tolerance: f64) -> Self {
        self.abs_tolerance = abs_tolerance;
        self
    }

    /// Enable/disable preconditioning
    pub fn with_preconditioner(mut self, use_precond: bool) -> Self {
        self.use_preconditioner = use_precond;
        self
    }

    /// Preconditioned Conjugate Gradient algorithm
    #[allow(non_snake_case)]
    pub fn solve_with_operator<O: crate::linalg::solver::LinearOperator, P: Preconditioner>(
        &self,
        a: &O,
        b: &[f64],
        precond: &P,
    ) -> (Vec<f64>, SolverStats) {
        let n = b.len();
        let start = Instant::now();

        // Initial guess: x = 0
        let mut x = vec![0.0; n];

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

        let inv_b_norm = 1.0 / b_norm;

        // Apply preconditioner: z = M^{-1} r
        // Initially r = b. We work with normalized r = b * inv_b_norm
        let r_init_norm: Vec<f64> = b.iter().map(|&bi| bi * inv_b_norm).collect();
        let mut r = r_init_norm.clone();
        let mut z = precond.apply(&r);

        // p = z
        let mut p = z.clone();

        // rz = r^T z
        let mut rz: f64 = r.iter().zip(z.iter()).map(|(&ri, &zi)| ri * zi).sum();

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            // ap = a * p
            let ap = a.apply(&p);

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

            // Check convergence (both relative AND absolute criteria)
            // Note: r is normalized, so r_norm is the relative residual
            let rel_res = SolverUtils::norm(&r);
            let r_norm_abs = rel_res * b_norm;

            if (rel_res < self.tolerance) || (r_norm_abs < self.abs_tolerance) {
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
        
        // Unscale x to the original system
        for i in 0..n {
            x[i] *= b_norm;
        }

        // Final residual norm calculation
        let residual_norm = SolverUtils::residual_norm(a, &x, b);
        let relative_residual = residual_norm / b_norm;

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

/// BiCGSTAB (Stabilized Bi-Conjugate Gradient) solver for non-symmetric systems
pub struct BiCGSTAB {
    max_iterations: usize,
    tolerance: f64,
    abs_tolerance: f64,
    use_preconditioner: bool,
    name: String,
}

impl BiCGSTAB {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            abs_tolerance: 1e6, // Safer default for geodynamic scales
            use_preconditioner: true,
            name: "BiCGSTAB".to_string(),
        }
    }

    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn with_abs_tolerance(mut self, abs_tol: f64) -> Self {
        self.abs_tolerance = abs_tol;
        self
    }

    pub fn with_preconditioner(mut self, use_precond: bool) -> Self {
        self.use_preconditioner = use_precond;
        self
    }

    #[allow(non_snake_case)]
    pub fn solve_with_operator<O: crate::linalg::solver::LinearOperator, P: Preconditioner>(
        &self,
        a: &O,
        b: &[f64],
        precond: &P,
    ) -> (Vec<f64>, SolverStats) {
        let n = b.len();
        let start = Instant::now();

        // Initial guess: x = 0
        let mut x = vec![0.0; n];

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

        let inv_b_norm = 1.0 / b_norm;
        let mut r: Vec<f64> = b.iter().map(|&bi| bi * inv_b_norm).collect();
        let r_hat = r.clone(); // Shadow residual
        
        let mut rho = 1.0;
        let mut alpha = 1.0;
        let mut omega = 1.0;
        let mut v = vec![0.0; n];
        let mut p = vec![0.0; n];

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            let rho_prev = rho;
            rho = r_hat.iter().zip(r.iter()).map(|(&rhi, &ri)| rhi * ri).sum();
            
            if rho.abs() < 1e-20 { 
                println!("    BiCGSTAB Stability Break: rho too small ({:.3e})", rho);
                break; 
            }

            if iteration == 0 {
                p = r.clone();
            } else {
                let beta = (rho / rho_prev) * (alpha / omega);
                for i in 0..n {
                    p[i] = r[i] + beta * (p[i] - omega * v[i]);
                }
            }

            // y = M^-1 p
            let y = precond.apply(&p);
            let _y_norm = SolverUtils::norm(&y);
            if iteration == 0 && self.max_iterations > 1 {
                 // println!("    DEBUG BiCGSTAB: p_norm = {:.1e}, y_norm = {:.1e}", SolverUtils::norm(&p), _y_norm);
            }
            
            // v = Ay 
            v = a.apply(&y);

            let r_hat_v: f64 = r_hat.iter().zip(v.iter()).map(|(&rhi, &vi)| rhi * vi).sum();
            if r_hat_v.abs() < 1e-60 || r_hat_v.is_nan() { 
                println!("    BiCGSTAB Stability Break: r_hat_v too small or NaN ({:.3e})", r_hat_v);
                break; 
            }
            alpha = rho / r_hat_v;

            // s = r - alpha * v
            let mut s = r.clone();
            for i in 0..n { s[i] -= alpha * v[i]; }
            
            let s_norm = SolverUtils::norm(&s);
            let s_norm_abs = s_norm * b_norm;
            if (s_norm < self.tolerance) || (s_norm_abs < self.abs_tolerance) {
                for i in 0..n { x[i] += alpha * y[i]; }
                converged = true;
                iteration += 1;
                break;
            }

            // z = M^-1 s
            let z = precond.apply(&s);
            
            // t = Az
            let t = a.apply(&z);
            
            let ts: f64 = t.iter().zip(s.iter()).map(|(&ti, &si)| ti * si).sum();
            let tt: f64 = t.iter().zip(t.iter()).map(|(&ti, &ti2)| ti * ti2).sum();
            
            if tt.abs() < 1e-60 || tt.is_nan() {
                println!("    BiCGSTAB Stability Break: tt too small or NaN ({:.3e})", tt);
                // Stability break - don't update x with omega
                for i in 0..n { x[i] += alpha * y[i]; }
                break;
            }
            omega = ts / tt;

            // x = x + alpha * y + omega * z
            for i in 0..n {
                x[i] += alpha * y[i] + omega * z[i];
            }

            // r = s - omega * t
            for i in 0..n {
                r[i] = s[i] - omega * t[i];
            }

            let rel_res = SolverUtils::norm(&r);
            let r_norm_abs = rel_res * b_norm;
            if iteration % 20 == 0 {
                println!("      BiCGSTAB iter {:4}: res = {:.3e}", iteration, r_norm_abs);
            }
            if (rel_res < self.tolerance) || (r_norm_abs < self.abs_tolerance) {
                converged = true;
                iteration += 1;
                break;
            }

            if omega.abs() < 1e-20 { break; }
            iteration += 1;
        }

        // Unscale x back to original system
        for i in 0..n {
            x[i] *= b_norm;
        }

        let stats = SolverStats {
            iterations: iteration,
            residual_norm: SolverUtils::residual_norm(a, &x, b),
            relative_residual: SolverUtils::relative_residual(a, &x, b),
            converged,
            solve_time: start.elapsed().as_secs_f64(),
        };

        (x, stats)
    }
}

impl Solver for BiCGSTAB {
    fn solve(&mut self, a: &CsMat<f64>, b: &[f64]) -> (Vec<f64>, SolverStats) {
        if self.use_preconditioner {
            let precond = JacobiPreconditioner::new(a);
            self.solve_with_operator(a, b, &precond)
        } else {
            let precond = IdentityPreconditioner;
            self.solve_with_operator(a, b, &precond)
        }
    }

    #[allow(non_snake_case)]
    fn solve_with_operator<O, P>(
        &self,
        A: &O,
        b: &[f64],
        precond: &P,
    ) -> (Vec<f64>, SolverStats)
    where
        O: crate::linalg::solver::LinearOperator,
        P: Preconditioner,
    {
        self.solve_with_operator(A, b, precond)
    }

    fn name(&self) -> &str { &self.name }
    fn abs_tolerance(&self) -> f64 { self.abs_tolerance }
    fn set_abs_tolerance(&mut self, tolerance: f64) { self.abs_tolerance = tolerance; }
    fn tolerance(&self) -> f64 { self.tolerance }
    fn set_tolerance(&mut self, tolerance: f64) { self.tolerance = tolerance; }
}

impl Default for BiCGSTAB {
    fn default() -> Self { Self::new() }
}

impl Default for ConjugateGradient {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for ConjugateGradient {
    #[allow(non_snake_case)]
    fn solve(&mut self, a: &CsMat<f64>, b: &[f64]) -> (Vec<f64>, SolverStats) {
        if self.use_preconditioner {
            let precond = JacobiPreconditioner::new(a);
            self.solve_with_operator(a, b, &precond)
        } else {
            let precond = IdentityPreconditioner;
            self.solve_with_operator(a, b, &precond)
        }
    }

    #[allow(non_snake_case)]
    fn solve_with_operator<O, P>(
        &self,
        A: &O,
        b: &[f64],
        precond: &P,
    ) -> (Vec<f64>, SolverStats)
    where
        O: crate::linalg::solver::LinearOperator,
        P: Preconditioner,
    {
        self.solve_with_operator(A, b, precond)
    }

    fn name(&self) -> &str {
        &self.name
    }
    fn abs_tolerance(&self) -> f64 { self.abs_tolerance }
    fn set_abs_tolerance(&mut self, tolerance: f64) { self.abs_tolerance = tolerance; }
    fn tolerance(&self) -> f64 { self.tolerance }
    fn set_tolerance(&mut self, tolerance: f64) { self.tolerance = tolerance; }
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
