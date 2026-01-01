use sprs::CsMat;
use std::time::Instant;
use super::solver::{Solver, SolverStats, SolverUtils, LinearOperator};
use super::preconditioner::{Preconditioner, JacobiPreconditioner, IdentityPreconditioner};

/// Conjugate Gradient solver for symmetric positive definite systems
pub struct ConjugateGradient {
    max_iterations: usize,
    tolerance: f64,
    abs_tolerance: f64,
    use_preconditioner: bool,
    name: String,
}

impl ConjugateGradient {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            abs_tolerance: 1e-12,
            use_preconditioner: true,
            name: "ConjugateGradient".to_string(),
        }
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_abs_tolerance(mut self, abs_tolerance: f64) -> Self {
        self.abs_tolerance = abs_tolerance;
        self
    }

    pub fn with_preconditioner(mut self, use_precond: bool) -> Self {
        self.use_preconditioner = use_precond;
        self
    }

    pub fn solve_with_operator_internal<O, P>(
        &self,
        a: &O,
        b: &[f64],
        precond: &P,
    ) -> (Vec<f64>, SolverStats)
    where
        O: LinearOperator,
        P: Preconditioner,
    {
        let n = b.len();
        let start = Instant::now();
        let b_norm = SolverUtils::norm(b);

        if b_norm < 1e-25 {
            return (vec![0.0; n], SolverStats {
                iterations: 0,
                residual_norm: 0.0,
                relative_residual: 0.0,
                converged: true,
                solve_time: start.elapsed().as_secs_f64(),
            });
        }

        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        
        let mut z = precond.apply(&r);
        let mut p = z.clone();
        let mut rz = r.iter().zip(z.iter()).map(|(&ri, &zi)| ri * zi).sum::<f64>();

        let mut iteration = 0;
        let mut converged = false;
        let mut final_res = b_norm;

        while iteration < self.max_iterations {
            let ap = a.apply(&p);
            let p_ap = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).sum::<f64>();

            if p_ap.abs() < 1e-30 { break; }
            let alpha = rz / p_ap;

            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            let r_norm = SolverUtils::norm(&r);
            final_res = r_norm;
            if r_norm < self.tolerance * b_norm || r_norm < self.abs_tolerance {
                converged = true;
                break;
            }

            z = precond.apply(&r);
            let rz_new = r.iter().zip(z.iter()).map(|(&ri, &zi)| ri * zi).sum::<f64>();
            let beta = rz_new / rz;
            rz = rz_new;

            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }

            iteration += 1;
        }

        (x, SolverStats {
            iterations: iteration,
            residual_norm: final_res,
            relative_residual: if b_norm > 1e-20 { final_res / b_norm } else { 0.0 },
            converged,
            solve_time: start.elapsed().as_secs_f64(),
        })
    }
}

impl Solver for ConjugateGradient {
    fn solve(&mut self, a: &CsMat<f64>, b: &[f64]) -> (Vec<f64>, SolverStats) {
        if self.use_preconditioner {
            let precond = JacobiPreconditioner::new(a);
            self.solve_with_operator_internal(a, b, &precond)
        } else {
            let precond = IdentityPreconditioner;
            self.solve_with_operator_internal(a, b, &precond)
        }
    }

    fn solve_with_operator<O: LinearOperator, P: Preconditioner>(
        &self,
        a: &O,
        b: &[f64],
        precond: &P,
    ) -> (Vec<f64>, SolverStats) {
        self.solve_with_operator_internal(a, b, precond)
    }

    fn name(&self) -> &str { &self.name }
    fn abs_tolerance(&self) -> f64 { self.abs_tolerance }
    fn set_abs_tolerance(&mut self, tolerance: f64) { self.abs_tolerance = tolerance; }
    fn tolerance(&self) -> f64 { self.tolerance }
    fn set_tolerance(&mut self, tolerance: f64) { self.tolerance = tolerance; }
}

/// BiCGSTAB (Biconjugate Gradient Stabilized) solver for non-symmetric systems
pub struct BiCGSTAB {
    max_iterations: usize,
    tolerance: f64,
    abs_tolerance: f64,
    use_preconditioner: bool,
    verbose: bool,
    name: String,
}

impl BiCGSTAB {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            abs_tolerance: 1e-12,
            use_preconditioner: true,
            verbose: false,
            name: "BiCGSTAB".to_string(),
        }
    }

    pub fn with_max_iterations(mut self, max_iter: usize) -> Self { self.max_iterations = max_iter; self }
    pub fn with_tolerance(mut self, tol: f64) -> Self { self.tolerance = tol; self }
    pub fn with_abs_tolerance(mut self, abs_tol: f64) -> Self { self.abs_tolerance = abs_tol; self }
    pub fn with_verbose(mut self, verbose: bool) -> Self { self.verbose = verbose; self }
    pub fn with_preconditioner(mut self, use_precond: bool) -> Self { self.use_preconditioner = use_precond; self }

    pub fn solve_with_operator_internal<O, P>(
        &self,
        a: &O,
        b: &[f64],
        precond: &P,
    ) -> (Vec<f64>, SolverStats)
    where
        O: LinearOperator,
        P: Preconditioner,
    {
        let n = b.len();
        let start = Instant::now();
        let b_norm = SolverUtils::norm(b);

        if b_norm < 1e-25 {
            return (vec![0.0; n], SolverStats { iterations: 0, residual_norm: 0.0, relative_residual: 0.0, converged: true, solve_time: start.elapsed().as_secs_f64() });
        }

        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        let r_hat = r.clone();
        
        let mut rho = 1.0;
        let mut alpha = 1.0;
        let mut omega = 1.0;
        let mut v = vec![0.0; n];
        let mut p = vec![0.0; n];

        let mut total_iter = 0;
        let mut converged = false;
        let mut final_res = b_norm;

        while total_iter < self.max_iterations {
            let rho_prev = rho;
            rho = r_hat.iter().zip(r.iter()).map(|(&ri_h, &ri)| ri_h * ri).sum::<f64>();
            
            if rho.abs() < 1e-40 { break; }

            if total_iter == 0 {
                p = r.clone();
            } else {
                let beta = (rho / rho_prev) * (alpha / omega);
                for i in 0..n {
                    p[i] = r[i] + beta * (p[i] - omega * v[i]);
                }
            }

            let p_hat = precond.apply(&p);
            v = a.apply(&p_hat);

            let rhat_v = r_hat.iter().zip(v.iter()).map(|(&ri_h, &vi)| ri_h * vi).sum::<f64>();
            if rhat_v.abs() < 1e-40 { break; }
            alpha = rho / rhat_v;

            let mut s = vec![0.0; n];
            for i in 0..n { s[i] = r[i] - alpha * v[i]; }

            let s_norm = SolverUtils::norm(&s);
            if s_norm < self.tolerance * b_norm || s_norm < self.abs_tolerance {
                for i in 0..n { x[i] += alpha * p_hat[i]; }
                final_res = s_norm;
                converged = true;
                break;
            }

            let s_hat = precond.apply(&s);
            let t = a.apply(&s_hat);

            let t_t = t.iter().map(|&ti| ti * ti).sum::<f64>();
            let t_s = t.iter().zip(s.iter()).map(|(&ti, &si)| ti * si).sum::<f64>();
            
            if t_t.abs() < 1e-40 { break; }
            omega = t_s / t_t;

            for i in 0..n {
                x[i] += alpha * p_hat[i] + omega * s_hat[i];
                r[i] = s[i] - omega * t[i];
            }

            final_res = SolverUtils::norm(&r);
            total_iter += 1;

            if self.verbose && total_iter % 50 == 0 {
                println!("        BiCGSTAB iter {:4}: res = {:.3e}, rel = {:.3e}", total_iter, final_res, final_res / b_norm);
            }

            if final_res < self.tolerance * b_norm || final_res < self.abs_tolerance {
                converged = true;
                break;
            }
            if omega.abs() < 1e-40 { break; }
        }

        (x, SolverStats {
            iterations: total_iter,
            residual_norm: final_res,
            relative_residual: final_res / b_norm,
            converged,
            solve_time: start.elapsed().as_secs_f64(),
        })
    }
}

impl Solver for BiCGSTAB {
    fn solve(&mut self, a: &CsMat<f64>, b: &[f64]) -> (Vec<f64>, SolverStats) {
        if self.use_preconditioner {
            let precond = JacobiPreconditioner::new(a);
            self.solve_with_operator_internal(a, b, &precond)
        } else {
            let precond = IdentityPreconditioner;
            self.solve_with_operator_internal(a, b, &precond)
        }
    }

    fn solve_with_operator<O: LinearOperator, P: Preconditioner>(&self, a: &O, b: &[f64], precond: &P) -> (Vec<f64>, SolverStats) {
        self.solve_with_operator_internal(a, b, precond)
    }

    fn name(&self) -> &str { &self.name }
    fn abs_tolerance(&self) -> f64 { self.abs_tolerance }
    fn set_abs_tolerance(&mut self, tolerance: f64) { self.abs_tolerance = tolerance; }
    fn tolerance(&self) -> f64 { self.tolerance }
    fn set_tolerance(&mut self, tolerance: f64) { self.tolerance = tolerance; }
}

/// GMRES (Generalized Minimal Residual) solver
pub struct GMRES {
    max_iterations: usize,
    restart: usize,
    tolerance: f64,
    abs_tolerance: f64,
    use_preconditioner: bool,
    verbose: bool,
    name: String,
}

impl GMRES {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            restart: 30,
            tolerance: 1e-8,
            abs_tolerance: 1e-12,
            use_preconditioner: true,
            verbose: false,
            name: "GMRES".to_string(),
        }
    }

    pub fn with_restart(mut self, m: usize) -> Self { self.restart = m; self }
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self { self.max_iterations = max_iter; self }
    pub fn with_tolerance(mut self, tol: f64) -> Self { self.tolerance = tol; self }
    pub fn with_abs_tolerance(mut self, abs_tol: f64) -> Self { self.abs_tolerance = abs_tol; self }
    pub fn with_verbose(mut self, verbose: bool) -> Self { self.verbose = verbose; self }
    pub fn with_preconditioner(mut self, use_precond: bool) -> Self { self.use_preconditioner = use_precond; self }

    pub fn solve_with_operator_internal<O, P>(
        &self,
        a: &O,
        b: &[f64],
        precond: &P,
    ) -> (Vec<f64>, SolverStats)
    where
        O: LinearOperator,
        P: Preconditioner,
    {
        let n = b.len();
        let start = Instant::now();
        let b_norm = SolverUtils::norm(b);

        if b_norm < 1e-25 {
            return (vec![0.0; n], SolverStats { iterations: 0, residual_norm: 0.0, relative_residual: 0.0, converged: true, solve_time: start.elapsed().as_secs_f64() });
        }

        let mut x = vec![0.0; n];
        let mut total_iter = 0;
        let mut converged = false;
        let mut final_res = b_norm;

        while total_iter < self.max_iterations {
            let r_vec = SolverUtils::compute_residual(a, &x, b);
            let r_norm = SolverUtils::norm(&r_vec);
            final_res = r_norm;

            if r_norm < self.tolerance * b_norm || r_norm < self.abs_tolerance {
                converged = true;
                break;
            }

            if self.verbose && total_iter % 50 == 0 {
                println!("        GMRES iter {:4}: res = {:.3e}, rel = {:.3e}", total_iter, r_norm, r_norm / b_norm);
            }

            let m = self.restart;
            let mut v = vec![vec![0.0; n]; m + 1];
            let mut h = vec![vec![0.0; m]; m + 1];
            for i in 0..n { v[0][i] = r_vec[i] / r_norm; }

            let mut g = vec![0.0; m + 1];
            g[0] = r_norm;

            let mut cs = vec![0.0; m];
            let mut sn = vec![0.0; m];

            let mut k = 0;
            for j in 0..m {
                if total_iter >= self.max_iterations { break; }
                
                let w = a.apply(&precond.apply(&v[j]));

                let mut w_ortho = w;
                for i in 0..=j {
                    h[i][j] = v[i].iter().zip(w_ortho.iter()).map(|(&ai, &bi)| ai * bi).sum::<f64>();
                    for l in 0..n { w_ortho[l] -= h[i][j] * v[i][l]; }
                }
                h[j + 1][j] = SolverUtils::norm(&w_ortho);

                if h[j + 1][j].abs() > 1e-40 {
                    for l in 0..n { v[j + 1][l] = w_ortho[l] / h[j + 1][j]; }
                }

                for i in 0..j {
                    let temp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                    h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                    h[i][j] = temp;
                }

                let (c, s, rho) = self.givens_rotation(h[j][j], h[j + 1][j]);
                cs[j] = c; sn[j] = s; h[j][j] = rho; h[j + 1][j] = 0.0;

                let temp = cs[j] * g[j];
                g[j + 1] = -sn[j] * g[j];
                g[j] = temp;

                k = j + 1;
                total_iter += 1;
                final_res = g[k].abs();

                if final_res < self.tolerance * b_norm || final_res < self.abs_tolerance { break; }
            }

            let mut y = vec![0.0; k];
            for i in (0..k).rev() {
                let mut sum = 0.0;
                for j in (i + 1)..k { sum += h[i][j] * y[j]; }
                if h[i][i].abs() < 1e-40 { break; }
                y[i] = (g[i] - sum) / h[i][i];
            }

            let mut dy = vec![0.0; n];
            for j in 0..k { for i in 0..n { dy[i] += v[j][i] * y[j]; } }
            let precond_dy = precond.apply(&dy);
            for i in 0..n { x[i] += precond_dy[i]; }

            if final_res < self.tolerance * b_norm || final_res < self.abs_tolerance {
                converged = true;
                break;
            }
        }

        (x, SolverStats {
            iterations: total_iter,
            residual_norm: final_res,
            relative_residual: final_res / b_norm,
            converged,
            solve_time: start.elapsed().as_secs_f64(),
        })
    }

    fn givens_rotation(&self, a: f64, b: f64) -> (f64, f64, f64) {
        if b.abs() < 1e-40 { (1.0, 0.0, a) }
        else if b.abs() > a.abs() {
            let tau = a / b; let s = 1.0 / (1.0 + tau * tau).sqrt(); let c = s * tau;
            (c, s, b * (1.0 + tau * tau).sqrt())
        } else {
            let tau = b / a; let c = 1.0 / (1.0 + tau * tau).sqrt(); let s = c * tau;
            (c, s, a * (1.0 + tau * tau).sqrt())
        }
    }
}

impl Solver for GMRES {
    fn solve(&mut self, a: &CsMat<f64>, b: &[f64]) -> (Vec<f64>, SolverStats) {
        if self.use_preconditioner {
            let precond = JacobiPreconditioner::new(a);
            self.solve_with_operator_internal(a, b, &precond)
        } else {
            let precond = IdentityPreconditioner;
            self.solve_with_operator_internal(a, b, &precond)
        }
    }

    fn solve_with_operator<O: LinearOperator, P: Preconditioner>(&self, a: &O, b: &[f64], precond: &P) -> (Vec<f64>, SolverStats) {
        self.solve_with_operator_internal(a, b, precond)
    }

    fn name(&self) -> &str { &self.name }
    fn abs_tolerance(&self) -> f64 { self.abs_tolerance }
    fn set_abs_tolerance(&mut self, tolerance: f64) { self.abs_tolerance = tolerance; }
    fn tolerance(&self) -> f64 { self.tolerance }
    fn set_tolerance(&mut self, tolerance: f64) { self.tolerance = tolerance; }
}

impl Default for BiCGSTAB { fn default() -> Self { Self::new() } }
impl Default for ConjugateGradient { fn default() -> Self { Self::new() } }
impl Default for GMRES { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;
    use approx::assert_relative_eq;

    #[test]
    fn test_cg_basic() {
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
    }
}
