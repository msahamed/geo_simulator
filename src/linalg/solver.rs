use sprs::CsMat;

/// Statistics from solver execution
#[derive(Debug, Clone)]
pub struct SolverStats {
    /// Number of iterations (0 for direct solvers)
    pub iterations: usize,

    /// Final residual norm ||r|| = ||b - Ax||
    pub residual_norm: f64,

    /// Relative residual ||r|| / ||b||
    pub relative_residual: f64,

    /// Whether solver converged
    pub converged: bool,

    /// Solve time in seconds
    pub solve_time: f64,
}

impl SolverStats {
    pub fn new() -> Self {
        Self {
            iterations: 0,
            residual_norm: 0.0,
            relative_residual: 0.0,
            converged: false,
            solve_time: 0.0,
        }
    }
}

impl Default for SolverStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Linear system solver trait
///
/// Solves Ax = b for x
pub trait Solver {
    /// Solve the linear system Ax = b
    ///
    /// # Arguments
    /// * `A` - System matrix (n x n)
    /// * `b` - Right-hand side vector (n)
    ///
    /// # Returns
    /// * Solution vector x (n)
    /// * Solver statistics
    #[allow(non_snake_case)]
    fn solve(&mut self, A: &CsMat<f64>, b: &[f64]) -> (Vec<f64>, SolverStats);

    /// Get solver name
    fn name(&self) -> &str;
}

/// Helper functions for solver validation
pub struct SolverUtils;

impl SolverUtils {
    /// Compute residual r = b - Ax
    #[allow(non_snake_case)]
    pub fn compute_residual(A: &CsMat<f64>, x: &[f64], b: &[f64]) -> Vec<f64> {
        // Manually compute matrix-vector product to get dense result
        let n = b.len();
        let mut ax = vec![0.0; n];

        for (row_idx, row) in A.outer_iterator().enumerate() {
            let mut sum = 0.0;
            for (col_idx, &val) in row.iter() {
                sum += val * x[col_idx];
            }
            ax[row_idx] = sum;
        }

        b.iter()
            .zip(ax.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect()
    }

    /// Compute L2 norm of a vector
    pub fn norm(v: &[f64]) -> f64 {
        v.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Compute residual norm ||b - Ax||
    #[allow(non_snake_case)]
    pub fn residual_norm(A: &CsMat<f64>, x: &[f64], b: &[f64]) -> f64 {
        let r = Self::compute_residual(A, x, b);
        Self::norm(&r)
    }

    /// Compute relative residual ||b - Ax|| / ||b||
    #[allow(non_snake_case)]
    pub fn relative_residual(A: &CsMat<f64>, x: &[f64], b: &[f64]) -> f64 {
        let r_norm = Self::residual_norm(A, x, b);
        let b_norm = Self::norm(b);

        if b_norm < 1e-14 {
            r_norm
        } else {
            r_norm / b_norm
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;
    use approx::assert_relative_eq;

    #[test]
    fn test_norm() {
        let v = vec![3.0, 4.0];
        let norm = SolverUtils::norm(&v);
        assert_relative_eq!(norm, 5.0, epsilon = 1e-14);
    }

    #[test]
    fn test_residual() {
        // Simple 2x2 system: [2 1; 1 2] x = [3; 3]
        // Solution: x = [1; 1]
        let mut triplets = TriMat::new((2, 2));
        triplets.add_triplet(0, 0, 2.0);
        triplets.add_triplet(0, 1, 1.0);
        triplets.add_triplet(1, 0, 1.0);
        triplets.add_triplet(1, 1, 2.0);
        let A = triplets.to_csr();

        let x = vec![1.0, 1.0];
        let b = vec![3.0, 3.0];

        let r_norm = SolverUtils::residual_norm(&A, &x, &b);
        assert_relative_eq!(r_norm, 0.0, epsilon = 1e-14);
    }
}
