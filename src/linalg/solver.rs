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

/// Trait for a linear operator A that can be applied to a vector x to get Ax
pub trait LinearOperator {
    /// Apply the operator to vector v: out = A * v
    fn apply(&self, v: &[f64]) -> Vec<f64>;

    /// Apply the operator to vector v and add to out: out += A * v
    /// Default implementation uses apply() for convenience, but can be optimized
    fn apply_add(&self, v: &[f64], out: &mut [f64]) {
        let result = self.apply(v);
        for (i, &val) in result.iter().enumerate() {
            out[i] += val;
        }
    }

    /// Number of rows (output dimension)
    fn rows(&self) -> usize;

    /// Number of columns (input dimension)
    fn cols(&self) -> usize;
}

impl LinearOperator for CsMat<f64> {
    fn apply(&self, v: &[f64]) -> Vec<f64> {
        let n = self.rows();
        let mut result = vec![0.0; n];
        for (row_idx, row) in self.outer_iterator().enumerate() {
            let mut sum = 0.0;
            for (col_idx, &val) in row.iter() {
                sum += val * v[col_idx];
            }
            result[row_idx] = sum;
        }
        result
    }

    fn apply_add(&self, v: &[f64], out: &mut [f64]) {
        for (row_idx, row) in self.outer_iterator().enumerate() {
            let mut sum = 0.0;
            for (col_idx, &val) in row.iter() {
                sum += val * v[col_idx];
            }
            out[row_idx] += sum;
        }
    }

    fn rows(&self) -> usize {
        self.rows()
    }

    fn cols(&self) -> usize {
        self.cols()
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

    /// Solve the linear system using a generic operator
    #[allow(non_snake_case)]
    fn solve_with_operator<O, P>(
        &self,
        A: &O,
        b: &[f64],
        precond: &P,
    ) -> (Vec<f64>, SolverStats)
    where
        O: LinearOperator,
        P: crate::linalg::preconditioner::Preconditioner;

    /// Get solver name
    fn name(&self) -> &str;

    /// Get absolute tolerance
    fn abs_tolerance(&self) -> f64;

    /// Set absolute tolerance
    fn set_abs_tolerance(&mut self, tolerance: f64);

    /// Get relative tolerance
    fn tolerance(&self) -> f64;

    /// Set relative tolerance
    fn set_tolerance(&mut self, tolerance: f64);
}

/// Helper functions for solver validation
pub struct SolverUtils;

impl SolverUtils {
    /// Compute residual r = b - Ax
    #[allow(non_snake_case)]
    pub fn compute_residual<O: LinearOperator>(A: &O, x: &[f64], b: &[f64]) -> Vec<f64> {
        let ax = A.apply(x);
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
    pub fn residual_norm<O: LinearOperator>(A: &O, x: &[f64], b: &[f64]) -> f64 {
        let r = Self::compute_residual(A, x, b);
        Self::norm(&r)
    }

    /// Compute relative residual ||b - Ax|| / ||b||
    #[allow(non_snake_case)]
    pub fn relative_residual<O: LinearOperator>(A: &O, x: &[f64], b: &[f64]) -> f64 {
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
