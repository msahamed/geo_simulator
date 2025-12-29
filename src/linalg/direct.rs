use sprs::CsMat;
use std::time::Instant;
use super::solver::{Solver, SolverStats, SolverUtils};

/// Direct sparse solver using LU decomposition
///
/// Uses sprs's built-in sparse LU factorization
/// Good for small to medium problems (<100k DOF)
pub struct DirectSolver {
    /// Solver name
    name: String,
}

impl DirectSolver {
    pub fn new() -> Self {
        Self {
            name: "Direct (Sparse LU)".to_string(),
        }
    }
}

impl Default for DirectSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver for DirectSolver {
    #[allow(non_snake_case)]
    fn solve(&mut self, A: &CsMat<f64>, b: &[f64]) -> (Vec<f64>, SolverStats) {
        let start = Instant::now();

        let n = b.len();

        // Convert sparse matrix to dense nalgebra matrix
        let mut a_dense = nalgebra::DMatrix::zeros(n, n);
        for (row_idx, row) in A.outer_iterator().enumerate() {
            for (col_idx, &val) in row.iter() {
                a_dense[(row_idx, col_idx)] = val;
            }
        }

        // Use nalgebra LU factorization
        let lu = a_dense.lu();

        // Convert b to DVector
        let b_vec = nalgebra::DVector::from_vec(b.to_vec());

        // Solve
        let x_vec = lu.solve(&b_vec).expect("LU solve failed");

        let x: Vec<f64> = x_vec.iter().copied().collect();

        let solve_time = start.elapsed().as_secs_f64();

        // Compute residual
        let residual_norm = SolverUtils::residual_norm(A, &x, b);
        let relative_residual = SolverUtils::relative_residual(A, &x, b);

        let stats = SolverStats {
            iterations: 0, // Direct solver doesn't iterate
            residual_norm,
            relative_residual,
            converged: relative_residual < 1e-8,
            solve_time,
        };

        (x, stats)
    }

    #[allow(non_snake_case)]
    fn solve_with_operator<O, P>(
        &self,
        _A: &O,
        _b: &[f64],
        _precond: &P,
    ) -> (Vec<f64>, SolverStats)
    where
        O: super::solver::LinearOperator,
        P: super::preconditioner::Preconditioner,
    {
        panic!("DirectSolver does not support matrix-free operator. Use iterative solvers instead.");
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
    fn test_direct_solver_simple() {
        // Solve [2 1; 1 2] x = [3; 3]
        // Solution: x = [1; 1]
        let mut triplets = TriMat::new((2, 2));
        triplets.add_triplet(0, 0, 2.0);
        triplets.add_triplet(0, 1, 1.0);
        triplets.add_triplet(1, 0, 1.0);
        triplets.add_triplet(1, 1, 2.0);
        let A = triplets.to_csr();

        let b = vec![3.0, 3.0];

        let mut solver = DirectSolver::new();
        let (x, stats) = solver.solve(&A, &b);

        assert_relative_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-10);
        assert!(stats.converged);
        assert!(stats.relative_residual < 1e-8);
    }

    #[test]
    fn test_direct_solver_diagonal() {
        // Diagonal matrix: easy to solve
        let n = 10;
        let mut triplets = TriMat::new((n, n));
        for i in 0..n {
            triplets.add_triplet(i, i, (i + 1) as f64);
        }
        let A = triplets.to_csr();

        let b: Vec<f64> = (1..=n).map(|i| (i * i) as f64).collect();

        let mut solver = DirectSolver::new();
        let (x, stats) = solver.solve(&A, &b);

        // x[i] = b[i] / A[i][i] = (i+1)^2 / (i+1) = i+1
        for i in 0..n {
            assert_relative_eq!(x[i], (i + 1) as f64, epsilon = 1e-10);
        }
        assert!(stats.converged);
    }
}
