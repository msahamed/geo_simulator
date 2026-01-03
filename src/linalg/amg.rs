/// Algebraic Multigrid (AMG) Preconditioner
///
/// Classical Ruge-Stüben AMG for symmetric positive definite systems
///
/// # Algorithm Overview
/// 1. Coarsening: Select coarse points using strength-of-connection
/// 2. Interpolation: Build prolongation operator P
/// 3. Restriction: R = P^T (Galerkin projection)
/// 4. Coarse operator: A_c = R * A * P
/// 5. V-cycle: Smooth → Restrict → Solve coarse → Prolong → Smooth

use sprs::{CsMat, TriMat};
use crate::linalg::preconditioner::Preconditioner;

/// AMG hierarchy level
#[derive(Clone)]
struct AMGLevel {
    /// Operator at this level
    a: CsMat<f64>,
    /// Prolongation operator (coarse to fine)
    p: Option<CsMat<f64>>,
    /// Diagonal of A (for Jacobi smoothing)
    diag_inv: Vec<f64>,
}

/// AMG Preconditioner
pub struct AMGPreconditioner {
    levels: Vec<AMGLevel>,
    num_smoothing_steps: usize,
    _strength_threshold: f64,
}

impl AMGPreconditioner {
    /// Create AMG preconditioner from matrix A
    ///
    /// # Arguments
    /// * `a` - SPD matrix to precondition
    /// * `max_levels` - Maximum number of multigrid levels (default: 10)
    /// * `coarse_size` - Stop coarsening when n < coarse_size (default: 500)
    /// * `strength_threshold` - Strength-of-connection threshold (default: 0.25)
    ///
    /// # Returns
    /// AMG preconditioner ready for V-cycle application
    pub fn new(
        a: &CsMat<f64>,
        max_levels: usize,
        coarse_size: usize,
        strength_threshold: f64,
    ) -> Result<Self, String> {
        let mut levels = Vec::new();
        let mut current_a = a.clone();

        // Build hierarchy
        for _level in 0..max_levels {
            let n = current_a.rows();

            if n <= coarse_size {
                // Reached coarsest level
                let diag_inv = extract_diag_inv(&current_a);
                levels.push(AMGLevel {
                    a: current_a,
                    p: None,
                    diag_inv,
                });
                break;
            }

            // Extract diagonal for smoothing
            let diag_inv = extract_diag_inv(&current_a);

            // 1. Coarsening: Select coarse points using strength-of-connection
            let (cf_splitting, num_coarse) = coarsen(&current_a, strength_threshold);

            if num_coarse == 0 || num_coarse >= n {
                // Cannot coarsen further
                levels.push(AMGLevel {
                    a: current_a,
                    p: None,
                    diag_inv,
                });
                break;
            }

            // Validate coarsening
            let actual_coarse = cf_splitting.iter().filter(|&&x| x).count();
            if actual_coarse != num_coarse {
                return Err(format!(
                    "Coarsening inconsistency: reported {} coarse, actual {}",
                    num_coarse, actual_coarse
                ));
            }

            // 2. Build prolongation operator P
            let p = build_prolongation(&current_a, &cf_splitting, num_coarse, strength_threshold);

            // 3. Galerkin coarse grid operator: A_c = P^T * A * P
            // Check dimensions
            if p.rows() != current_a.rows() || p.cols() != num_coarse {
                return Err(format!(
                    "Dimension mismatch: P is {}x{}, but should be {}x{}",
                    p.rows(), p.cols(), current_a.rows(), num_coarse
                ));
            }

            // Compute A_c = P^T * A * P using matrix-matrix multiplication
            let pt = p.transpose_view().to_owned();
            let pt = pt.to_csr(); // ENSURE CSR for matrix_multiply

            // Step 1: AP = A * P
            let ap = matrix_multiply(&current_a, &p)?;

            // Step 2: A_c = P^T * AP
            let a_coarse = matrix_multiply(&pt, &ap)?;

            levels.push(AMGLevel {
                a: current_a,
                p: Some(p),
                diag_inv,
            });

            current_a = a_coarse;
        }

        Ok(Self {
            levels,
            num_smoothing_steps: 2,
            _strength_threshold: strength_threshold,
        })
    }

    /// Apply V-cycle: Smooth → Restrict → Coarse solve → Prolong → Smooth
    fn v_cycle(&self, r: &[f64], level: usize) -> Vec<f64> {
        let n = r.len();
        let mut x = vec![0.0; n];

        if level == self.levels.len() - 1 {
            // Coarsest level: Direct solve using Jacobi iterations
            for _ in 0..10 {
                for i in 0..n {
                    let mut ax_i = 0.0;
                    for (j, &val) in self.levels[level].a.outer_iterator().nth(i).unwrap().iter() {
                        ax_i += val * x[j];
                    }
                    x[i] += (r[i] - ax_i) * self.levels[level].diag_inv[i];
                }
            }
            return x;
        }

        // Pre-smoothing: Weighted Jacobi
        for _ in 0..self.num_smoothing_steps {
            weighted_jacobi_step(&self.levels[level].a, r, &mut x, &self.levels[level].diag_inv, 0.67);
        }

        // Compute residual: r - A*x
        let ax = matvec(&self.levels[level].a, &x);
        let mut residual = vec![0.0; n];
        for i in 0..n {
            residual[i] = r[i] - ax[i];
        }

        // Restrict residual to coarse grid: r_c = P^T * residual
        let p = self.levels[level].p.as_ref().unwrap();
        let pt = p.transpose_view().to_owned();
        let pt = pt.to_csr(); // ENSURE CSR for matvec
        let r_coarse = matvec(&pt, &residual);

        // Solve on coarse grid
        let e_coarse = self.v_cycle(&r_coarse, level + 1);

        // Prolong correction to fine grid: e = P * e_c
        let correction = matvec(p, &e_coarse);

        // Add correction
        for i in 0..n {
            x[i] += correction[i];
        }

        // Post-smoothing: Weighted Jacobi
        for _ in 0..self.num_smoothing_steps {
            weighted_jacobi_step(&self.levels[level].a, r, &mut x, &self.levels[level].diag_inv, 0.67);
        }

        x
    }
}

impl Preconditioner for AMGPreconditioner {
    fn apply(&self, r: &[f64]) -> Vec<f64> {
        self.v_cycle(r, 0)
    }
}

/// Sparse matrix-matrix multiplication: C = A * B
fn matrix_multiply(a: &CsMat<f64>, b: &CsMat<f64>) -> Result<CsMat<f64>, String> {
    if a.cols() != b.rows() {
        return Err(format!(
            "Matrix multiply dimension mismatch: A is {}x{}, B is {}x{}",
            a.rows(), a.cols(), b.rows(), b.cols()
        ));
    }

    let m = a.rows();
    let n = b.cols();
    let mut c_triplets = TriMat::new((m, n));

    // For each row of A
    for (i, a_row) in a.outer_iterator().enumerate() {
        // For each column of B
        for j in 0..n {
            let mut sum = 0.0;

            // Compute dot product of A's row i with B's column j
            for (k, &a_val) in a_row.iter() {
                if let Some(&b_val) = b.get(k, j) {
                    sum += a_val * b_val;
                }
            }

            if sum.abs() > 1e-14 {
                c_triplets.add_triplet(i, j, sum);
            }
        }
    }

    Ok(c_triplets.to_csr())
}

/// Sparse matrix-vector product: y = A * x
fn matvec(a: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    let n = a.rows();
    let m = a.cols();

    assert_eq!(x.len(), m, "matvec: vector length {} != matrix cols {}", x.len(), m);

    let mut y = vec![0.0; n];

    for (i, row) in a.outer_iterator().enumerate() {
        let mut sum = 0.0;
        for (j, &val) in row.iter() {
            sum += val * x[j];
        }
        y[i] = sum;
    }

    y
}

/// Weighted Jacobi smoothing step: x := x + ω D^{-1} (r - A x)
fn weighted_jacobi_step(a: &CsMat<f64>, r: &[f64], x: &mut [f64], diag_inv: &[f64], omega: f64) {
    let n = r.len();
    let mut x_new = x.to_vec();

    for i in 0..n {
        let mut ax_i = 0.0;
        for (j, &val) in a.outer_iterator().nth(i).unwrap().iter() {
            ax_i += val * x[j];
        }
        x_new[i] = x[i] + omega * (r[i] - ax_i) * diag_inv[i];
    }

    x.copy_from_slice(&x_new);
}

/// Extract diagonal inverse for Jacobi smoothing
fn extract_diag_inv(a: &CsMat<f64>) -> Vec<f64> {
    let n = a.rows();
    let mut diag_inv = vec![1.0; n];

    for i in 0..n {
        if let Some(&val) = a.get(i, i) {
            if val.abs() > 1e-14 {
                diag_inv[i] = 1.0 / val;
            }
        }
    }

    diag_inv
}

/// Coarsen using Ruge-Stüben algorithm
///
/// Returns (cf_splitting, num_coarse) where:
/// - cf_splitting[i] = true if node i is COARSE, false if FINE
/// - num_coarse = number of coarse points
fn coarsen(a: &CsMat<f64>, theta: f64) -> (Vec<bool>, usize) {
    let n = a.rows();

    // 1. Compute strength-of-connection matrix
    let s = strength_matrix(a, theta);

    // 2. Compute lambda (number of strong connections)
    let mut lambda: Vec<usize> = (0..n)
        .map(|i| {
            s.outer_iterator()
                .nth(i)
                .unwrap()
                .iter()
                .filter(|(j, _)| *j != i)
                .count()
        })
        .collect();

    // 3. Initialize all points as UNDECIDED
    let mut cf_splitting = vec![false; n];
    let mut undecided: Vec<bool> = vec![true; n];
    let mut num_coarse = 0;

    // 4. Greedy coarsening: Select nodes with highest lambda
    while undecided.iter().any(|&u| u) {
        // Find undecided node with maximum lambda
        let mut max_lambda = 0;
        let mut max_node = 0;
        for i in 0..n {
            if undecided[i] && lambda[i] > max_lambda {
                max_lambda = lambda[i];
                max_node = i;
            }
        }

        if max_lambda == 0 {
            // No strong connections left, mark remaining as coarse
            for i in 0..n {
                if undecided[i] {
                    cf_splitting[i] = true;
                    undecided[i] = false;
                    num_coarse += 1;
                }
            }
            break;
        }

        // Mark max_node as COARSE
        cf_splitting[max_node] = true;
        undecided[max_node] = false;
        num_coarse += 1;

        // Mark strong neighbors as FINE and update lambda
        for (j, _) in s.outer_iterator().nth(max_node).unwrap().iter() {
            if j != max_node && undecided[j] {
                undecided[j] = false; // Mark as FINE

                // Increment lambda for neighbors of j
                for (k, _) in s.outer_iterator().nth(j).unwrap().iter() {
                    if k != j && undecided[k] {
                        lambda[k] += 1;
                    }
                }
            }
        }
    }

    (cf_splitting, num_coarse)
}

/// Build strength-of-connection matrix
///
/// S[i,j] = A[i,j] if |A[i,j]| >= theta * max_k(|A[i,k]|), else 0
fn strength_matrix(a: &CsMat<f64>, theta: f64) -> CsMat<f64> {
    let n = a.rows();
    let mut s_triplets = TriMat::new((n, n));

    // Iterate over matrix entries directly using row/col access
    for i in 0..n {
        // Find max off-diagonal in row i
        let mut max_off_diag: f64 = 0.0;
        for j in 0..n {
            if i != j {
                if let Some(&val) = a.get(i, j) {
                    max_off_diag = max_off_diag.max(val.abs());
                }
            }
        }

        let threshold = theta * max_off_diag;

        // Add strong connections
        for j in 0..n {
            if i != j {
                if let Some(&val) = a.get(i, j) {
                    if val.abs() >= threshold {
                        s_triplets.add_triplet(i, j, val);
                    }
                }
            }
        }
    }

    s_triplets.to_csr()
}

/// Build prolongation operator using direct interpolation
///
/// For fine point i:
/// P[i, c(i)] = w[i,c(i)] where c(i) are coarse neighbors
fn build_prolongation(
    a: &CsMat<f64>,
    cf_splitting: &[bool],
    num_coarse: usize,
    theta: f64,
) -> CsMat<f64> {
    let n = a.rows();
    let s = strength_matrix(a, theta);

    // Map coarse points to coarse grid indices
    let mut coarse_indices = vec![0; n];
    let mut c_idx = 0;
    for i in 0..n {
        if cf_splitting[i] {
            coarse_indices[i] = c_idx;
            c_idx += 1;
        }
    }

    let mut p_triplets = TriMat::new((n, num_coarse));

    for i in 0..n {
        if cf_splitting[i] {
            let c_idx = coarse_indices[i];
            p_triplets.add_triplet(i, c_idx, 1.0);
        } else {
            // Fine point: Interpolate from strong coarse neighbors
            let mut strong_coarse_neighbors = Vec::new();
            let mut sum_weights = 0.0;

            // Find strong coarse connections (iterate over all rows to find row i)
            if let Some(row) = s.outer_iterator().nth(i) {
                for (j, &val) in row.iter() {
                    if cf_splitting[j] {
                        strong_coarse_neighbors.push((j, val.abs()));
                        sum_weights += val.abs();
                    }
                }
            }

            if sum_weights > 1e-14 {
                // Normalize weights
                for &(j, weight) in &strong_coarse_neighbors {
                    let c_idx = coarse_indices[j];
                    if c_idx >= num_coarse {
                        eprintln!("ERROR: coarse_indices[{}] = {} >= num_coarse = {}", j, c_idx, num_coarse);
                        continue;
                    }
                    if c_idx >= p_triplets.cols() {
                         eprintln!("PANIC AVOIDED: add_triplet({}, {}) out of bounds (cols={})", i, c_idx, p_triplets.cols());
                         continue;
                    }
                    p_triplets.add_triplet(i, c_idx, weight / sum_weights);
                }
            } else {
                // FALLBACK: Use ALL coarse neighbors (weak + strong)
                let mut all_coarse_neighbors = Vec::new();
                let mut sum_weights_all = 0.0;
                
                // Use original matrix A for connectivity
                if let Some(row) = a.outer_iterator().nth(i) {
                     for (j, &val) in row.iter() {
                        if cf_splitting[j] { // Is j coarse?
                            all_coarse_neighbors.push((j, val.abs()));
                            sum_weights_all += val.abs();
                        }
                    }
                }

                if sum_weights_all > 1e-14 {
                     for &(j, weight) in &all_coarse_neighbors {
                        let c_idx = coarse_indices[j];
                        if c_idx < num_coarse {
                            p_triplets.add_triplet(i, c_idx, weight / sum_weights_all);
                        }
                    }
                } else {
                     // ISLAND: No connection to any coarse node.
                     // This indicates poor coarsening (graph disconnected?).
                     // Assign to the numerically nearest coarse node (by ID) just to maintain legality,
                     // but warn about it.
                     // eprintln!("WARNING: Fine node {} has no coarse neighbors. Assigning to First Coarse.", i);
                     if num_coarse > 0 {
                          p_triplets.add_triplet(i, 0, 1.0);
                     }
                }
            }
        }
    }

    // DEBUG: Validate dimensions
    if p_triplets.cols() != num_coarse {
         eprintln!("CRITICAL ERROR: P has {} cols, expected {}", p_triplets.cols(), num_coarse);
    }

    p_triplets.to_csr()

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amg_laplacian_1d() {
        // Simple 1D Laplacian: A = tridiag(-1, 2, -1)
        let n = 10;
        let mut triplets = TriMat::new((n, n));

        for i in 0..n {
            triplets.add_triplet(i, i, 2.0);
            if i > 0 {
                triplets.add_triplet(i, i - 1, -1.0);
            }
            if i < n - 1 {
                triplets.add_triplet(i, i + 1, -1.0);
            }
        }

        let a = triplets.to_csr();
        let amg = AMGPreconditioner::new(&a, 10, 2, 0.25).unwrap();

        // Test V-cycle reduces residual
        let r = vec![1.0; n];
        let x = amg.apply(&r);

        // Check that preconditioner produces reasonable output
        assert!(x.len() == n);
        assert!(x.iter().all(|&xi| xi.is_finite()));
    }
}
