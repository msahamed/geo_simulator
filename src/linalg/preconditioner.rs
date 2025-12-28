use sprs::CsMat;

/// Preconditioner trait for iterative solvers
///
/// Solves M z = r approximately (where M ≈ A)
pub trait Preconditioner {
    /// Apply preconditioner: solve M z = r
    ///
    /// # Arguments
    /// * `r` - Input vector
    ///
    /// # Returns
    /// * z - Preconditioned vector
    fn apply(&self, r: &[f64]) -> Vec<f64>;
}

/// Jacobi (diagonal) preconditioner
///
/// M = diag(A)
/// Very cheap but less effective
pub struct JacobiPreconditioner {
    /// Inverse of diagonal entries: 1/A_ii
    diag_inv: Vec<f64>,
}

impl JacobiPreconditioner {
    /// Create Jacobi preconditioner from matrix A
    #[allow(non_snake_case)]
    pub fn new(A: &CsMat<f64>) -> Self {
        let n = A.rows();
        let mut diag_inv = vec![1.0; n];

        // Extract diagonal
        for i in 0..n {
            if let Some(&val) = A.get(i, i) {
                if val.abs() > 1e-14 {
                    diag_inv[i] = 1.0 / val;
                }
            }
        }

        Self { diag_inv }
    }
}

impl Preconditioner for JacobiPreconditioner {
    fn apply(&self, r: &[f64]) -> Vec<f64> {
        // z = D^{-1} r
        r.iter()
            .zip(self.diag_inv.iter())
            .map(|(&ri, &di)| ri * di)
            .collect()
    }
}

/// Identity preconditioner (no preconditioning)
pub struct IdentityPreconditioner;

impl Preconditioner for IdentityPreconditioner {
    fn apply(&self, r: &[f64]) -> Vec<f64> {
        r.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;
    use approx::assert_relative_eq;

    #[test]
    fn test_jacobi_preconditioner() {
        // Diagonal matrix
        let mut triplets = TriMat::new((3, 3));
        triplets.add_triplet(0, 0, 2.0);
        triplets.add_triplet(1, 1, 4.0);
        triplets.add_triplet(2, 2, 8.0);
        let A = triplets.to_csr();

        let precond = JacobiPreconditioner::new(&A);

        let r = vec![2.0, 4.0, 8.0];
        let z = precond.apply(&r);

        // z[i] = r[i] / A[i][i]
        assert_relative_eq!(z[0], 1.0, epsilon = 1e-14);
        assert_relative_eq!(z[1], 1.0, epsilon = 1e-14);
        assert_relative_eq!(z[2], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_identity_preconditioner() {
        let precond = IdentityPreconditioner;
        let r = vec![1.0, 2.0, 3.0];
        let z = precond.apply(&r);

        assert_eq!(z, r);
    }
}
