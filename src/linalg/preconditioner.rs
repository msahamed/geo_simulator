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

/// Incomplete LU preconditioner with zero fill-in (ILU(0))
///
/// M = L * U ≈ A
/// L and U have the same sparsity pattern as A.
/// Efficient and robust for geodynamic systems with high viscosity contrasts.
pub struct ILUPreconditioner {
    /// Combined L and U factors sharing the same sparsity as the original matrix.
    factors: CsMat<f64>,
    /// Pre-calculated indices of the diagonal elements for each row.
    diag_indices: Vec<usize>,
}

impl ILUPreconditioner {
    /// Create a new ILU(0) preconditioner from matrix A
    #[allow(non_snake_case)]
    pub fn new(A: &CsMat<f64>) -> Result<Self, String> {
        let mut factors = A.clone();
        let n = factors.rows();
        let indptr = factors.indptr().as_slice().unwrap().to_vec();
        let indices = factors.indices().to_vec();

        // 1. Pre-calculate diagonal indices once
        let mut diag_indices = vec![0; n];
        for i in 0..n {
            let mut found = false;
            for idx in indptr[i]..indptr[i+1] {
                if indices[idx] == i {
                    diag_indices[i] = idx;
                    found = true;
                    break;
                }
            }
            if !found { return Err(format!("Row {} has no diag", i)); }
        }

        Self::factorize_in_place(&mut factors, &indptr, &indices, &diag_indices)?;
        
        Ok(Self { factors, diag_indices })
    }

    /// Update the preconditions factors in-place using a new matrix A with the SAME sparsity pattern.
    pub fn update(&mut self, A: &CsMat<f64>) -> Result<(), String> {
        let n = A.rows();
        if n != self.factors.rows() {
            return Err("Size mismatch".to_string());
        }

        self.factors.data_mut().copy_from_slice(A.data());

        let indptr = self.factors.indptr().as_slice().unwrap().to_vec();
        let indices = self.factors.indices().to_vec();

        Self::factorize_in_place(&mut self.factors, &indptr, &indices, &self.diag_indices)
    }

    fn factorize_in_place(factors: &mut CsMat<f64>, indptr: &[usize], indices: &[usize], diag_indices: &[usize]) -> Result<(), String> {
        let n = factors.rows();
        
        for i in 0..n {
            let row_start = indptr[i];
            let diag_i = diag_indices[i];
            
            // Elimination using previous rows k < i
            for k_idx_abs in row_start..diag_i {
                let k = indices[k_idx_abs];
                let k_diag = diag_indices[k];
                let k_end = indptr[k+1];
                
                let diag_val_k = factors.data()[k_diag];
                if diag_val_k.abs() < 1e-15 { return Err(format!("Zero diag at {}", k)); }
                
                let val_ik = factors.data()[k_idx_abs] / diag_val_k;
                factors.data_mut()[k_idx_abs] = val_ik;
                
                // Update A(i, j) = A(i, j) - A(i, k) * A(k, j) for j > k
                // We use the sorted nature of CSR column indices for a fast merge-style update
                let mut cur_i_idx = k_idx_abs + 1;
                let row_i_end = indptr[i+1];
                
                for k_j_idx in k_diag + 1..k_end {
                    let col_j = indices[k_j_idx];
                    while cur_i_idx < row_i_end && indices[cur_i_idx] < col_j { cur_i_idx += 1; }
                    if cur_i_idx == row_i_end { break; }
                    if indices[cur_i_idx] == col_j {
                        let val_kj = factors.data()[k_j_idx];
                        factors.data_mut()[cur_i_idx] -= val_ik * val_kj;
                    }
                }
            }
            if factors.data()[diag_i].abs() < 1e-15 { return Err(format!("Zero diag after update at {}", i)); }
        }
        Ok(())
    }
}

impl Preconditioner for ILUPreconditioner {
    fn apply(&self, r: &[f64]) -> Vec<f64> {
        let n = self.factors.rows();
        let mut z = r.to_vec();
        
        let binding = self.factors.indptr();
        let indptr = binding.as_slice().unwrap();
        let indices = self.factors.indices();
        let data = self.factors.data();
        
        // Forward solve: L y = r (L is lower part, unit diagonal)
        for i in 0..n {
            let start = indptr[i];
            let diag = self.diag_indices[i];
            let mut sum = 0.0;
            // Iterate only over L part (before diagonal)
            for idx in start..diag {
                sum += data[idx] * z[indices[idx]];
            }
            z[i] -= sum;
        }
        
        // Backward solve: U x = y (U is upper part, includes diagonal)
        for i in (0..n).rev() {
            let diag = self.diag_indices[i];
            let end = indptr[i+1];
            let mut sum = 0.0;
            // Iterate only over U part (after diagonal)
            for idx in diag + 1..end {
                sum += data[idx] * z[indices[idx]];
            }
            z[i] = (z[i] - sum) / data[diag];
        }
        
        z
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

    #[test]
    fn test_ilu0_preconditioner() {
        // Simple tridiagonal matrix
        // [ 2 -1  0 ]
        // [-1  2 -1 ]
        // [ 0 -1  2 ]
        let mut triplets = TriMat::new((3, 3));
        triplets.add_triplet(0, 0, 2.0);
        triplets.add_triplet(0, 1, -1.0);
        triplets.add_triplet(1, 0, -1.0);
        triplets.add_triplet(1, 1, 2.0);
        triplets.add_triplet(1, 2, -1.0);
        triplets.add_triplet(2, 1, -1.0);
        triplets.add_triplet(2, 2, 2.0);
        let A = triplets.to_csr();

        // Manual ILU(0) calculation:
        // i=0: diag=2.0
        // i=1:
        //   k=0 (col 0): val_ik = -1.0 / 2.0 = -0.5. update A(1,1): 2.0 - (-0.5)*(-1.0) = 1.5
        // i=2:
        //   k=1 (col 1): val_ik = -1.0 / 1.5 = -0.666... update A(2,2): 2.0 - (-0.666)*(-1.0) = 1.333...

        let precond = ILUPreconditioner::new(&A).unwrap();
        
        // Test apply: solve (L*U)z = r
        // Test with r = A * [1, 2, 3]^T = [0, 0, 4]^T
        let r = vec![0.0, 0.0, 4.0];
        let z = precond.apply(&r);
        
        // z should be close to [1, 2, 3]
        assert_relative_eq!(z[0], 1.0, epsilon = 0.1); // ILU(0) is an approximation
        assert_relative_eq!(z[1], 2.0, epsilon = 0.1);
        assert_relative_eq!(z[2], 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_ilu0_fail_fast() {
        // Matrix with a zero on diagonal
        let mut triplets = TriMat::new((2, 2));
        triplets.add_triplet(0, 0, 0.0);
        triplets.add_triplet(1, 1, 1.0);
        let A = triplets.to_csr();

        let result = ILUPreconditioner::new(&A);
        assert!(result.is_err());
    }
}
