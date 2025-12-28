/// Element-level state storage for viscoelasticity
///
/// Stores stress history required for time-dependent rheology models.

use nalgebra::SMatrix;

/// Stress state storage for viscoelasticity
///
/// Maintains deviatoric stress tensors at each element for Maxwell viscoelasticity.
/// Uses Voigt notation: [σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_xz]
#[derive(Debug, Clone)]
pub struct StressHistory {
    /// Deviatoric stress at each element (6 components, Voigt notation)
    /// Indexed by element ID
    pub stress: Vec<SMatrix<f64, 6, 1>>,
}

impl StressHistory {
    /// Create new stress history for mesh with n_elements
    ///
    /// Initializes all stresses to zero.
    ///
    /// # Arguments
    /// * `n_elements` - Number of elements in the mesh
    ///
    /// # Returns
    /// StressHistory with zero-initialized stress vectors
    pub fn new(n_elements: usize) -> Self {
        Self {
            stress: vec![SMatrix::<f64, 6, 1>::zeros(); n_elements],
        }
    }

    /// Get stress for element
    ///
    /// # Arguments
    /// * `elem_id` - Element index
    ///
    /// # Returns
    /// Reference to 6×1 stress vector in Voigt notation
    pub fn get(&self, elem_id: usize) -> &SMatrix<f64, 6, 1> {
        &self.stress[elem_id]
    }

    /// Set stress for element
    ///
    /// # Arguments
    /// * `elem_id` - Element index
    /// * `stress` - New stress tensor (6×1 Voigt notation)
    pub fn set(&mut self, elem_id: usize, stress: SMatrix<f64, 6, 1>) {
        self.stress[elem_id] = stress;
    }

    /// Update all stresses (for time step)
    ///
    /// Replaces entire stress history with new values.
    /// Used after computing stress update at end of time step.
    ///
    /// # Arguments
    /// * `new_stresses` - Vector of new stress tensors (one per element)
    ///
    /// # Panics
    /// Panics if size doesn't match current number of elements
    pub fn update_all(&mut self, new_stresses: Vec<SMatrix<f64, 6, 1>>) {
        assert_eq!(
            self.stress.len(),
            new_stresses.len(),
            "New stress vector size must match existing size"
        );
        self.stress = new_stresses;
    }

    /// Get number of elements
    pub fn num_elements(&self) -> usize {
        self.stress.len()
    }
}

/// Plastic strain state storage
/// 
/// Tracks accumulated plastic strain at each element for strain softening.
#[derive(Debug, Clone)]
pub struct PlasticityState {
    /// Accumulated plastic strain (scalar magnitude)
    /// Indexed by element ID
    pub accumulated_strain: Vec<f64>,
}

impl PlasticityState {
    /// Create new plasticity state for mesh
    pub fn new(n_elements: usize) -> Self {
        Self {
            accumulated_strain: vec![0.0; n_elements],
        }
    }

    pub fn get(&self, elem_id: usize) -> f64 {
        self.accumulated_strain[elem_id]
    }

    pub fn set(&mut self, elem_id: usize, strain: f64) {
        self.accumulated_strain[elem_id] = strain;
    }

    pub fn add(&mut self, elem_id: usize, increment: f64) {
        self.accumulated_strain[elem_id] += increment;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_zero_stresses() {
        let history = StressHistory::new(5);

        assert_eq!(history.num_elements(), 5);

        for i in 0..5 {
            let stress = history.get(i);
            for j in 0..6 {
                assert_eq!(stress[j], 0.0, "Stress component {} should be zero", j);
            }
        }
    }

    #[test]
    fn test_get_set_individual_element() {
        let mut history = StressHistory::new(3);

        // Set stress for element 1
        let mut test_stress = SMatrix::<f64, 6, 1>::zeros();
        test_stress[0] = 100.0; // σ_xx
        test_stress[3] = 50.0;  // σ_xy

        history.set(1, test_stress);

        // Verify element 1 has new stress
        let retrieved = history.get(1);
        assert_eq!(retrieved[0], 100.0);
        assert_eq!(retrieved[3], 50.0);

        // Verify other elements still zero
        for i in 0..6 {
            assert_eq!(history.get(0)[i], 0.0);
            assert_eq!(history.get(2)[i], 0.0);
        }
    }

    #[test]
    fn test_update_all() {
        let mut history = StressHistory::new(2);

        // Create new stress vectors
        let mut stress0 = SMatrix::<f64, 6, 1>::zeros();
        stress0[0] = 10.0;

        let mut stress1 = SMatrix::<f64, 6, 1>::zeros();
        stress1[1] = 20.0;

        let new_stresses = vec![stress0, stress1];

        // Update all
        history.update_all(new_stresses);

        // Verify
        assert_eq!(history.get(0)[0], 10.0);
        assert_eq!(history.get(1)[1], 20.0);
    }

    #[test]
    #[should_panic(expected = "New stress vector size must match existing size")]
    fn test_update_all_size_mismatch_panics() {
        let mut history = StressHistory::new(3);

        let new_stresses = vec![SMatrix::<f64, 6, 1>::zeros(); 2]; // Wrong size

        history.update_all(new_stresses);
    }

    #[test]
    fn test_clone() {
        let mut history = StressHistory::new(2);

        let mut stress = SMatrix::<f64, 6, 1>::zeros();
        stress[0] = 42.0;
        history.set(0, stress);

        let cloned = history.clone();

        assert_eq!(cloned.get(0)[0], 42.0);
        assert_eq!(cloned.num_elements(), 2);
    }
}
