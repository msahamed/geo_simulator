/// Stress update algorithms for time-dependent rheology
///
/// Implements stress integration for viscoelastic materials.

use nalgebra::{Point3, SMatrix};
use crate::mesh::Mesh;
use crate::fem::DofManager;
use crate::mechanics::{MaxwellViscoelasticity, StrainDisplacement};

/// Update stress history after time step for Maxwell viscoelasticity
///
/// Computes σ_{n+1} from u_{n+1} and σ_n using Maxwell update:
///
/// σ_{n+1} = [σ_n + 2G Δε] / [1 + Δt/τ_M]
///
/// where Δε = ε(u_{n+1}) - ε(u_n)
///
/// **Algorithm:**
/// For each element:
/// 1. Compute displacement increment Δu = u_{n+1} - u_n
/// 2. Compute strain increment Δε = B · Δu at element centroid
/// 3. Update stress: σ_{n+1} = [σ_n + 2G Δε] / [1 + Δt/τ_M]
///
/// # Arguments
/// * `mesh` - Mesh with current stress_history (σ_n)
/// * `dof_mgr` - DOF manager (3 DOF/node)
/// * `material` - Maxwell material properties
/// * `u_n` - Displacement at t_n
/// * `u_next` - Displacement at t_{n+1}
/// * `dt` - Time step size
///
/// # Returns
/// New stress history for all elements (6×1 vectors in Voigt notation)
///
/// # References
/// - Simo & Hughes, "Computational Inelasticity"
/// - de Souza Neto et al., "Computational Methods for Plasticity"
#[allow(non_snake_case)]
pub fn update_stresses_maxwell(
    mesh: &Mesh,
    dof_mgr: &DofManager,
    material: &MaxwellViscoelasticity,
    u_n: &[f64],
    u_next: &[f64],
    dt: f64,
) -> Vec<SMatrix<f64, 6, 1>> {
    let stress_history = mesh.stress_history.as_ref()
        .expect("Mesh must have stress_history");

    let G = material.shear_modulus();
    let tau_M = material.relaxation_time();
    let factor = 1.0 / (1.0 + dt / tau_M);

    mesh.connectivity.tet10_elements
        .iter()
        .enumerate()
        .map(|(elem_id, elem)| {
            // Get all 10 node coordinates
            let mut nodes = [Point3::origin(); 10];
            for i in 0..10 {
                nodes[i] = mesh.geometry.nodes[elem.nodes[i]];
            }

            // Get old stress
            let sigma_n = stress_history.get(elem_id).clone();

            // Extract element DOFs
            let mut u_elem_n = SMatrix::<f64, 30, 1>::zeros();
            let mut u_elem_next = SMatrix::<f64, 30, 1>::zeros();

            for i in 0..10 {
                for dof in 0..3 {
                    let global_dof = dof_mgr.global_dof(elem.nodes[i], dof);
                    let elem_dof = 3 * i + dof;
                    u_elem_n[elem_dof] = u_n[global_dof];
                    u_elem_next[elem_dof] = u_next[global_dof];
                }
            }

            let du = u_elem_next - u_elem_n;

            // Compute strain increment averaged over Gauss points (same as assembly!)
            // This ensures consistency between stiffness assembly and stress update
            let quad = crate::fem::GaussQuadrature::tet_4point();
            let mut d_strain_avg = SMatrix::<f64, 6, 1>::zeros();

            for qp in quad.points.iter() {
                let B = StrainDisplacement::compute_b_at_point(qp, &nodes);
                let d_strain_qp = B * du;
                d_strain_avg += d_strain_qp;
            }
            d_strain_avg /= quad.points.len() as f64;
            let d_strain = d_strain_avg;

            // Operator-split Maxwell update (predictor-corrector):
            //
            // Step 1: Elastic predictor - compute trial stress as if material were elastic
            // σ^trial = σ_n + D_elastic * Δε
            //
            // Step 2: Maxwell corrector - relax deviatoric part only
            // p = p^trial (volumetric unchanged)
            // s = s^trial / [1 + Δt/τ_M] (deviatoric relaxed)

            // Elastic predictor: σ^trial = σ_n + D_elastic * Δε
            let K = material.youngs_modulus / (3.0 * (1.0 - 2.0 * material.poisson_ratio));
            let lambda = K - 2.0 * G / 3.0;

            // Apply elastic constitutive law to strain increment
            let sigma_trial = SMatrix::<f64, 6, 1>::from_column_slice(&[
                sigma_n[0] + lambda * (d_strain[0] + d_strain[1] + d_strain[2]) + 2.0 * G * d_strain[0],
                sigma_n[1] + lambda * (d_strain[0] + d_strain[1] + d_strain[2]) + 2.0 * G * d_strain[1],
                sigma_n[2] + lambda * (d_strain[0] + d_strain[1] + d_strain[2]) + 2.0 * G * d_strain[2],
                sigma_n[3] + G * d_strain[3],
                sigma_n[4] + G * d_strain[4],
                sigma_n[5] + G * d_strain[5],
            ]);

            // Maxwell corrector: relax deviatoric part only
            let p_trial = (sigma_trial[0] + sigma_trial[1] + sigma_trial[2]) / 3.0;
            let s_trial = SMatrix::<f64, 6, 1>::from_column_slice(&[
                sigma_trial[0] - p_trial,
                sigma_trial[1] - p_trial,
                sigma_trial[2] - p_trial,
                sigma_trial[3],
                sigma_trial[4],
                sigma_trial[5],
            ]);

            let s_next = s_trial * factor;  // Relax deviatoric

            // Recombine
            let sigma_next = SMatrix::<f64, 6, 1>::from_column_slice(&[
                s_next[0] + p_trial,
                s_next[1] + p_trial,
                s_next[2] + p_trial,
                s_next[3],
                s_next[4],
                s_next[5],
            ]);

            sigma_next
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImprovedMeshGenerator;
    use crate::mechanics::MaxwellViscoelasticity;
    use crate::mesh::StressHistory;

    #[test]
    fn test_stress_update_zero_displacement() {
        // Create simple mesh
        let mut mesh = ImprovedMeshGenerator::generate_cube(1, 1, 1, 1.0, 1.0, 1.0);
        let n_elems = mesh.connectivity.tet10_elements.len();
        mesh.stress_history = Some(StressHistory::new(n_elems));

        // Setup DOF manager
        let dof_mgr = DofManager::new(mesh.num_nodes(), 3);
        let n_dofs = dof_mgr.total_dofs();

        // Create material
        let material = MaxwellViscoelasticity::new(100e9, 0.25, 1e19);

        // Zero displacement (no deformation)
        let u_n = vec![0.0; n_dofs];
        let u_next = vec![0.0; n_dofs];
        let dt = 1.0;

        // Update stresses
        let new_stresses = update_stresses_maxwell(&mesh, &dof_mgr, &material, &u_n, &u_next, dt);

        // With zero displacement and zero initial stress, stresses should remain zero
        for stress in new_stresses {
            for component in stress.iter() {
                assert_eq!(*component, 0.0, "Stress should be zero with zero displacement");
            }
        }
    }

    #[test]
    fn test_stress_update_dimensions() {
        // Create simple mesh
        let mut mesh = ImprovedMeshGenerator::generate_cube(1, 1, 1, 1.0, 1.0, 1.0);
        let n_elems = mesh.connectivity.tet10_elements.len();
        mesh.stress_history = Some(StressHistory::new(n_elems));

        let dof_mgr = DofManager::new(mesh.num_nodes(), 3);
        let n_dofs = dof_mgr.total_dofs();

        let material = MaxwellViscoelasticity::new(100e9, 0.25, 1e19);

        let u_n = vec![0.0; n_dofs];
        let u_next = vec![0.01; n_dofs];  // Small uniform displacement
        let dt = 1.0;

        let new_stresses = update_stresses_maxwell(&mesh, &dof_mgr, &material, &u_n, &u_next, dt);

        // Should have one stress vector per element
        assert_eq!(new_stresses.len(), n_elems);

        // Each stress vector should be 6×1
        for stress in new_stresses {
            assert_eq!(stress.nrows(), 6);
            assert_eq!(stress.ncols(), 1);
        }
    }

    #[test]
    fn test_stress_relaxation_with_constant_strain() {
        // Create simple mesh
        let mut mesh = ImprovedMeshGenerator::generate_cube(1, 1, 1, 1.0, 1.0, 1.0);
        let n_elems = mesh.connectivity.tet10_elements.len();

        // Initialize with non-zero stress
        let mut stress_history = StressHistory::new(n_elems);
        let mut initial_stress = SMatrix::<f64, 6, 1>::zeros();
        initial_stress[0] = 1e6;  // 1 MPa
        stress_history.set(0, initial_stress);
        mesh.stress_history = Some(stress_history);

        let dof_mgr = DofManager::new(mesh.num_nodes(), 3);
        let n_dofs = dof_mgr.total_dofs();

        let material = MaxwellViscoelasticity::new(100e9, 0.25, 1e19);
        let tau_M = material.relaxation_time();

        // Constant displacement (no strain increment)
        let u_n = vec![0.01; n_dofs];
        let u_next = vec![0.01; n_dofs];  // Same as u_n
        let dt = tau_M;  // One relaxation time

        let new_stresses = update_stresses_maxwell(&mesh, &dof_mgr, &material, &u_n, &u_next, dt);

        // With dt = tau_M and no strain increment (Δε = 0):
        // Only DEVIATORIC part decays, volumetric stays constant
        // σ_xx = p + s_xx where p = σ_xx/3, s_xx = 2σ_xx/3
        // After decay: s_xx_new = s_xx / (1 + dt/τ_M) = s_xx/2
        // σ_xx_new = p + s_xx_new = σ_xx/3 + (2σ_xx/3)/2 = σ_xx/3 + σ_xx/3 = 2σ_xx/3
        let expected_stress = initial_stress[0] * 2.0 / 3.0;
        let actual_stress = new_stresses[0][0];

        assert!((actual_stress - expected_stress).abs() / expected_stress < 1e-6,
                "Stress should decay to 2/3 after dt=tau_M (only deviatoric relaxes). Expected {}, got {}",
                expected_stress, actual_stress);
    }
}
