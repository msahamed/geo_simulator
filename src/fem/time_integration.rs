/// Time integration for transient problems
///
/// Implements time-stepping schemes for quasi-static and dynamic problems.

use crate::mesh::Mesh;
use crate::fem::{DofManager, Assembler};
use crate::mechanics::MaxwellViscoelasticity;
use crate::linalg::Solver;

/// Time integration statistics for a single step
#[derive(Debug, Clone)]
pub struct TimeStepStats {
    /// Current simulation time
    pub time: f64,
    /// Time step size used
    pub dt: f64,
    /// Number of solver iterations
    pub iterations: usize,
    /// Final residual norm
    pub residual: f64,
}

/// Backward Euler time integrator for Maxwell viscoelasticity
///
/// Implements first-order implicit time integration for quasi-static problems.
/// Unconditionally stable for any time step size.
///
/// **Algorithm:**
/// At each time step t_n → t_{n+1}:
/// 1. Assemble effective system: K_eff(Δt) u_{n+1} = f_ext + f_history(σ_n)
/// 2. Apply boundary conditions
/// 3. Solve for u_{n+1}
/// 4. Update stress: σ_{n+1} = [σ_n + 2G Δε] / [1 + Δt/τ_M]
///
/// # References
/// - Hughes, "The Finite Element Method"
/// - Zienkiewicz & Taylor, "The Finite Element Method", Vol. 2
pub struct BackwardEuler {
    /// Current simulation time
    pub time: f64,
    /// Time step size
    pub dt: f64,
    /// Current displacement field
    pub displacement: Vec<f64>,
}

impl BackwardEuler {
    /// Create new backward Euler integrator
    ///
    /// # Arguments
    /// * `n_dofs` - Total number of degrees of freedom
    /// * `dt` - Time step size (seconds)
    ///
    /// # Returns
    /// Integrator initialized at t=0 with zero displacement
    pub fn new(n_dofs: usize, dt: f64) -> Self {
        assert!(dt > 0.0, "Time step must be positive, got {}", dt);
        assert!(n_dofs > 0, "Number of DOFs must be positive, got {}", n_dofs);

        Self {
            time: 0.0,
            dt,
            displacement: vec![0.0; n_dofs],
        }
    }

    /// Take one time step: t_n → t_{n+1}
    ///
    /// Solves the global system:
    /// K_eff u_{n+1} = f_ext + f_history
    ///
    /// where:
    /// - K_eff = effective stiffness with time-dependent relaxation
    /// - f_ext = external forces (gravity, surface tractions)
    /// - f_history = pseudo-force from stress history
    ///
    /// # Arguments
    /// * `mesh` - Mesh with current stress_history at t_n
    /// * `dof_mgr` - DOF manager with boundary conditions
    /// * `material` - Maxwell material properties
    /// * `f_ext` - External force vector
    /// * `solver` - Linear system solver
    ///
    /// # Returns
    /// (u_{n+1}, statistics)
    ///
    /// # Note
    /// This method does NOT update the stress history. Call stress update separately.
    #[allow(non_snake_case)]
    pub fn step<S: Solver>(
        &mut self,
        mesh: &Mesh,
        dof_mgr: &DofManager,
        material: &MaxwellViscoelasticity,
        f_ext: &[f64],
        solver: &mut S,
    ) -> (Vec<f64>, TimeStepStats) {
        // Assemble effective system with stress history
        let (K_eff, f_history) = Assembler::assemble_maxwell_viscoelastic_parallel(
            mesh, dof_mgr, material, self.dt
        );

        // Total RHS: f = f_ext + f_history
        let mut f_total = f_ext.to_vec();
        for i in 0..f_total.len() {
            f_total[i] += f_history[i];
        }

        // Apply boundary conditions
        let (K_bc, f_bc) = Assembler::apply_dirichlet_bcs(&K_eff, &f_total, dof_mgr);

        // Solve linear system
        let (u_next, stats) = solver.solve(&K_bc, &f_bc);

        // Update internal state
        self.displacement = u_next.clone();
        self.time += self.dt;

        let step_stats = TimeStepStats {
            time: self.time,
            dt: self.dt,
            iterations: stats.iterations,
            residual: stats.residual_norm,
        };

        (u_next, step_stats)
    }

    /// Get current simulation time
    pub fn current_time(&self) -> f64 {
        self.time
    }

    /// Get current displacement
    pub fn current_displacement(&self) -> &[f64] {
        &self.displacement
    }

    /// Reset integrator to initial state
    ///
    /// Sets time to 0 and clears displacement.
    pub fn reset(&mut self) {
        self.time = 0.0;
        for disp in &mut self.displacement {
            *disp = 0.0;
        }
    }

    /// Change time step size
    ///
    /// # Arguments
    /// * `new_dt` - New time step size (must be positive)
    pub fn set_timestep(&mut self, new_dt: f64) {
        assert!(new_dt > 0.0, "Time step must be positive, got {}", new_dt);
        self.dt = new_dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward_euler_creation() {
        let integrator = BackwardEuler::new(100, 1.0);

        assert_eq!(integrator.time, 0.0);
        assert_eq!(integrator.dt, 1.0);
        assert_eq!(integrator.displacement.len(), 100);
        assert!(integrator.displacement.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_current_time() {
        let integrator = BackwardEuler::new(10, 0.5);
        assert_eq!(integrator.current_time(), 0.0);
    }

    #[test]
    fn test_reset() {
        let mut integrator = BackwardEuler::new(10, 1.0);

        // Manually modify state
        integrator.time = 5.0;
        integrator.displacement[0] = 1.0;

        // Reset
        integrator.reset();

        assert_eq!(integrator.time, 0.0);
        assert!(integrator.displacement.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_set_timestep() {
        let mut integrator = BackwardEuler::new(10, 1.0);

        integrator.set_timestep(0.5);
        assert_eq!(integrator.dt, 0.5);
    }

    #[test]
    #[should_panic(expected = "Time step must be positive")]
    fn test_negative_timestep() {
        BackwardEuler::new(10, -1.0);
    }

    #[test]
    #[should_panic(expected = "Number of DOFs must be positive")]
    fn test_zero_dofs() {
        BackwardEuler::new(0, 1.0);
    }

    #[test]
    #[should_panic(expected = "Time step must be positive")]
    fn test_set_negative_timestep() {
        let mut integrator = BackwardEuler::new(10, 1.0);
        integrator.set_timestep(-0.5);
    }
}
