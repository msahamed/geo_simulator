//! Adaptive timestepping strategies
//!
//! This module provides adaptive timestep computation based on multiple
//! constraints including CFL condition and Maxwell relaxation time.

use crate::fem::DofManager;
use crate::mechanics::ElastoViscoPlastic;
use crate::mesh::Mesh;

/// Adaptive timestepping parameters and diagnostics
///
/// Contains the computed timestep along with diagnostic information
/// about the constraints that were applied.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveTimestep {
    /// Computed timestep (seconds)
    pub dt: f64,
    /// CFL-limited timestep (seconds)
    pub cfl_dt: f64,
    /// Maxwell relaxation time limit (seconds)
    pub maxwell_dt: f64,
    /// Maximum velocity magnitude (m/s)
    pub max_velocity: f64,
    /// Minimum cell size (m)
    pub min_cell_size: f64,
}

impl AdaptiveTimestep {
    /// Check which constraint is limiting the timestep
    pub fn limiting_constraint(&self) -> &'static str {
        if (self.dt - self.cfl_dt).abs() < 1e-10 {
            "CFL"
        } else if (self.dt - self.maxwell_dt).abs() < 1e-10 {
            "Maxwell"
        } else {
            "Manual (dt_max)"
        }
    }

    /// Get timestep in years
    pub fn dt_years(&self) -> f64 {
        self.dt / (365.25 * 24.0 * 3600.0)
    }

    /// Get CFL number
    pub fn cfl_number(&self) -> f64 {
        if self.max_velocity > 1e-20 && self.min_cell_size > 0.0 {
            self.dt * self.max_velocity / self.min_cell_size
        } else {
            0.0
        }
    }
}

/// Compute adaptive timestep based on CFL and Maxwell constraints
///
/// This function computes a stable timestep by considering:
/// 1. **CFL constraint**: dt ≤ CFL_target * dx / v_max
///    - Ensures numerical stability for advection
///    - Prevents tracers from moving more than ~CFL_target cells per step
///
/// 2. **Maxwell constraint**: dt ≤ η / G
///    - Ensures accurate viscoelastic coupling
///    - Maxwell time is the characteristic time for stress relaxation
///
/// The final timestep is the minimum of all active constraints, clamped
/// between dt_min and dt_max.
///
/// # Arguments
/// * `mesh` - Computational mesh
/// * `dof_mgr` - DOF manager for velocity indexing
/// * `current_sol` - Current solution vector (velocity + pressure)
/// * `materials` - Material properties (ElastoViscoPlastic) for Maxwell time
/// * `cfl_target` - Target CFL number (typically 0.3-0.5)
/// * `dt_min` - Minimum allowed timestep (seconds)
/// * `dt_max` - Maximum allowed timestep (seconds)
/// * `use_cfl_constraint` - Whether to apply CFL constraint
/// * `use_maxwell_constraint` - Whether to apply Maxwell relaxation constraint
///
/// # Returns
/// AdaptiveTimestep structure with computed dt and diagnostics
///
/// # Examples
/// ```ignore
/// use geo_simulator::timestepping::compute_adaptive_timestep;
///
/// let adaptive = compute_adaptive_timestep(
///     &mesh,
///     &dof_mgr,
///     &solution,
///     &materials,
///     0.5,    // CFL target
///     100.0 * 365.25 * 24.0 * 3600.0,   // dt_min: 100 years
///     5000.0 * 365.25 * 24.0 * 3600.0,  // dt_max: 5000 years
///     true,   // use CFL
///     false,  // don't use Maxwell
/// );
///
/// println!("Timestep: {:.1} years (limited by: {})",
///          adaptive.dt_years(),
///          adaptive.limiting_constraint());
/// ```
pub fn compute_adaptive_timestep(
    mesh: &Mesh,
    dof_mgr: &DofManager,
    current_sol: &[f64],
    materials: &[ElastoViscoPlastic],
    cfl_target: f64,
    dt_min: f64,
    dt_max: f64,
    use_cfl_constraint: bool,
    use_maxwell_constraint: bool,
) -> AdaptiveTimestep {
    // 1. Compute minimum cell size (characteristic length scale)
    let mut min_cell_size = f64::INFINITY;
    for elem in &mesh.connectivity.tet10_elements {
        // Compute element characteristic size (minimum edge length)
        let nodes = &elem.nodes;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let p1 = &mesh.geometry.nodes[nodes[i]];
                let p2 = &mesh.geometry.nodes[nodes[j]];
                let dx = p1.x - p2.x;
                let dy = p1.y - p2.y;
                let dz = p1.z - p2.z;
                let edge_length = (dx * dx + dy * dy + dz * dz).sqrt();
                min_cell_size = min_cell_size.min(edge_length);
            }
        }
    }

    // 2. Compute maximum velocity magnitude
    let mut max_velocity: f64 = 0.0;
    for node_id in 0..mesh.num_nodes() {
        let vx = current_sol[dof_mgr.velocity_dof(node_id, 0)];
        let vy = current_sol[dof_mgr.velocity_dof(node_id, 1)];
        let vz = current_sol[dof_mgr.velocity_dof(node_id, 2)];
        let v_mag = (vx * vx + vy * vy + vz * vz).sqrt();
        max_velocity = max_velocity.max(v_mag);
    }

    // 3. CFL constraint: dt_cfl = CFL * dx / v_max
    let cfl_dt = if max_velocity > 1e-20 {
        cfl_target * min_cell_size / max_velocity
    } else {
        dt_max // No velocity -> use maximum timestep
    };

    // 4. Maxwell relaxation time constraint: dt_maxwell = eta / G
    // Use minimum Maxwell time across all materials
    let mut min_maxwell_time = f64::INFINITY;
    for mat in materials {
        let maxwell_time = mat.viscosity / mat.shear_modulus();
        min_maxwell_time = min_maxwell_time.min(maxwell_time);
    }
    let maxwell_dt = min_maxwell_time; // Full Maxwell time

    // 5. Apply constraints and clamp
    let mut dt = dt_max;

    if use_cfl_constraint {
        dt = dt.min(cfl_dt);
    }

    if use_maxwell_constraint {
        dt = dt.min(maxwell_dt);
    }

    // Clamp between min and max
    dt = dt.max(dt_min).min(dt_max);

    AdaptiveTimestep {
        dt,
        cfl_dt,
        maxwell_dt,
        max_velocity,
        min_cell_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_timestep_diagnostics() {
        let adaptive = AdaptiveTimestep {
            dt: 1000.0 * 365.25 * 24.0 * 3600.0,  // 1000 years
            cfl_dt: 1000.0 * 365.25 * 24.0 * 3600.0,
            maxwell_dt: 5000.0 * 365.25 * 24.0 * 3600.0,
            max_velocity: 1e-9,
            min_cell_size: 5000.0,
        };

        assert_eq!(adaptive.limiting_constraint(), "CFL");
        assert!((adaptive.dt_years() - 1000.0).abs() < 1.0);
        assert!(adaptive.cfl_number() > 0.0);
    }

    #[test]
    fn test_limiting_constraint_detection() {
        // CFL limiting
        let cfl_limited = AdaptiveTimestep {
            dt: 100.0,
            cfl_dt: 100.0,
            maxwell_dt: 200.0,
            max_velocity: 1e-9,
            min_cell_size: 5000.0,
        };
        assert_eq!(cfl_limited.limiting_constraint(), "CFL");

        // Maxwell limiting
        let maxwell_limited = AdaptiveTimestep {
            dt: 100.0,
            cfl_dt: 200.0,
            maxwell_dt: 100.0,
            max_velocity: 1e-9,
            min_cell_size: 5000.0,
        };
        assert_eq!(maxwell_limited.limiting_constraint(), "Maxwell");

        // Manual limiting (dt_max)
        let manual_limited = AdaptiveTimestep {
            dt: 100.0,
            cfl_dt: 200.0,
            maxwell_dt: 300.0,
            max_velocity: 1e-9,
            min_cell_size: 5000.0,
        };
        assert_eq!(manual_limited.limiting_constraint(), "Manual (dt_max)");
    }
}
