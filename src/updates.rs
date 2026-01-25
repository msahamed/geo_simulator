//! Standardized update routines for geodynamic simulations
//!
//! This module provides reusable functions for updating physical properties,
//! visualization fields, and state variables during time-stepping.

use crate::fem::{DofManager, GaussQuadrature, Tet10Basis};
use crate::mechanics::{ElastoViscoPlastic, StrainDisplacement};
use crate::mesh::{Mesh, TracerSwarm, SearchGrid};
use nalgebra::Point3;
use rayon::prelude::*;

/// Physical properties for an element
#[derive(Debug, Clone, Copy)]
pub struct ElementProperties {
    pub strain_rate_ii: f64,      // Second invariant of strain rate tensor
    pub stress_ii: f64,            // Second invariant of deviatoric stress
    pub effective_viscosity: f64,  // Min of viscous and plastic viscosity
    pub pressure: f64,             // Element pressure
}

/// Visualization fields for diagnostics
#[derive(Debug, Clone)]
pub struct VisualizationFields {
    pub is_yielding: Vec<f64>,           // 1.0 if yielding, 0.0 otherwise
    pub yield_stress: Vec<f64>,          // Yield stress (Pa)
    pub plastic_strain_rate: Vec<f64>,   // Plastic strain rate (1/s)
    pub softened_cohesion: Vec<f64>,     // Softened cohesion (Pa)
    pub plastic_viscosity: Vec<f64>,     // Plastic viscosity (Pa·s)
}

/// Compute physical properties for all elements (parallelized)
///
/// Computes strain rate, stress, effective viscosity, and pressure for each element
/// from the current velocity/pressure solution.
///
/// # Arguments
/// * `mesh` - Computational mesh
/// * `dof_mgr` - DOF manager for velocity/pressure indexing
/// * `current_sol` - Current solution vector (velocity + pressure)
/// * `materials` - Material properties (ElastoViscoPlastic) for each material ID
/// * `elem_mat_ids` - Material ID for each element
/// * `elem_strains` - Accumulated plastic strain for each element
///
/// # Returns
/// Vector of ElementProperties for each element
#[allow(non_snake_case)]
pub fn compute_element_properties(
    mesh: &Mesh,
    dof_mgr: &DofManager,
    current_sol: &[f64],
    materials: &[ElastoViscoPlastic],
    elem_mat_ids: &[u32],
    elem_strains: &[f64],
) -> Vec<ElementProperties> {
    (0..mesh.num_elements())
        .into_par_iter()
        .map(|elem_id| {
            let elem = &mesh.connectivity.tet10_elements[elem_id];
            let mut nodes_elem = [Point3::origin(); 10];
            for i in 0..10 {
                nodes_elem[i] = mesh.geometry.nodes[elem.nodes[i]];
            }

            // Compute average strain rate using Gaussian quadrature
            let mut strain_rate = nalgebra::SMatrix::<f64, 6, 1>::zeros();
            let quad = GaussQuadrature::tet_4point();
            for qp in &quad.points {
                let b = StrainDisplacement::compute_b_at_point(qp, &nodes_elem);
                let mut v_elem = nalgebra::SMatrix::<f64, 30, 1>::zeros();
                for i in 0..10 {
                    for comp in 0..3 {
                        v_elem[3 * i + comp] =
                            current_sol[dof_mgr.velocity_dof(elem.nodes[i], comp)];
                    }
                }
                strain_rate += b * v_elem;
            }
            strain_rate /= quad.points.len() as f64;

            // Compute second invariant: J₂(ε̇) = ½ ε̇ᵢⱼ ε̇ᵢⱼ
            let j2_edot = 0.5
                * (strain_rate[0] * strain_rate[0]
                    + strain_rate[1] * strain_rate[1]
                    + strain_rate[2] * strain_rate[2]
                    + 0.5
                        * (strain_rate[3] * strain_rate[3]
                            + strain_rate[4] * strain_rate[4]
                            + strain_rate[5] * strain_rate[5]));
            let sr_mag = j2_edot.sqrt();

            let mat_idx = elem_mat_ids[elem_id] as usize;
            let eps_p = elem_strains[elem_id];

            // Compute average pressure from corner nodes
            let mut p_elem = 0.0;
            for i in 0..4 {
                p_elem += current_sol[dof_mgr.pressure_dof(elem.nodes[i]).unwrap()];
            }
            p_elem /= 4.0;

            // Compute effective viscosity (min of viscous and plastic)
            let mu_p = materials[mat_idx]
                .plasticity
                .softened_viscosity(&strain_rate, p_elem, eps_p);
            let mu_v = materials[mat_idx].viscosity;
            let mu_eff = mu_v.min(mu_p);

            // Compute deviatoric stress: τ = 2μ_eff √J₂(ε̇)
            let stress = 2.0 * mu_eff * sr_mag;

            ElementProperties {
                strain_rate_ii: sr_mag,
                stress_ii: stress,
                effective_viscosity: mu_eff,
                pressure: p_elem,
            }
        })
        .collect()
}

/// Compute visualization fields for diagnostics
///
/// # Arguments
/// * `mesh` - Computational mesh
/// * `materials` - Material properties (ElastoViscoPlastic)
/// * `elem_mat_ids` - Material ID for each element
/// * `elem_strains` - Accumulated plastic strain
/// * `elem_props` - Computed element properties
///
/// # Returns
/// VisualizationFields structure with diagnostic fields
pub fn compute_visualization_fields(
    mesh: &Mesh,
    materials: &[ElastoViscoPlastic],
    elem_mat_ids: &[u32],
    elem_strains: &[f64],
    elem_props: &[ElementProperties],
) -> VisualizationFields {
    let num_elems = mesh.num_elements();
    let mut viz = VisualizationFields {
        is_yielding: vec![0.0; num_elems],
        yield_stress: vec![0.0; num_elems],
        plastic_strain_rate: vec![0.0; num_elems],
        softened_cohesion: vec![0.0; num_elems],
        plastic_viscosity: vec![0.0; num_elems],
    };

    for elem_id in 0..num_elems {
        let mat_id = elem_mat_ids[elem_id] as usize;
        let eps_p = elem_strains[elem_id];
        let props = &elem_props[elem_id];

        // Get softened material properties
        let (c_soft, phi_soft) = materials[mat_id].plasticity.softened_properties(eps_p);
        viz.softened_cohesion[elem_id] = c_soft;

        // Compute yield stress: σ_y = c·cos(φ) + p·sin(φ) (Drucker-Prager)
        viz.yield_stress[elem_id] =
            c_soft * phi_soft.cos() + props.pressure * phi_soft.sin();

        // Compute plastic viscosity: μ_p = σ_y / (2·√J₂(ε̇))
        if props.strain_rate_ii > 1e-30 {
            viz.plastic_viscosity[elem_id] =
                viz.yield_stress[elem_id] / (2.0 * props.strain_rate_ii);
            // Apply minimum to avoid numerical issues
            viz.plastic_viscosity[elem_id] = viz.plastic_viscosity[elem_id].max(1e18);
        } else {
            viz.plastic_viscosity[elem_id] = 1e30; // Very large (not yielding)
        }

        // Check if element is yielding: μ_eff < μ_viscous
        if props.effective_viscosity < materials[mat_id].viscosity {
            viz.is_yielding[elem_id] = 1.0;
            viz.plastic_strain_rate[elem_id] = props.strain_rate_ii;
        } else {
            viz.is_yielding[elem_id] = 0.0;
            viz.plastic_strain_rate[elem_id] = 0.0;
        }
    }

    viz
}

/// Update tracer properties from element data
///
/// Updates plastic strain, strain rate, stress, viscosity, and pressure for each tracer
/// based on the element it resides in.
///
/// # Arguments
/// * `mesh` - Computational mesh
/// * `swarm` - Tracer swarm (modified in place)
/// * `materials` - Material properties (ElastoViscoPlastic)
/// * `elem_mat_ids` - Material ID for each element
/// * `elem_props` - Computed element properties
/// * `dt` - Time step size (seconds)
#[allow(clippy::too_many_arguments)]
pub fn update_tracer_properties(
    mesh: &Mesh,
    swarm: &mut TracerSwarm,
    materials: &[ElastoViscoPlastic],
    elem_mat_ids: &[u32],
    elem_props: &[ElementProperties],
    dt: f64,
) {
    let grid = SearchGrid::build(mesh, [10, 10, 10]);

    for i in 0..swarm.num_tracers() {
        let p_tracer = Point3::new(swarm.x[i], swarm.y[i], swarm.z[i]);
        let candidates = grid.get_potential_elements(p_tracer);

        for &elem_id in candidates {
            let elem = &mesh.connectivity.tet10_elements[elem_id];
            let mut vertices = [Point3::origin(); 4];
            for k in 0..4 {
                vertices[k] = mesh.geometry.nodes[elem.nodes[k]];
            }

            let l = Tet10Basis::cartesian_to_barycentric(&p_tracer, &vertices);
            if l.iter().all(|&val| val >= -1e-5 && val <= 1.0 + 1e-5) {
                let mat_id = elem_mat_ids[elem_id] as usize;
                let props = &elem_props[elem_id];

                // Update plastic strain if yielding is active
                // Accumulate whenever plastic viscosity is limiting (μ_eff < μ_v)
                if props.effective_viscosity < materials[mat_id].viscosity {
                    swarm.plastic_strain[i] += props.strain_rate_ii * dt;
                }

                // Update visualization properties
                swarm.strain_rate_ii[i] = props.strain_rate_ii;
                swarm.stress_ii[i] = props.stress_ii;
                swarm.viscosity[i] = props.effective_viscosity;
                swarm.pressure[i] = props.pressure;

                break; // Found containing element
            }
        }
    }
}

/// Advect tracers and update mesh nodes
///
/// # Arguments
/// * `mesh` - Computational mesh (nodes modified in place)
/// * `swarm` - Tracer swarm (modified in place)
/// * `dof_mgr` - DOF manager
/// * `current_sol` - Current velocity solution
/// * `dt` - Time step size (seconds)
pub fn advect_tracers_and_mesh(
    mesh: &mut Mesh,
    swarm: &mut TracerSwarm,
    dof_mgr: &DofManager,
    current_sol: &[f64],
    dt: f64,
) {
    // Build search grid for tracer advection
    let grid = SearchGrid::build(mesh, [10, 10, 10]);

    // Advect tracers using RK2
    swarm.advect_rk2(mesh, &grid, dof_mgr, current_sol, dt);

    // Update mesh node positions
    for (node_id, node) in mesh.geometry.nodes.iter_mut().enumerate() {
        node.x += current_sol[dof_mgr.velocity_dof(node_id, 0)] * dt;
        node.y += current_sol[dof_mgr.velocity_dof(node_id, 1)] * dt;
        node.z += current_sol[dof_mgr.velocity_dof(node_id, 2)] * dt;
    }
}
