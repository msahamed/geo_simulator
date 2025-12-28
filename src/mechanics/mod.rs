/// Solid mechanics module for linear elasticity and rheology
///
/// This module provides implementations for:
/// - Linear elastic constitutive models
/// - Strain-displacement relationships
/// - Element matrices for displacement-based FEM
/// - Body force computations (gravity, surface tractions)

pub mod constitutive;
pub mod strain;
pub mod element;
pub mod body_force;
pub mod stress_update;
pub mod plasticity;

pub use constitutive::{IsotropicElasticity, NewtonianViscosity, MaxwellViscoelasticity};
pub use strain::StrainDisplacement;
pub use element::ElasticityElement;
pub use body_force::BodyForce;
pub use stress_update::update_stresses_maxwell;
pub use plasticity::{DruckerPrager, ElastoViscoPlastic};
