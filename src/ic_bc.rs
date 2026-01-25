//! Initial and Boundary Condition Setup (Compatibility Layer)
//!
//! **Note**: This module has been split into separate `ic` and `bc` modules for better organization.
//! This file re-exports functionality from those modules for backwards compatibility.
//!
//! ## Migration Guide
//! - For boundary conditions: use `crate::bc::*` instead of `crate::ic_bc::*`
//! - For initial conditions: use `crate::ic::*` instead of `crate::ic_bc::*`
//!
//! ## New Features
//! The split modules provide additional functionality:
//! - `bc`: BoundaryType enum, BoundaryNodes caching, ramp fraction calculation
//! - `ic`: Lithostatic pressure, extension velocity, geotherm initialization

// Re-export boundary condition functions from bc module
pub use crate::bc::{
    setup_boundary_conditions,
    find_boundary_nodes,
    print_bc_summary,
    BoundaryType,
    BoundaryNodes,
    compute_ramp_fraction,
};

// Re-export initial condition functions from ic module
pub use crate::ic::{
    setup_initial_tracers,
    get_material_properties,
    MaterialProps,
    initialize_lithostatic_pressure,
    initialize_extension_velocity,
    initialize_geotherm,
    get_gravity_vector,
};
