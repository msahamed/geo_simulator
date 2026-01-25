//! Time-stepping strategies for geodynamic simulations
//!
//! This module provides various time-stepping strategies including
//! adaptive timestep computation and fixed timestep approaches.

pub mod adaptive;

// Re-export commonly used items
pub use adaptive::{AdaptiveTimestep, compute_adaptive_timestep};
