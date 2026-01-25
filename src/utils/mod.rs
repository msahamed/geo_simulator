//! Utility modules for geodynamic simulations
//!
//! This module contains helper functions and utilities that are used
//! throughout the codebase.

pub mod units;

// Re-export commonly used items
pub use units::{
    years_to_seconds, seconds_to_years,
    myr_to_seconds, seconds_to_myr,
    kyr_to_seconds, seconds_to_kyr,
    pa_to_mpa, mpa_to_pa,
    pa_to_gpa, gpa_to_pa,
    m_to_km, km_to_m,
    m_to_cm, cm_to_m,
    cm_per_year_to_m_per_s, m_per_s_to_cm_per_year,
    deg_to_rad, rad_to_deg,
};
