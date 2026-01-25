//! Unit conversion utilities for geodynamic simulations
//!
//! This module provides constants and conversion functions for commonly used
//! units in geodynamic modeling, eliminating magic numbers and ensuring
//! consistent conversions throughout the codebase.

// ============================================================================
// Time Conversions
// ============================================================================

/// Seconds per year (365.25 days accounting for leap years)
pub const SECONDS_PER_YEAR: f64 = 365.25 * 24.0 * 3600.0;

/// Seconds per million years
pub const SECONDS_PER_MYR: f64 = SECONDS_PER_YEAR * 1e6;

/// Seconds per thousand years (kiloyear)
pub const SECONDS_PER_KYR: f64 = SECONDS_PER_YEAR * 1e3;

/// Convert years to seconds
///
/// # Examples
/// ```
/// use geo_simulator::utils::units::years_to_seconds;
/// let dt_sec = years_to_seconds(1000.0); // 1000 years
/// ```
#[inline]
pub fn years_to_seconds(years: f64) -> f64 {
    years * SECONDS_PER_YEAR
}

/// Convert seconds to years
///
/// # Examples
/// ```
/// use geo_simulator::utils::units::seconds_to_years;
/// let dt_years = seconds_to_years(3.15e10); // ~1000 years
/// ```
#[inline]
pub fn seconds_to_years(seconds: f64) -> f64 {
    seconds / SECONDS_PER_YEAR
}

/// Convert million years to seconds
#[inline]
pub fn myr_to_seconds(myr: f64) -> f64 {
    myr * SECONDS_PER_MYR
}

/// Convert seconds to million years
#[inline]
pub fn seconds_to_myr(seconds: f64) -> f64 {
    seconds / SECONDS_PER_MYR
}

/// Convert thousand years (kiloyears) to seconds
#[inline]
pub fn kyr_to_seconds(kyr: f64) -> f64 {
    kyr * SECONDS_PER_KYR
}

/// Convert seconds to thousand years (kiloyears)
#[inline]
pub fn seconds_to_kyr(seconds: f64) -> f64 {
    seconds / SECONDS_PER_KYR
}

// ============================================================================
// Pressure Conversions
// ============================================================================

/// Pascals to megapascals conversion factor
pub const PA_TO_MPA: f64 = 1e-6;

/// Megapascals to pascals conversion factor
pub const MPA_TO_PA: f64 = 1e6;

/// Pascals to gigapascals conversion factor
pub const PA_TO_GPA: f64 = 1e-9;

/// Gigapascals to pascals conversion factor
pub const GPA_TO_PA: f64 = 1e9;

/// Convert pascals to megapascals
///
/// # Examples
/// ```
/// use geo_simulator::utils::units::pa_to_mpa;
/// let pressure_mpa = pa_to_mpa(44e6); // 44 MPa
/// ```
#[inline]
pub fn pa_to_mpa(pa: f64) -> f64 {
    pa * PA_TO_MPA
}

/// Convert megapascals to pascals
///
/// # Examples
/// ```
/// use geo_simulator::utils::units::mpa_to_pa;
/// let cohesion_pa = mpa_to_pa(44.0); // 44 MPa cohesion
/// ```
#[inline]
pub fn mpa_to_pa(mpa: f64) -> f64 {
    mpa * MPA_TO_PA
}

/// Convert pascals to gigapascals
#[inline]
pub fn pa_to_gpa(pa: f64) -> f64 {
    pa * PA_TO_GPA
}

/// Convert gigapascals to pascals
#[inline]
pub fn gpa_to_pa(gpa: f64) -> f64 {
    gpa * GPA_TO_PA
}

// ============================================================================
// Length Conversions
// ============================================================================

/// Meters to kilometers conversion factor
pub const M_TO_KM: f64 = 1e-3;

/// Kilometers to meters conversion factor
pub const KM_TO_M: f64 = 1e3;

/// Meters to centimeters conversion factor
pub const M_TO_CM: f64 = 1e2;

/// Centimeters to meters conversion factor
pub const CM_TO_M: f64 = 1e-2;

/// Convert meters to kilometers
///
/// # Examples
/// ```
/// use geo_simulator::utils::units::m_to_km;
/// let domain_width_km = m_to_km(100_000.0); // 100 km
/// ```
#[inline]
pub fn m_to_km(m: f64) -> f64 {
    m * M_TO_KM
}

/// Convert kilometers to meters
///
/// # Examples
/// ```
/// use geo_simulator::utils::units::km_to_m;
/// let domain_width_m = km_to_m(100.0); // 100 km = 100,000 m
/// ```
#[inline]
pub fn km_to_m(km: f64) -> f64 {
    km * KM_TO_M
}

/// Convert meters to centimeters
#[inline]
pub fn m_to_cm(m: f64) -> f64 {
    m * M_TO_CM
}

/// Convert centimeters to meters
#[inline]
pub fn cm_to_m(cm: f64) -> f64 {
    cm * CM_TO_M
}

/// Convert centimeters per year to meters per second
///
/// # Examples
/// ```
/// use geo_simulator::utils::units::cm_per_year_to_m_per_s;
/// let extension_rate = cm_per_year_to_m_per_s(3.15); // π cm/yr ≈ 1e-9 m/s
/// ```
#[inline]
pub fn cm_per_year_to_m_per_s(cm_per_year: f64) -> f64 {
    cm_per_year * CM_TO_M / SECONDS_PER_YEAR
}

/// Convert meters per second to centimeters per year
#[inline]
pub fn m_per_s_to_cm_per_year(m_per_s: f64) -> f64 {
    m_per_s * M_TO_CM * SECONDS_PER_YEAR
}

// ============================================================================
// Angle Conversions
// ============================================================================

/// Degrees to radians conversion factor
pub const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

/// Radians to degrees conversion factor
pub const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;

/// Convert degrees to radians
///
/// # Examples
/// ```
/// use geo_simulator::utils::units::deg_to_rad;
/// let friction_angle_rad = deg_to_rad(30.0); // 30° friction angle
/// ```
#[inline]
pub fn deg_to_rad(degrees: f64) -> f64 {
    degrees * DEG_TO_RAD
}

/// Convert radians to degrees
#[inline]
pub fn rad_to_deg(radians: f64) -> f64 {
    radians * RAD_TO_DEG
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_conversions() {
        // Test years <-> seconds
        let years = 1000.0;
        let seconds = years_to_seconds(years);
        assert!((seconds_to_years(seconds) - years).abs() < 1e-10);

        // Test Myr <-> seconds
        let myr = 2.0;
        let seconds = myr_to_seconds(myr);
        assert!((seconds_to_myr(seconds) - myr).abs() < 1e-10);

        // Test kyr <-> seconds
        let kyr = 50.0;
        let seconds = kyr_to_seconds(kyr);
        assert!((seconds_to_kyr(seconds) - kyr).abs() < 1e-10);

        // Verify constants
        assert!((SECONDS_PER_YEAR - 31_557_600.0).abs() < 1.0);
        assert!((SECONDS_PER_MYR - 3.15576e13).abs() < 1e8);
    }

    #[test]
    fn test_pressure_conversions() {
        // Test Pa <-> MPa
        let mpa = 44.0;
        let pa = mpa_to_pa(mpa);
        assert!((pa_to_mpa(pa) - mpa).abs() < 1e-10);
        assert_eq!(pa, 44e6);

        // Test Pa <-> GPa
        let gpa = 30.0;
        let pa = gpa_to_pa(gpa);
        assert!((pa_to_gpa(pa) - gpa).abs() < 1e-10);
        assert_eq!(pa, 30e9);
    }

    #[test]
    fn test_length_conversions() {
        // Test m <-> km
        let km = 100.0;
        let m = km_to_m(km);
        assert!((m_to_km(m) - km).abs() < 1e-10);
        assert_eq!(m, 100_000.0);

        // Test m <-> cm
        let cm = 3.15;
        let m = cm_to_m(cm);
        assert!((m_to_cm(m) - cm).abs() < 1e-10);
        assert_eq!(m, 0.0315);
    }

    #[test]
    fn test_velocity_conversions() {
        // Test cm/yr <-> m/s
        let cm_per_yr = 3.15;
        let m_per_s = cm_per_year_to_m_per_s(cm_per_yr);

        // π cm/yr ≈ 1e-9 m/s
        assert!((m_per_s - 1e-9).abs() < 1e-10);

        // Round trip
        let cm_per_yr_back = m_per_s_to_cm_per_year(m_per_s);
        assert!((cm_per_yr_back - cm_per_yr).abs() < 1e-10);
    }

    #[test]
    fn test_angle_conversions() {
        // Test deg <-> rad
        let deg = 30.0;
        let rad = deg_to_rad(deg);
        assert!((rad_to_deg(rad) - deg).abs() < 1e-10);

        // Common angles
        assert!((deg_to_rad(180.0) - std::f64::consts::PI).abs() < 1e-10);
        assert!((deg_to_rad(90.0) - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        assert!((deg_to_rad(45.0) - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    }
}
