use crate::mesh::Mesh;

/// Pressure field computation
pub struct PressureField;

impl PressureField {
    /// Compute lithostatic pressure field
    ///
    /// P(z) = ρ × g × depth
    ///
    /// # Arguments
    /// * `mesh` - The mesh to compute pressure for
    /// * `density` - Rock density (kg/m³), typical: 2700-3300
    /// * `gravity` - Gravitational acceleration (m/s²), Earth: 9.81
    /// * `max_depth` - Maximum depth of domain (km)
    ///
    /// # Returns
    /// Vector of pressures at each node (Pa)
    pub fn compute_lithostatic(
        mesh: &Mesh,
        density: f64,
        gravity: f64,
        max_depth: f64,
    ) -> Vec<f64> {
        mesh.geometry
            .nodes
            .iter()
            .map(|node| {
                // Depth in km, convert to meters
                let depth = (max_depth - node.z) * 1000.0; // km to m

                // P = ρ × g × h (Pa)
                density * gravity * depth
            })
            .collect()
    }

    /// Compute lithostatic pressure with layered density structure
    ///
    /// # Arguments
    /// * `mesh` - The mesh
    /// * `density_crust` - Crustal density (kg/m³), typical: 2700
    /// * `density_mantle` - Mantle density (kg/m³), typical: 3300
    /// * `gravity` - Gravitational acceleration (m/s²)
    /// * `moho_depth` - Depth to Moho (km)
    /// * `max_depth` - Maximum depth (km)
    ///
    /// # Returns
    /// Vector of pressures at each node (Pa)
    pub fn compute_layered_lithostatic(
        mesh: &Mesh,
        density_crust: f64,
        density_mantle: f64,
        gravity: f64,
        moho_depth: f64,
        max_depth: f64,
    ) -> Vec<f64> {
        // Pressure at Moho
        let p_moho = density_crust * gravity * (moho_depth * 1000.0);

        mesh.geometry
            .nodes
            .iter()
            .map(|node| {
                let depth = max_depth - node.z; // km

                if depth <= moho_depth {
                    // In crust
                    density_crust * gravity * (depth * 1000.0)
                } else {
                    // In mantle: P = P_moho + ρ_mantle × g × (depth - moho_depth)
                    p_moho + density_mantle * gravity * ((depth - moho_depth) * 1000.0)
                }
            })
            .collect()
    }

    /// Convert pressure from Pa to GPa
    pub fn pa_to_gpa(pressure_pa: f64) -> f64 {
        pressure_pa / 1.0e9
    }

    /// Convert pressure from Pa to MPa
    pub fn pa_to_mpa(pressure_pa: f64) -> f64 {
        pressure_pa / 1.0e6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lithostatic_pressure() {
        let mut mesh = Mesh::new();
        mesh.geometry.add_node(0.0, 0.0, 0.0);   // Bottom: 600 km deep
        mesh.geometry.add_node(0.0, 0.0, 600.0); // Top: surface

        let density = 3300.0; // kg/m³
        let gravity = 9.81;   // m/s²
        let max_depth = 600.0; // km

        let pressures = PressureField::compute_lithostatic(&mesh, density, gravity, max_depth);

        assert_eq!(pressures.len(), 2);

        // Surface pressure should be ~0
        assert!(pressures[1].abs() < 1e-6);

        // Bottom pressure: ρ × g × h = 3300 × 9.81 × 600,000 m
        let expected_bottom = 3300.0 * 9.81 * 600_000.0;
        assert!((pressures[0] - expected_bottom).abs() < 1.0);

        // Convert to GPa for sanity check (~19.4 GPa at 600 km)
        let p_gpa = PressureField::pa_to_gpa(pressures[0]);
        assert!(p_gpa > 19.0 && p_gpa < 20.0);
    }
}
