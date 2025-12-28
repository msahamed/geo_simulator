use crate::mesh::Mesh;

/// Thermal field computation
pub struct ThermalField;

impl ThermalField {
    /// Compute temperature field with a simple geothermal gradient
    ///
    /// T(z) = T_surface + dT/dz * depth
    ///
    /// # Arguments
    /// * `mesh` - The mesh to compute temperature for
    /// * `t_surface` - Surface temperature (°C)
    /// * `geothermal_gradient` - Temperature gradient (°C/km)
    /// * `max_depth` - Maximum depth of the domain (km) - z values are measured from top
    ///
    /// # Returns
    /// Vector of temperatures at each node (°C)
    pub fn compute_geothermal_gradient(
        mesh: &Mesh,
        t_surface: f64,
        geothermal_gradient: f64,
        max_depth: f64,
    ) -> Vec<f64> {
        mesh.geometry
            .nodes
            .iter()
            .map(|node| {
                // Depth is measured from the top surface (z = max_depth is surface, z = 0 is bottom)
                let depth = max_depth - node.z;
                t_surface + geothermal_gradient * depth
            })
            .collect()
    }

    /// Compute temperature field with depth-dependent gradient
    /// (more realistic - gradient decreases in the mantle)
    ///
    /// # Arguments
    /// * `mesh` - The mesh
    /// * `t_surface` - Surface temperature (°C)
    /// * `gradient_crust` - Gradient in crust (°C/km)
    /// * `gradient_mantle` - Gradient in mantle (°C/km)
    /// * `moho_depth` - Depth to Moho discontinuity (km)
    /// * `max_depth` - Maximum depth (km)
    pub fn compute_layered_gradient(
        mesh: &Mesh,
        t_surface: f64,
        gradient_crust: f64,
        gradient_mantle: f64,
        moho_depth: f64,
        max_depth: f64,
    ) -> Vec<f64> {
        // Temperature at Moho
        let t_moho = t_surface + gradient_crust * moho_depth;

        mesh.geometry
            .nodes
            .iter()
            .map(|node| {
                let depth = max_depth - node.z;

                if depth <= moho_depth {
                    // In crust
                    t_surface + gradient_crust * depth
                } else {
                    // In mantle
                    t_moho + gradient_mantle * (depth - moho_depth)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geothermal_gradient() {
        // Create a simple mesh for testing
        let mut mesh = Mesh::new();
        mesh.geometry.add_node(0.0, 0.0, 0.0);   // Bottom
        mesh.geometry.add_node(0.0, 0.0, 300.0); // Middle
        mesh.geometry.add_node(0.0, 0.0, 600.0); // Top (surface)

        let temps = ThermalField::compute_geothermal_gradient(&mesh, 0.0, 25.0, 600.0);

        assert_eq!(temps.len(), 3);
        assert!((temps[2] - 0.0).abs() < 1e-10);     // Surface: 0°C
        assert!((temps[1] - 300.0 * 25.0).abs() < 1e-10); // Middle: 7500°C
        assert!((temps[0] - 600.0 * 25.0).abs() < 1e-10); // Bottom: 15000°C
    }
}
