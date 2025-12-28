/// Body force computations for solid mechanics
///
/// Implements gravitational and other volumetric force contributions.

use nalgebra::{Point3, SVector, Vector3};
use crate::fem::{Tet10Basis, GaussQuadrature};

/// Body force vector computations
pub struct BodyForce;

impl BodyForce {
    /// Compute body force vector for gravity
    ///
    /// f_e = ∫ N^T ρg dV
    ///
    /// where:
    /// - N are shape functions (10×1 for each node)
    /// - ρ is material density (kg/m³)
    /// - g is gravity vector (m/s²)
    /// - Result is distributed to all 3 DOFs at each node
    ///
    /// # Arguments
    /// * `vertices` - Physical coordinates of the 4 element vertices
    /// * `density` - Material density ρ (kg/m³)
    /// * `gravity` - Gravity acceleration vector g (m/s²)
    ///               Typically [0, 0, -9.81] for z-up convention
    ///
    /// # Returns
    /// 30×1 element load vector: [f_0x, f_0y, f_0z, ..., f_9x, f_9y, f_9z]
    ///
    /// # Example
    /// ```
    /// use geo_simulator::mechanics::BodyForce;
    /// use nalgebra::{Point3, Vector3};
    ///
    /// let vertices = [
    ///     Point3::new(0.0, 0.0, 0.0),
    ///     Point3::new(1.0, 0.0, 0.0),
    ///     Point3::new(0.0, 1.0, 0.0),
    ///     Point3::new(0.0, 0.0, 1.0),
    /// ];
    /// let density = 3000.0;  // kg/m³
    /// let gravity = Vector3::new(0.0, 0.0, -9.81);  // m/s²
    ///
    /// let f = BodyForce::gravity_load(&vertices, density, &gravity);
    /// // f now contains gravitational force distributed to element nodes
    /// ```
    #[allow(non_snake_case)]
    pub fn gravity_load(
        vertices: &[Point3<f64>; 4],
        density: f64,
        gravity: &Vector3<f64>,
    ) -> SVector<f64, 30> {
        let mut f_elem = SVector::<f64, 30>::zeros();

        // Use 4-point Gaussian quadrature
        let quad = GaussQuadrature::tet_4point();

        // Compute Jacobian determinant for volume transformation
        let J = Tet10Basis::jacobian(vertices);
        let det_J = J.determinant();

        // Body force per unit volume (force density)
        let body_force = density * gravity;

        // Integrate over element
        for (qp, weight) in quad.points.iter().zip(quad.weights.iter()) {
            // Evaluate shape functions at quadrature point
            let N = Tet10Basis::shape_functions(qp);

            // Integration weight (includes volume element)
            let w = weight * det_J.abs();

            // Distribute force to each node's 3 DOFs
            for i in 0..10 {
                // Node i has DOFs at indices [3*i, 3*i+1, 3*i+2]
                f_elem[3 * i + 0] += w * N[i] * body_force.x;
                f_elem[3 * i + 1] += w * N[i] * body_force.y;
                f_elem[3 * i + 2] += w * N[i] * body_force.z;
            }
        }

        f_elem
    }

    /// Compute element volume (utility function)
    ///
    /// V = |det(J)| / 6
    ///
    /// Useful for validating that total force equals weight.
    pub fn element_volume(vertices: &[Point3<f64>; 4]) -> f64 {
        Tet10Basis::element_volume(vertices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_unit_tet() -> [Point3<f64>; 4] {
        [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ]
    }

    #[test]
    fn test_gravity_load_total_equals_weight() {
        // Total force should equal element weight: ρ V g

        let vertices = create_unit_tet();
        let density = 3000.0;  // kg/m³
        let g_mag = 9.81;      // m/s²
        let gravity = Vector3::new(0.0, 0.0, -g_mag);

        let f = BodyForce::gravity_load(&vertices, density, &gravity);

        // Sum all z-components (vertical forces)
        let total_fz: f64 = (0..10).map(|i| f[3 * i + 2]).sum();

        // Expected: -ρ V g (negative because gravity points down)
        let volume = BodyForce::element_volume(&vertices);
        let expected_weight = -density * volume * g_mag;

        assert_relative_eq!(total_fz, expected_weight, epsilon = 1e-6);
    }

    #[test]
    fn test_gravity_load_dimensions() {
        let vertices = create_unit_tet();
        let density = 3000.0;
        let gravity = Vector3::new(0.0, 0.0, -9.81);

        let f = BodyForce::gravity_load(&vertices, density, &gravity);

        assert_eq!(f.len(), 30);
    }

    #[test]
    fn test_gravity_load_x_direction() {
        // Gravity in x-direction should only affect x-components

        let vertices = create_unit_tet();
        let density = 3000.0;
        let gravity = Vector3::new(9.81, 0.0, 0.0);

        let f = BodyForce::gravity_load(&vertices, density, &gravity);

        // Sum forces in each direction
        let total_fx: f64 = (0..10).map(|i| f[3 * i + 0]).sum();
        let total_fy: f64 = (0..10).map(|i| f[3 * i + 1]).sum();
        let total_fz: f64 = (0..10).map(|i| f[3 * i + 2]).sum();

        let volume = BodyForce::element_volume(&vertices);
        let expected = density * volume * 9.81;

        assert_relative_eq!(total_fx, expected, epsilon = 1e-6);
        assert_relative_eq!(total_fy, 0.0, epsilon = 1e-10);
        assert_relative_eq!(total_fz, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_gravity() {
        let vertices = create_unit_tet();
        let density = 3000.0;
        let gravity = Vector3::new(0.0, 0.0, 0.0);

        let f = BodyForce::gravity_load(&vertices, density, &gravity);

        // All components should be zero
        for i in 0..30 {
            assert_relative_eq!(f[i], 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_force_distribution() {
        // Forces should be distributed to all nodes (not concentrated)

        let vertices = create_unit_tet();
        let density = 3000.0;
        let gravity = Vector3::new(0.0, 0.0, -9.81);

        let f = BodyForce::gravity_load(&vertices, density, &gravity);

        // All nodes should have some non-zero z-force
        for i in 0..10 {
            let fz = f[3 * i + 2];
            assert!(fz.abs() > 1e-10, "Node {} should have non-zero force", i);
        }
    }

    #[test]
    fn test_element_volume() {
        // Unit tetrahedron should have volume = 1/6
        let vertices = create_unit_tet();
        let volume = BodyForce::element_volume(&vertices);

        assert_relative_eq!(volume, 1.0 / 6.0, epsilon = 1e-10);
    }
}
