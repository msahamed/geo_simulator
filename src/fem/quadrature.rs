/// Gaussian quadrature rules for tetrahedral elements
pub struct GaussQuadrature {
    /// Integration point coordinates in barycentric form [L0, L1, L2, L3]
    pub points: Vec<[f64; 4]>,
    /// Integration weights
    pub weights: Vec<f64>,
}

impl GaussQuadrature {
    /// 1-point quadrature (degree 1 exactness) - centroid rule
    ///
    /// Exact for linear polynomials
    pub fn tet_1point() -> Self {
        Self {
            points: vec![[0.25, 0.25, 0.25, 0.25]],
            weights: vec![1.0 / 6.0], // Volume of reference tet
        }
    }

    /// 4-point quadrature (degree 2 exactness)
    ///
    /// Exact for quadratic polynomials
    /// Good for Tet10 elements with linear problems
    pub fn tet_4point() -> Self {
        let a = 0.5854101966249685; // (5 + √5) / 20
        let b = 0.1381966011250105; // (5 - √5) / 20
        let w = 1.0 / 24.0; // 1/6 (tet volume) * 1/4 (symmetry)

        Self {
            points: vec![
                [a, b, b, b],
                [b, a, b, b],
                [b, b, a, b],
                [b, b, b, a],
            ],
            weights: vec![w, w, w, w],
        }
    }

    /// 5-point quadrature (degree 3 exactness)
    ///
    /// Exact for cubic polynomials
    /// Recommended for most Tet10 applications
    pub fn tet_5point() -> Self {
        let a = 0.25;
        let b = 1.0 / 6.0;
        let c = 0.5;

        Self {
            points: vec![
                [a, a, a, a],
                [b, b, b, c],
                [b, b, c, b],
                [b, c, b, b],
                [c, b, b, b],
            ],
            weights: vec![-2.0 / 15.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0],
        }
    }

    /// 11-point quadrature (degree 4 exactness)
    ///
    /// Exact for quartic polynomials
    /// High accuracy for demanding problems
    pub fn tet_11point() -> Self {
        let a1 = 0.25;
        let a2 = 0.0714285714285714;
        let b2 = 0.7857142857142857;
        let a3 = 0.3994035761667992;
        let b3 = 0.1005964238332008;

        Self {
            points: vec![
                // Central point
                [a1, a1, a1, a1],
                // 4 points near vertices
                [a2, a2, a2, b2],
                [a2, a2, b2, a2],
                [a2, b2, a2, a2],
                [b2, a2, a2, a2],
                // 6 points on edges
                [a3, a3, b3, b3],
                [a3, b3, a3, b3],
                [a3, b3, b3, a3],
                [b3, a3, a3, b3],
                [b3, a3, b3, a3],
                [b3, b3, a3, a3],
            ],
            weights: vec![
                -0.01315555555555556,  // -0.0789333.../6
                0.007622222222222222,  // 0.0457333.../6
                0.007622222222222222,
                0.007622222222222222,
                0.007622222222222222,
                0.024888888888888888,  // 0.1493333.../6
                0.024888888888888888,
                0.024888888888888888,
                0.024888888888888888,
                0.024888888888888888,
                0.024888888888888888,
            ],
        }
    }

    /// Get the number of integration points
    pub fn num_points(&self) -> usize {
        self.points.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_weights_sum() {
        // Weights should sum to volume of reference tetrahedron (1/6)
        let rules = vec![
            GaussQuadrature::tet_1point(),
            GaussQuadrature::tet_4point(),
            GaussQuadrature::tet_5point(),
            GaussQuadrature::tet_11point(),
        ];

        for rule in rules {
            let sum: f64 = rule.weights.iter().sum();
            assert_relative_eq!(sum, 1.0 / 6.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_barycentric_sum() {
        // Barycentric coordinates should sum to 1
        let rule = GaussQuadrature::tet_11point();

        for point in &rule.points {
            let sum: f64 = point.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_polynomial_exactness() {
        // Test that 4-point rule integrates quadratic exactly
        // ∫∫∫ x^2 dV over reference tet = 1/60
        // Using barycentric: x = L1, so ∫∫∫ L1^2 dV

        let rule = GaussQuadrature::tet_4point();
        let mut integral = 0.0;

        for (point, weight) in rule.points.iter().zip(rule.weights.iter()) {
            let L1 = point[1];
            integral += L1 * L1 * weight;
        }

        // Analytical: ∫∫∫ L1^2 dV = 1/60 for reference tet
        assert_relative_eq!(integral, 1.0 / 60.0, epsilon = 1e-14);
    }
}
