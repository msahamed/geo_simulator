use geo_simulator::{NewtonianViscosity};
use nalgebra::SMatrix;
use approx::assert_relative_eq;

#[test]
fn test_incompressibility_penalty() {
    let mu = 1000.0;
    let mut mat = NewtonianViscosity::new(mu);
    
    // 1. Deviatoric-only case (bulk_viscosity = 0)
    let D_pure = mat.constitutive_matrix();
    
    // Pure compression strain rate: ε̇ = [1, 1, 1, 0, 0, 0]
    let e_comp = SMatrix::<f64, 6, 1>::from_column_slice(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    let sigma_pure = D_pure * e_comp;
    
    // For pure deviatoric material, pure compression should result in ZERO stress 
    // (since ε̇_dev = 0 for ε̇_xx=ε̇_yy=ε̇_zz)
    // Wait, let's verify: ε̇_dev = ε̇ - 1/3 tr(ε̇) I = [1, 1, 1] - 1/3(3)[1, 1, 1] = [0, 0, 0].
    // Correct!
    assert_relative_eq!(sigma_pure[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(sigma_pure[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(sigma_pure[2], 0.0, epsilon = 1e-10);

    // 2. Incompressible penalty case
    let zeta = 1e6 * mu; // Large penalty
    mat = mat.with_penalty(zeta);
    let D_pen = mat.constitutive_matrix();
    let sigma_pen = D_pen * e_comp;

    // Now pure compression should result in large stress: σ_ii = ζ * tr(ε̇) = 3 * ζ
    let expected_p = zeta * 3.0;
    assert_relative_eq!(sigma_pen[0], expected_p, epsilon = 1e-3);
    assert_relative_eq!(sigma_pen[1], expected_p, epsilon = 1e-3);
    assert_relative_eq!(sigma_pen[2], expected_p, epsilon = 1e-3);

    // 3. Pure shear case (tr(ε̇) = 0)
    // ε̇ = [1, -1, 0, 0, 0, 0]
    let e_shear = SMatrix::<f64, 6, 1>::from_column_slice(&[1.0, -1.0, 0.0, 0.0, 0.0, 0.0]);
    let sigma_shear_pen = D_pen * e_shear;
    let sigma_shear_pure = D_pure * e_shear;

    // Penalty should not affect pure shear
    assert_relative_eq!(sigma_shear_pen[0], sigma_shear_pure[0], epsilon = 1e-10);
    assert_relative_eq!(sigma_shear_pen[1], sigma_shear_pure[1], epsilon = 1e-10);
}
