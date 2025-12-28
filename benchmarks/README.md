# Benchmarks and Analytical Tests

This directory contains analytical validation tests for verifying the correctness of implementations.

## Structure

```
benchmarks/
├── README.md              # This file
├── shape_functions/       # Tet10 shape function tests
├── quadrature/            # Integration accuracy tests
├── element_matrices/      # Element stiffness/mass matrix tests
├── assembly/              # Global assembly verification
├── solvers/               # Linear solver accuracy tests
├── thermal/               # Thermal diffusion benchmarks
├── mechanics/             # Solid mechanics benchmarks
└── multi_physics/         # Coupled physics benchmarks
```

## Testing Philosophy

Each benchmark should:
1. **Have an analytical solution** - Closed-form or reference solution
2. **Test a specific feature** - Isolate what's being validated
3. **Quantify error** - Compute L2 norm, max error, convergence rate
4. **Be reproducible** - Fixed random seed, documented parameters

## Running Benchmarks

```bash
# Run all benchmarks
cargo test --release --test benchmarks

# Run specific benchmark
cargo test --release --test shape_functions

# Run with output
cargo test --release --test shape_functions -- --nocapture
```

## Benchmark Categories

### 1. Shape Functions (`shape_functions/`)
- Partition of unity: Σ N_i = 1
- Kronecker delta property: N_i(x_j) = δ_ij
- Derivative accuracy
- Reference element vs physical element

### 2. Quadrature (`quadrature/`)
- Polynomial exactness
- Integration of monomials x^p y^q z^r
- Volume calculation

### 3. Element Matrices (`element_matrices/`)
- Stiffness matrix symmetry
- Positive definiteness
- Known analytical element matrices

### 4. Thermal Diffusion (`thermal/`)
- 1D steady-state (linear profile)
- 2D/3D with analytical solution (method of manufactured solutions)
- Convergence rate (should be O(h^3) for Tet10)

### 5. Solid Mechanics (`mechanics/`)
- Patch test (constant stress/strain)
- Gravity loading (analytical solution)
- Benchmark problems from literature

### 6. Multi-Physics (`multi_physics/`)
- Blankenbach convection benchmarks
- Subduction zone benchmarks
- Comparison with DynEarthSol3D

## Acceptance Criteria

Each implementation must pass:
- **Unit tests** - Individual functions work correctly
- **Analytical benchmarks** - Match analytical solution within tolerance
- **Convergence tests** - Achieve expected convergence rate with mesh refinement

## Adding New Benchmarks

1. Create file: `benchmarks/<category>/<test_name>.rs`
2. Implement test with clear documentation
3. Add expected result and tolerance
4. Update this README
