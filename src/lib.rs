pub mod mesh;
pub mod mesh_generator;
pub mod mesh_generator_improved;
pub mod physics;
pub mod fem;
pub mod linalg;
pub mod mechanics;

pub use mesh::{Mesh, Tet10Element, VtkWriter, ScalarField, VectorField};
pub use mesh_generator::MeshGenerator;
pub use mesh_generator_improved::ImprovedMeshGenerator;
pub use physics::{ThermalField, PressureField};
pub use fem::{Tet10Basis, GaussQuadrature, DofManager, ElementMatrix, Assembler, BoundaryConditions, BoundaryFace};
pub use linalg::{Solver, DirectSolver, ConjugateGradient};
pub use mechanics::{IsotropicElasticity, NewtonianViscosity, StrainDisplacement, ElasticityElement, BodyForce};
