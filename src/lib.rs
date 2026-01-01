pub mod mesh;
pub mod mesh_generator;
pub mod mesh_generator_improved;
pub mod physics;
pub mod fem;
pub mod linalg;
pub mod mechanics;

pub use mesh::{Mesh, Tet10Element, VtkWriter, ScalarField, VectorField, StressHistory, PlasticityState, TracerSwarm, SearchGrid, MeshQuality, assess_mesh_quality, smooth_mesh_auto, smooth_laplacian, build_node_neighbors, compute_tet_jacobian};
pub use mesh_generator::MeshGenerator;
pub use mesh_generator_improved::ImprovedMeshGenerator;
pub use physics::{ThermalField, PressureField, HillslopeDiffusion};
pub use fem::{Tet10Basis, GaussQuadrature, DofManager, ElementMatrix, Assembler, BoundaryConditions, BoundaryFace, BackwardEuler, TimeStepStats};
pub use linalg::{Solver, DirectSolver, ConjugateGradient, BiCGSTAB, GMRES, picard_solve, PicardConfig, PicardStats, jfnk_solve, JFNKConfig, JFNKStats};
pub use mechanics::{IsotropicElasticity, NewtonianViscosity, MaxwellViscoelasticity, StrainDisplacement, ElasticityElement, BodyForce, update_stresses_maxwell, DruckerPrager, ElastoViscoPlastic, WinklerFoundation};
