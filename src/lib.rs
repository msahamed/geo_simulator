pub mod mesh;
pub mod mesh_generator;
pub mod mesh_generator_improved;
pub mod physics;
pub mod fem;
pub mod linalg;
pub mod mechanics;
pub mod config;
pub mod ic;     // Initial conditions
pub mod bc;     // Boundary conditions
pub mod updates;
pub mod utils;
pub mod timestepping;

pub use mesh::{Mesh, Tet10Element, VtkWriter, VtkOutputBuilder, VtkFormat, quick_write_vtu, quick_write_vtu_with_tracers, ScalarField, VectorField, StressHistory, PlasticityState, TracerSwarm, SearchGrid, MeshQuality, assess_mesh_quality, smooth_mesh_auto, smooth_laplacian, build_node_neighbors, compute_tet_jacobian};
pub use mesh_generator::MeshGenerator;
pub use mesh_generator_improved::ImprovedMeshGenerator;
pub use physics::{ThermalField, PressureField, HillslopeDiffusion};
pub use fem::{Tet10Basis, GaussQuadrature, DofManager, ElementMatrix, Assembler, BoundaryConditions, BoundaryFace, BackwardEuler, TimeStepStats};
pub use linalg::{Solver, DirectSolver, ConjugateGradient, BiCGSTAB, GMRES, picard_solve, PicardConfig, PicardStats, AndersonAccelerator, jfnk_solve, jfnk_solve_nondimensional, JFNKConfig, JFNKStats, CharacteristicScales, AMGPreconditioner};
pub use mechanics::{IsotropicElasticity, NewtonianViscosity, MaxwellViscoelasticity, StrainDisplacement, ElasticityElement, BodyForce, update_stresses_maxwell, DruckerPrager, ElastoViscoPlastic, WinklerFoundation};
pub use config::SimulationConfig;
pub use updates::{ElementProperties, VisualizationFields, compute_element_properties, compute_visualization_fields, update_tracer_properties, advect_tracers_and_mesh};
pub use timestepping::{AdaptiveTimestep, compute_adaptive_timestep};
pub use utils::units;
