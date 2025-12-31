pub mod topology;
pub mod geometry;
pub mod vtk_writer;
pub mod fields;
pub mod state;
pub mod tracers;
pub mod quality;

pub use topology::Tet10Element;
pub use geometry::Mesh;
pub use vtk_writer::VtkWriter;
pub use fields::{ScalarField, VectorField, FieldData};
pub use state::{StressHistory, PlasticityState};
pub use tracers::{TracerSwarm, SearchGrid};
pub use quality::{MeshQuality, assess_mesh_quality, smooth_mesh_auto, smooth_laplacian, build_node_neighbors, compute_tet_jacobian};
