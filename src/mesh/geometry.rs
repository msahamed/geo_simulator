use nalgebra::Point3;
use super::topology::Connectivity;
use super::fields::FieldData;
use super::state::StressHistory;

/// Geometric information for the mesh
#[derive(Debug, Clone)]
pub struct Geometry {
    /// Node coordinates
    pub nodes: Vec<Point3<f64>>,
}

impl Geometry {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, x: f64, y: f64, z: f64) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Point3::new(x, y, z));
        idx
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn get_node(&self, idx: usize) -> Option<&Point3<f64>> {
        self.nodes.get(idx)
    }
}

impl Default for Geometry {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete mesh with geometry and topology
#[derive(Debug, Clone)]
pub struct Mesh {
    pub geometry: Geometry,
    pub connectivity: Connectivity,
    pub field_data: FieldData,
    /// Stress history for viscoelastic simulations
    pub stress_history: Option<StressHistory>,
    /// Plasticity state for strain softening
    pub plasticity_state: Option<super::state::PlasticityState>,
}

impl Mesh {
    pub fn new() -> Self {
        Self {
            geometry: Geometry::new(),
            connectivity: Connectivity::new(),
            field_data: FieldData::new(),
            stress_history: None,
            plasticity_state: None,
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.geometry.num_nodes()
    }

    pub fn num_elements(&self) -> usize {
        self.connectivity.num_elements()
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}
