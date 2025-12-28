use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use crate::mesh::Mesh;

/// High-performance Structure-of-Arrays (SoA) for tracers.
/// Storing millions of particles efficiently.
#[derive(Debug, Clone)]
pub struct TracerSwarm {
    // Positions
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,

    // Velocities (at tracer position, interpolated from mesh)
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub vz: Vec<f64>,

    // Material Properties
    pub material_id: Vec<u32>,
    pub plastic_strain: Vec<f64>,

    // History
    pub initial_x: Vec<f64>,
    pub initial_y: Vec<f64>,
    pub initial_z: Vec<f64>,
}

impl TracerSwarm {
    /// Create a new, empty tracer swarm with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
            z: Vec::with_capacity(capacity),
            vx: vec![0.0; 0],
            vy: vec![0.0; 0],
            vz: vec![0.0; 0],
            material_id: Vec::with_capacity(capacity),
            plastic_strain: Vec::with_capacity(capacity),
            initial_x: Vec::with_capacity(capacity),
            initial_y: Vec::with_capacity(capacity),
            initial_z: Vec::with_capacity(capacity),
        }
    }

    /// Add a tracer at a specific position.
    pub fn add_tracer(&mut self, pos: Point3<f64>, mat_id: u32) {
        self.x.push(pos.x);
        self.y.push(pos.y);
        self.z.push(pos.z);
        self.initial_x.push(pos.x);
        self.initial_y.push(pos.y);
        self.initial_z.push(pos.z);
        self.material_id.push(mat_id);
        self.plastic_strain.push(0.0);
        
        // Ensure velocity vectors stay in sync if already populated
        if !self.vx.is_empty() {
            self.vx.push(0.0);
            self.vy.push(0.0);
            self.vz.push(0.0);
        }
    }

    /// Number of tracers in the swarm.
    pub fn num_tracers(&self) -> usize {
        self.x.len()
    }

    /// Initialize velocity vectors.
    pub fn init_velocities(&mut self) {
        let n = self.num_tracers();
        self.vx = vec![0.0; n];
        self.vy = vec![0.0; n];
        self.vz = vec![0.0; n];
    }
}

/// A spatial grid to speed up point-in-element searches.
/// O(1) average lookup.
pub struct SearchGrid {
    pub min: Point3<f64>,
    pub max: Point3<f64>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub cell_size: Vector3<f64>,
    /// Maps grid cell to a list of element IDs that overlap with it.
    pub cells: Vec<Vec<usize>>,
}

impl SearchGrid {
    /// Build a search grid for a given mesh.
    pub fn build(mesh: &Mesh, resolution: [usize; 3]) -> Self {
        // Compute bounding box
        let mut min = mesh.geometry.nodes[0];
        let mut max = mesh.geometry.nodes[0];
        for node in &mesh.geometry.nodes {
            min.x = min.x.min(node.x);
            min.y = min.y.min(node.y);
            min.z = min.z.min(node.z);
            max.x = max.x.max(node.x);
            max.y = max.y.max(node.y);
            max.z = max.z.max(node.z);
        }

        // Add small padding to avoid boundary issues
        let padding = 1e-4 * (max.x - min.x).max(max.y - min.y).max(max.z - min.z);
        min -= Vector3::new(padding, padding, padding);
        max += Vector3::new(padding, padding, padding);

        let cell_size = Vector3::new(
            (max.x - min.x) / resolution[0] as f64,
            (max.y - min.y) / resolution[1] as f64,
            (max.z - min.z) / resolution[2] as f64,
        );

        let n_cells = resolution[0] * resolution[1] * resolution[2];
        let mut cells = vec![Vec::new(); n_cells];

        // Populate cells with element IDs
        for (elem_id, _elem) in mesh.connectivity.tet10_elements.iter().enumerate() {
            // Get element bounding box (approxmate using corners for speed)
            let nodes = &mesh.connectivity.tet10_elements[elem_id].nodes;
            let mut e_min = mesh.geometry.nodes[nodes[0]];
            let mut e_max = mesh.geometry.nodes[nodes[0]];
            for i in 1..4 { // Corner nodes are 0..4
                let p = mesh.geometry.nodes[nodes[i]];
                e_min.x = e_min.x.min(p.x);
                e_min.y = e_min.y.min(p.y);
                e_min.z = e_min.z.min(p.z);
                e_max.x = e_max.x.max(p.x);
                e_max.y = e_max.y.max(p.y);
                e_max.z = e_max.z.max(p.z);
            }

            // Find range of grid cells overlapping with element BB
            let ix_start = ((e_min.x - min.x) / cell_size.x).floor() as usize;
            let iy_start = ((e_min.y - min.y) / cell_size.y).floor() as usize;
            let iz_start = ((e_min.z - min.z) / cell_size.z).floor() as usize;

            let ix_end = (((e_max.x - min.x) / cell_size.x).floor() as usize).min(resolution[0] - 1);
            let iy_end = (((e_max.y - min.y) / cell_size.y).floor() as usize).min(resolution[1] - 1);
            let iz_end = (((e_max.z - min.z) / cell_size.z).floor() as usize).min(resolution[2] - 1);

            for ix in ix_start..=ix_end {
                for iy in iy_start..=iy_end {
                    for iz in iz_start..=iz_end {
                        let cell_idx = ix + iy * resolution[0] + iz * resolution[0] * resolution[1];
                        cells[cell_idx].push(elem_id);
                    }
                }
            }
        }

        Self {
            min,
            max,
            nx: resolution[0],
            ny: resolution[1],
            nz: resolution[2],
            cell_size,
            cells,
        }
    }

    /// Get potential elements containing the point.
    pub fn get_potential_elements(&self, p: Point3<f64>) -> &[usize] {
        if p.x < self.min.x || p.x > self.max.x ||
           p.y < self.min.y || p.y > self.max.y ||
           p.z < self.min.z || p.z > self.max.z {
            return &[];
        }

        let ix = (((p.x - self.min.x) / self.cell_size.x).floor() as usize).min(self.nx - 1);
        let iy = (((p.y - self.min.y) / self.cell_size.y).floor() as usize).min(self.ny - 1);
        let iz = (((p.z - self.min.z) / self.cell_size.z).floor() as usize).min(self.nz - 1);

        let cell_idx = ix + iy * self.nx + iz * self.nx * self.ny;
        &self.cells[cell_idx]
    }
}
