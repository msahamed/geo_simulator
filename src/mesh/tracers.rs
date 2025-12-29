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
    pub stress_ii: Vec<f64>,
    pub strain_rate_ii: Vec<f64>,
    pub viscosity: Vec<f64>,
    pub pressure: Vec<f64>,

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
            stress_ii: Vec::with_capacity(capacity),
            strain_rate_ii: Vec::with_capacity(capacity),
            viscosity: Vec::with_capacity(capacity),
            pressure: Vec::with_capacity(capacity),
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
        self.stress_ii.push(0.0);
        self.strain_rate_ii.push(0.0);
        self.viscosity.push(0.0);
        self.pressure.push(0.0);
        
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

    /// Advect tracers using Runge-Kutta 2 (Midpoint method).
    ///
    /// # Arguments
    /// * `mesh` - The mesh
    /// * `grid` - Spatial search grid
    /// * `dof_mgr` - DOF manager
    /// * `velocity_vec` - Global nodal velocity vector
    /// * `dt` - Time step
    pub fn advect_rk2(
        &mut self,
        mesh: &Mesh,
        grid: &SearchGrid,
        dof_mgr: &crate::fem::DofManager,
        velocity_vec: &[f64],
        dt: f64,
    ) {
        let n = self.num_tracers();
        if self.vx.len() != n {
            self.init_velocities();
        }

        // Parallel update: compute new positions and velocities
        let results: Vec<_> = (0..n).into_par_iter().map(|i| {
            let p_start = Point3::new(self.x[i], self.y[i], self.z[i]);
            
            // 1. Get velocity at start (v1)
            let v1 = self.get_velocity_at_point(&p_start, mesh, grid, dof_mgr, velocity_vec).unwrap_or(Vector3::zeros());

            // 2. Predict midpoint (p_mid = p_start + v1 * dt/2)
            let p_mid = p_start + v1 * (dt * 0.5);

            // 3. Get velocity at midpoint (v_mid)
            let v_mid = self.get_velocity_at_point(&p_mid, mesh, grid, dof_mgr, velocity_vec).unwrap_or(v1);

            // 4. Update final position (p_end = p_start + v_mid * dt)
            let p_end = p_start + v_mid * dt;
            
            (p_end, v_mid)
        }).collect();

        // Write back to SoA
        for i in 0..n {
            self.x[i] = results[i].0.x;
            self.y[i] = results[i].0.y;
            self.z[i] = results[i].0.z;
            self.vx[i] = results[i].1.x;
            self.vy[i] = results[i].1.y;
            self.vz[i] = results[i].1.z;
        }
    }

    /// Helper: Interpolate velocity from mesh to a spatial point.
    fn get_velocity_at_point(
        &self,
        p: &Point3<f64>,
        mesh: &Mesh,
        grid: &SearchGrid,
        dof_mgr: &crate::fem::DofManager,
        velocity_vec: &[f64],
    ) -> Option<Vector3<f64>> {
        // 1. Find candidate elements using the grid
        let candidates = grid.get_potential_elements(*p);
        
        for &elem_id in candidates {
            let elem = &mesh.connectivity.tet10_elements[elem_id];
            
            // Extract corner vertices for linear check
            let mut vertices = [Point3::origin(); 4];
            for i in 0..4 {
                vertices[i] = mesh.geometry.nodes[elem.nodes[i]];
            }

            // Quick linear check
            let l_lin = crate::fem::Tet10Basis::cartesian_to_barycentric(p, &vertices);
            
            // Check if inside reference tet (with small tolerance)
            let tol = 1e-6;
            if l_lin.iter().all(|&l| l >= -tol && l <= 1.0 + tol) {
                // Potential match! Now get high-accuracy barycentric for Tet10
                let mut nodes = [Point3::origin(); 10];
                for i in 0..10 {
                    nodes[i] = mesh.geometry.nodes[elem.nodes[i]];
                }
                
                let l_exact = crate::fem::Tet10Basis::find_barycentric_iterative(p, &nodes, l_lin);
                
                // Interpolate velocity at l_exact
                let mut v_nodal = [Vector3::zeros(); 10];
                for i in 0..10 {
                    let global_id = elem.nodes[i];
                    v_nodal[i] = Vector3::new(
                        velocity_vec[dof_mgr.global_dof(global_id, 0)],
                        velocity_vec[dof_mgr.global_dof(global_id, 1)],
                        velocity_vec[dof_mgr.global_dof(global_id, 2)],
                    );
                }
                
                return Some(crate::fem::Tet10Basis::evaluate_at_point(&l_exact, &v_nodal));
            }
        }
        
        None // Out of bounds or not in any element
    }

    /// Map tracer properties to elements.
    /// Returns (material_ids, plastic_strains) per element.
    pub fn get_element_properties(&self, mesh: &Mesh, grid: &SearchGrid) -> (Vec<u32>, Vec<f64>) {
        let n_elems = mesh.num_elements();
        let n_tracers = self.num_tracers();

        // 1. Concurrent binning of tracers to elements
        // We use a simple but parallel-friendly approach:
        // Each tracer finds its element and we store (tracer_idx, elem_id).
        let tracer_to_elem: Vec<Option<usize>> = (0..n_tracers).into_par_iter().map(|i| {
            let p = Point3::new(self.x[i], self.y[i], self.z[i]);
            let candidates = grid.get_potential_elements(p);
            
            for &elem_id in candidates {
                let elem = &mesh.connectivity.tet10_elements[elem_id];
                let mut vertices = [Point3::origin(); 4];
                for k in 0..4 {
                    vertices[k] = mesh.geometry.nodes[elem.nodes[k]];
                }
                let l = crate::fem::Tet10Basis::cartesian_to_barycentric(&p, &vertices);
                let tol = 1e-5;
                if l.iter().all(|&val| val >= -tol && val <= 1.0 + tol) {
                    return Some(elem_id);
                }
            }
            None
        }).collect();

        // 2. Accumulate properties per element
        // For performance, we'll do this serially for now or use atomics.
        // Let's use a simple per-element accumulation.
        let mut elem_mat_counts = vec![std::collections::HashMap::new(); n_elems];
        let mut elem_strain_sum = vec![0.0; n_elems];
        let mut elem_tracer_count = vec![0usize; n_elems];

        for i in 0..n_tracers {
            if let Some(elem_id) = tracer_to_elem[i] {
                let mat_id = self.material_id[i];
                let strain = self.plastic_strain[i];
                
                *elem_mat_counts[elem_id].entry(mat_id).or_insert(0) += 1;
                elem_strain_sum[elem_id] += strain;
                elem_tracer_count[elem_id] += 1;
            }
        }

        // 3. Finalize averages and majority vote
        let mut final_mat_ids = vec![0; n_elems];
        let mut final_strains = vec![0.0; n_elems];

        for i in 0..n_elems {
            // Majority material ID
            if let Some((&mat_id, _)) = elem_mat_counts[i].iter().max_by_key(|&(_, count)| count) {
                final_mat_ids[i] = mat_id;
            }
            
            // Average strain
            if elem_tracer_count[i] > 0 {
                final_strains[i] = elem_strain_sum[i] / elem_tracer_count[i] as f64;
            }
        }

        (final_mat_ids, final_strains)
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
