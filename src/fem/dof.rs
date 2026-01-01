use std::collections::HashSet;

/// Degree of Freedom (DOF) manager
///
/// Handles DOF numbering and boundary condition tracking for FEM assembly.
/// Supports both simple fixed-DOF systems and Mixed (P2-P1) formulations.
#[derive(Debug, Clone)]
pub struct DofManager {
    /// Number of nodes in the mesh
    num_nodes: usize,

    /// Number of velocity DOFs per node (usually 3)
    vel_dofs_per_node: usize,

    /// Total velocity DOFs (placed first in segregated ordering)
    total_vel_dofs: usize,

    /// Total number of pressure DOFs (placed after velocity)
    num_pressure_dofs: usize,

    /// Mapping from node_id to pressure DOF index (if any)
    pressure_node_map: Vec<Option<usize>>,

    /// Total number of DOFs (Vel + Pres)
    total_dofs: usize,

    /// Set of DOFs with Dirichlet boundary conditions
    dirichlet_dofs: HashSet<usize>,

    /// Values for Dirichlet DOFs
    dirichlet_values: Vec<f64>,
}

impl DofManager {
    /// Create a new DOF manager for a fixed number of DOFs per node
    /// (e.g., pure thermal, pure elasticity, or penalty Stokes)
    pub fn new(num_nodes: usize, dofs_per_node: usize) -> Self {
        let total_dofs = num_nodes * dofs_per_node;

        Self {
            num_nodes,
            vel_dofs_per_node: dofs_per_node,
            total_vel_dofs: total_dofs,
            num_pressure_dofs: 0,
            pressure_node_map: vec![None; num_nodes],
            total_dofs,
            dirichlet_dofs: HashSet::new(),
            dirichlet_values: vec![0.0; total_dofs],
        }
    }

    /// Create a new DOF manager for a Mixed P2-P1 formulation
    ///
    /// # Arguments
    /// * `num_nodes` - Total number of nodes (including mid-edges)
    /// * `corner_nodes` - Indices of nodes that also have a pressure DOF (usually element vertices)
    pub fn new_mixed(num_nodes: usize, corner_nodes: &[usize]) -> Self {
        let vel_dofs_per_node = 3;
        let total_vel_dofs = num_nodes * vel_dofs_per_node;
        
        // Identify unique pressure nodes and create mapping
        let unique_corners: HashSet<usize> = corner_nodes.iter().cloned().collect();
        let mut sorted_corners: Vec<usize> = unique_corners.into_iter().collect();
        sorted_corners.sort_unstable();
        
        let mut pressure_node_map = vec![None; num_nodes];
        for (idx, &node_id) in sorted_corners.iter().enumerate() {
            pressure_node_map[node_id] = Some(idx);
        }
        
        let num_pressure_dofs = sorted_corners.len();
        let total_dofs = total_vel_dofs + num_pressure_dofs;

        Self {
            num_nodes,
            vel_dofs_per_node,
            total_vel_dofs,
            num_pressure_dofs,
            pressure_node_map,
            total_dofs,
            dirichlet_dofs: HashSet::new(),
            dirichlet_values: vec![0.0; total_dofs],
        }
    }

    /// Get the global velocity DOF index for a node and component
    pub fn velocity_dof(&self, node_id: usize, component: usize) -> usize {
        debug_assert!(node_id < self.num_nodes);
        debug_assert!(component < self.vel_dofs_per_node);
        node_id * self.vel_dofs_per_node + component
    }

    /// Get the global pressure DOF index for a node
    pub fn pressure_dof(&self, node_id: usize) -> Option<usize> {
        self.pressure_node_map[node_id].map(|idx| self.total_vel_dofs + idx)
    }

    /// Backwards compatible global_dof lookup
    /// (Treats local_dof < vel_dofs_per_node as velocity, otherwise pressure)
    pub fn global_dof(&self, node_id: usize, local_dof: usize) -> usize {
        if local_dof < self.vel_dofs_per_node {
            self.velocity_dof(node_id, local_dof)
        } else {
            self.pressure_dof(node_id).expect("Node has no pressure DOF")
        }
    }

    /// Apply Dirichlet boundary condition to a DOF
    ///
    /// # Arguments
    /// * `dof` - Global DOF index
    /// * `value` - Prescribed value
    pub fn set_dirichlet(&mut self, dof: usize, value: f64) {
        debug_assert!(dof < self.total_dofs, "DOF index out of bounds");

        self.dirichlet_dofs.insert(dof);
        self.dirichlet_values[dof] = value;
    }

    /// Apply Dirichlet BC to a node (all DOFs at that node)
    ///
    /// # Arguments
    /// * `node_id` - Node index
    /// * `value` - Prescribed value for all DOFs at this node
    pub fn set_dirichlet_node(&mut self, node_id: usize, value: f64) {
        for local_dof in 0..self.vel_dofs_per_node {
            let dof = self.global_dof(node_id, local_dof);
            self.set_dirichlet(dof, value);
        }
    }

    /// Check if a DOF has Dirichlet BC
    pub fn is_dirichlet(&self, dof: usize) -> bool {
        self.dirichlet_dofs.contains(&dof)
    }

    /// Get the Dirichlet value for a DOF
    pub fn get_dirichlet_value(&self, dof: usize) -> f64 {
        self.dirichlet_values[dof]
    }

    /// Get total number of DOFs
    pub fn total_dofs(&self) -> usize {
        self.total_dofs
    }

    /// Get number of free DOFs (not constrained by Dirichlet BC)
    pub fn num_free_dofs(&self) -> usize {
        self.total_dofs - self.dirichlet_dofs.len()
    }

    /// Get number of constrained DOFs
    pub fn num_constrained_dofs(&self) -> usize {
        self.dirichlet_dofs.len()
    }

    /// Get DOFs per node
    pub fn dofs_per_node(&self) -> usize {
        self.vel_dofs_per_node
    }

    /// Get number of nodes
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get total velocity DOFs
    pub fn total_vel_dofs(&self) -> usize {
        self.total_vel_dofs
    }

    /// Get total pressure DOFs
    pub fn total_pressure_dofs(&self) -> usize {
        self.num_pressure_dofs
    }

    /// Get pressure node map
    pub fn pressure_node_map(&self) -> &Vec<Option<usize>> {
        &self.pressure_node_map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_dof_numbering() {
        let dof_mgr = DofManager::new(10, 1);

        assert_eq!(dof_mgr.total_dofs(), 10);
        assert_eq!(dof_mgr.global_dof(0, 0), 0);
        assert_eq!(dof_mgr.global_dof(5, 0), 5);
        assert_eq!(dof_mgr.global_dof(9, 0), 9);
    }

    #[test]
    fn test_vector_dof_numbering() {
        let dof_mgr = DofManager::new(10, 3);

        assert_eq!(dof_mgr.total_dofs(), 30);

        // Node 0: DOFs 0, 1, 2
        assert_eq!(dof_mgr.global_dof(0, 0), 0);
        assert_eq!(dof_mgr.global_dof(0, 1), 1);
        assert_eq!(dof_mgr.global_dof(0, 2), 2);

        // Node 1: DOFs 3, 4, 5
        assert_eq!(dof_mgr.global_dof(1, 0), 3);
        assert_eq!(dof_mgr.global_dof(1, 1), 4);
        assert_eq!(dof_mgr.global_dof(1, 2), 5);
    }

    #[test]
    fn test_dirichlet_bc() {
        let mut dof_mgr = DofManager::new(10, 1);

        assert_eq!(dof_mgr.num_free_dofs(), 10);
        assert_eq!(dof_mgr.num_constrained_dofs(), 0);

        // Apply BC to DOF 0
        dof_mgr.set_dirichlet(0, 100.0);

        assert!(dof_mgr.is_dirichlet(0));
        assert!(!dof_mgr.is_dirichlet(1));
        assert_eq!(dof_mgr.get_dirichlet_value(0), 100.0);
        assert_eq!(dof_mgr.num_free_dofs(), 9);
        assert_eq!(dof_mgr.num_constrained_dofs(), 1);
    }

    #[test]
    fn test_dirichlet_node() {
        let mut dof_mgr = DofManager::new(10, 3);

        // Apply BC to entire node
        dof_mgr.set_dirichlet_node(0, 50.0);

        assert!(dof_mgr.is_dirichlet(0));
        assert!(dof_mgr.is_dirichlet(1));
        assert!(dof_mgr.is_dirichlet(2));
        assert!(!dof_mgr.is_dirichlet(3));

        assert_eq!(dof_mgr.num_constrained_dofs(), 3);
        assert_eq!(dof_mgr.num_free_dofs(), 27);
    }
}
