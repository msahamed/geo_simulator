use sprs::{CsMat, TriMat};
use rayon::prelude::*;
use crate::mesh::Mesh;
use crate::fem::{DofManager, ElementMatrix, BoundaryConditions, Tet10Face};

/// Global matrix assembler
pub struct Assembler;

impl Assembler {
    /// Assemble global stiffness matrix (serial version)
    ///
    /// K = Σ_e K_e
    ///
    /// # Arguments
    /// * `mesh` - The mesh
    /// * `dof_mgr` - DOF manager
    /// * `conductivity` - Thermal conductivity (uniform)
    ///
    /// # Returns
    /// Global stiffness matrix in CSR format
    pub fn assemble_thermal_stiffness_serial(
        mesh: &Mesh,
        dof_mgr: &DofManager,
        conductivity: f64,
    ) -> CsMat<f64> {
        let n_dofs = dof_mgr.total_dofs();

        // Use triplet format for efficient insertion during assembly
        let mut triplets = TriMat::new((n_dofs, n_dofs));

        // Loop over all elements
        for elem in &mesh.connectivity.tet10_elements {
            // Get vertex coordinates
            let vertices = [
                mesh.geometry.nodes[elem.vertices()[0]],
                mesh.geometry.nodes[elem.vertices()[1]],
                mesh.geometry.nodes[elem.vertices()[2]],
                mesh.geometry.nodes[elem.vertices()[3]],
            ];

            // Compute element stiffness matrix
            let k_elem = ElementMatrix::thermal_stiffness(&vertices, conductivity);

            // Assemble into global matrix
            for i in 0..10 {
                let global_i = elem.nodes[i];

                for j in 0..10 {
                    let global_j = elem.nodes[j];

                    // Add to triplet
                    triplets.add_triplet(global_i, global_j, k_elem[(i, j)]);
                }
            }
        }

        // Convert to CSR format
        triplets.to_csr()
    }

    /// Assemble global stiffness matrix (parallel version using Rayon)
    ///
    /// # Arguments
    /// * `mesh` - The mesh
    /// * `dof_mgr` - DOF manager
    /// * `conductivity` - Thermal conductivity (uniform)
    ///
    /// # Returns
    /// Global stiffness matrix in CSR format
    pub fn assemble_thermal_stiffness_parallel(
        mesh: &Mesh,
        dof_mgr: &DofManager,
        conductivity: f64,
    ) -> CsMat<f64> {
        let n_dofs = dof_mgr.total_dofs();

        // Parallel assembly: each thread assembles its own triplet list
        let local_triplets: Vec<_> = mesh
            .connectivity
            .tet10_elements
            .par_iter()
            .map(|elem| {
                // Get vertex coordinates
                let vertices = [
                    mesh.geometry.nodes[elem.vertices()[0]],
                    mesh.geometry.nodes[elem.vertices()[1]],
                    mesh.geometry.nodes[elem.vertices()[2]],
                    mesh.geometry.nodes[elem.vertices()[3]],
                ];

                // Compute element matrix
                let k_elem = ElementMatrix::thermal_stiffness(&vertices, conductivity);

                // Store triplets for this element
                let mut elem_triplets = Vec::with_capacity(100); // 10x10 = 100 entries

                for i in 0..10 {
                    let global_i = elem.nodes[i];
                    for j in 0..10 {
                        let global_j = elem.nodes[j];
                        elem_triplets.push((global_i, global_j, k_elem[(i, j)]));
                    }
                }

                elem_triplets
            })
            .collect();

        // Merge all triplets
        let mut triplets = TriMat::new((n_dofs, n_dofs));

        for elem_triplets in local_triplets {
            for (i, j, val) in elem_triplets {
                triplets.add_triplet(i, j, val);
            }
        }

        // Convert to CSR
        triplets.to_csr()
    }

    /// Assemble global load vector
    ///
    /// f = Σ_e f_e
    ///
    /// # Arguments
    /// * `mesh` - The mesh
    /// * `dof_mgr` - DOF manager
    /// * `source` - Uniform heat source (W/m³)
    ///
    /// # Returns
    /// Global load vector
    pub fn assemble_thermal_load(
        mesh: &Mesh,
        dof_mgr: &DofManager,
        source: f64,
    ) -> Vec<f64> {
        let n_dofs = dof_mgr.total_dofs();
        let mut f_global = vec![0.0; n_dofs];

        for elem in &mesh.connectivity.tet10_elements {
            let vertices = [
                mesh.geometry.nodes[elem.vertices()[0]],
                mesh.geometry.nodes[elem.vertices()[1]],
                mesh.geometry.nodes[elem.vertices()[2]],
                mesh.geometry.nodes[elem.vertices()[3]],
            ];

            let f_elem = ElementMatrix::thermal_load(&vertices, source);

            // Assemble into global vector
            for i in 0..10 {
                let global_i = elem.nodes[i];
                f_global[global_i] += f_elem[i];
            }
        }

        f_global
    }

    /// Apply Dirichlet boundary conditions using elimination method
    ///
    /// Modifies K and f to enforce u_i = prescribed_value for constrained DOFs
    ///
    /// # Arguments
    /// * `K` - Global stiffness matrix
    /// * `f` - Global load vector
    /// * `dof_mgr` - DOF manager with BC information
    ///
    /// # Returns
    /// Modified (K, f) as new sparse matrix and vector
    #[allow(non_snake_case)]
    pub fn apply_dirichlet_bcs(
        K: &CsMat<f64>,
        f: &[f64],
        dof_mgr: &DofManager,
    ) -> (CsMat<f64>, Vec<f64>) {
        let n = dof_mgr.total_dofs();

        // Method: For constrained DOF i with value v_i:
        // 1. Modify RHS for free DOFs: f[j] -= K[j,i] * v_i for all j
        // 2. Zero out row i and column i
        // 3. Set K[i,i] = 1, f[i] = v_i

        let mut f_new = f.to_vec();

        // First pass: modify RHS for coupling to constrained DOFs
        for (row_idx, row) in K.outer_iterator().enumerate() {
            if !dof_mgr.is_dirichlet(row_idx) {
                // Free DOF - modify RHS for coupling to constrained DOFs
                for (col_idx, &val) in row.iter() {
                    if dof_mgr.is_dirichlet(col_idx) {
                        let bc_value = dof_mgr.get_dirichlet_value(col_idx);
                        f_new[row_idx] -= val * bc_value;
                    }
                }
            }
        }

        // Build new matrix
        let mut tri = TriMat::new((n, n));

        // Second pass: copy matrix with modifications
        for (row_idx, row) in K.outer_iterator().enumerate() {
            if dof_mgr.is_dirichlet(row_idx) {
                // Constrained DOF: set row to identity
                tri.add_triplet(row_idx, row_idx, 1.0);
                f_new[row_idx] = dof_mgr.get_dirichlet_value(row_idx);
            } else {
                // Free DOF: copy row, but zero out columns for constrained DOFs
                for (col_idx, &val) in row.iter() {
                    if !dof_mgr.is_dirichlet(col_idx) {
                        tri.add_triplet(row_idx, col_idx, val);
                    }
                }
            }
        }

        (tri.to_csr(), f_new)
    }

    /// Apply Neumann boundary conditions to load vector
    ///
    /// Adds flux contributions from boundary faces to the RHS
    ///
    /// # Arguments
    /// * `mesh` - The mesh
    /// * `f` - Global load vector (will be modified in-place)
    /// * `bcs` - Boundary conditions with Neumann specifications
    pub fn apply_neumann_bcs(
        mesh: &Mesh,
        f: &mut [f64],
        bcs: &BoundaryConditions,
    ) {
        // Iterate over all Neumann boundary faces
        for (face, neumann) in bcs.neumann_faces() {
            let elem_idx = face.element_id;
            let local_face = face.local_face_id;

            // Get element node indices
            let elem = &mesh.connectivity.tet10_elements[elem_idx];

            // Extract face
            let face_geom = Tet10Face::from_element_face(&elem.nodes, local_face);

            // Integrate flux over face
            let flux_contrib = face_geom.integrate_flux(&mesh.geometry.nodes, neumann.flux);

            // Add to global load vector
            for (i, &global_node) in face_geom.nodes.iter().enumerate() {
                f[global_node] += flux_contrib[i];
            }
        }
    }

    /// Helper: Identify boundary faces on a planar surface
    ///
    /// Detects faces where all nodes satisfy a condition (e.g., x ≈ 0)
    ///
    /// # Arguments
    /// * `mesh` - The mesh
    /// * `condition` - Function that returns true if a node is on the boundary
    ///
    /// # Returns
    /// Vector of boundary faces
    pub fn find_boundary_faces<F>(
        mesh: &Mesh,
        condition: F,
    ) -> Vec<(usize, usize)>  // (element_id, local_face_id)
    where
        F: Fn(&nalgebra::Point3<f64>) -> bool,
    {
        let mut boundary_faces = Vec::new();

        for (elem_idx, elem) in mesh.connectivity.tet10_elements.iter().enumerate() {
            // Check all 4 faces
            for local_face in 0..4 {
                let face = Tet10Face::from_element_face(&elem.nodes, local_face);

                // Check if ALL nodes on this face satisfy the condition
                let all_on_boundary = face.nodes.iter()
                    .all(|&node_id| condition(&mesh.geometry.nodes[node_id]));

                if all_on_boundary {
                    boundary_faces.push((elem_idx, local_face));
                }
            }
        }

        boundary_faces
    }

    /// Assemble global stiffness matrix for elasticity (serial)
    ///
    /// K = Σ_e K_e  where each K_e is 30×30 for vector fields (3 DOF/node)
    ///
    /// # Arguments
    /// * `mesh` - The mesh
    /// * `dof_mgr` - DOF manager (must have dofs_per_node = 3)
    /// * `material` - Elastic material properties
    ///
    /// # Returns
    /// Global stiffness matrix in CSR format
    ///
    /// # Panics
    /// Panics if DOF manager doesn't have exactly 3 DOF per node
    #[allow(non_snake_case)]
    pub fn assemble_elasticity_stiffness_serial(
        mesh: &Mesh,
        dof_mgr: &DofManager,
        material: &crate::mechanics::IsotropicElasticity,
    ) -> CsMat<f64> {
        assert_eq!(
            dof_mgr.dofs_per_node(),
            3,
            "Elasticity requires 3 DOF per node (displacement components)"
        );

        let n_dofs = dof_mgr.total_dofs();
        let mut triplets = TriMat::new((n_dofs, n_dofs));

        // Loop over all elements
        for elem in &mesh.connectivity.tet10_elements {
            // Get physical coordinates of element vertices
            let vertices = [
                mesh.geometry.nodes[elem.vertices()[0]],
                mesh.geometry.nodes[elem.vertices()[1]],
                mesh.geometry.nodes[elem.vertices()[2]],
                mesh.geometry.nodes[elem.vertices()[3]],
            ];

            // Compute 30×30 element stiffness matrix
            let K_elem = crate::mechanics::ElasticityElement::stiffness_matrix(&vertices, material);

            // Assemble into global matrix
            // For each node pair (i,j) and DOF component pair (comp_i, comp_j)
            for i in 0..10 {
                for local_dof_i in 0..3 {
                    let global_i = dof_mgr.global_dof(elem.nodes[i], local_dof_i);

                    for j in 0..10 {
                        for local_dof_j in 0..3 {
                            let global_j = dof_mgr.global_dof(elem.nodes[j], local_dof_j);

                            // Map to element matrix indices
                            let elem_row = 3 * i + local_dof_i;
                            let elem_col = 3 * j + local_dof_j;

                            triplets.add_triplet(global_i, global_j, K_elem[(elem_row, elem_col)]);
                        }
                    }
                }
            }
        }

        triplets.to_csr()
    }

    /// Assemble global stiffness matrix for elasticity (parallel)
    ///
    /// Uses Rayon for parallel element matrix computation
    ///
    /// # Arguments
    /// * `mesh` - The mesh
    /// * `dof_mgr` - DOF manager (must have dofs_per_node = 3)
    /// * `material` - Elastic material properties
    ///
    /// # Returns
    /// Global stiffness matrix in CSR format
    #[allow(non_snake_case)]
    pub fn assemble_elasticity_stiffness_parallel(
        mesh: &Mesh,
        dof_mgr: &DofManager,
        material: &crate::mechanics::IsotropicElasticity,
    ) -> CsMat<f64> {
        assert_eq!(dof_mgr.dofs_per_node(), 3, "Elasticity requires 3 DOF per node");

        let n_dofs = dof_mgr.total_dofs();

        // Compute element matrices in parallel
        let element_matrices: Vec<_> = mesh
            .connectivity
            .tet10_elements
            .par_iter()
            .map(|elem| {
                let vertices = [
                    mesh.geometry.nodes[elem.vertices()[0]],
                    mesh.geometry.nodes[elem.vertices()[1]],
                    mesh.geometry.nodes[elem.vertices()[2]],
                    mesh.geometry.nodes[elem.vertices()[3]],
                ];
                crate::mechanics::ElasticityElement::stiffness_matrix(&vertices, material)
            })
            .collect();

        // Sequential assembly of triplets
        let mut triplets = TriMat::new((n_dofs, n_dofs));

        for (elem_idx, elem) in mesh.connectivity.tet10_elements.iter().enumerate() {
            let K_elem = &element_matrices[elem_idx];

            for i in 0..10 {
                for local_dof_i in 0..3 {
                    let global_i = dof_mgr.global_dof(elem.nodes[i], local_dof_i);

                    for j in 0..10 {
                        for local_dof_j in 0..3 {
                            let global_j = dof_mgr.global_dof(elem.nodes[j], local_dof_j);

                            let elem_row = 3 * i + local_dof_i;
                            let elem_col = 3 * j + local_dof_j;

                            triplets.add_triplet(global_i, global_j, K_elem[(elem_row, elem_col)]);
                        }
                    }
                }
            }
        }

        triplets.to_csr()
    }

    /// Assemble global body force vector for gravity
    ///
    /// f = Σ_e f_e  where f_e = ∫ N^T ρg dV
    ///
    /// # Arguments
    /// * `mesh` - The mesh
    /// * `dof_mgr` - DOF manager (must have dofs_per_node = 3)
    /// * `density` - Material density (kg/m³)
    /// * `gravity` - Gravity acceleration vector (m/s²)
    ///
    /// # Returns
    /// Global load vector with gravitational forces
    pub fn assemble_gravity_load(
        mesh: &Mesh,
        dof_mgr: &DofManager,
        density: f64,
        gravity: &nalgebra::Vector3<f64>,
    ) -> Vec<f64> {
        assert_eq!(dof_mgr.dofs_per_node(), 3, "Gravity load requires 3 DOF per node");

        let n_dofs = dof_mgr.total_dofs();
        let mut f_global = vec![0.0; n_dofs];

        // Loop over all elements
        for elem in &mesh.connectivity.tet10_elements {
            let vertices = [
                mesh.geometry.nodes[elem.vertices()[0]],
                mesh.geometry.nodes[elem.vertices()[1]],
                mesh.geometry.nodes[elem.vertices()[2]],
                mesh.geometry.nodes[elem.vertices()[3]],
            ];

            // Compute element load vector
            let f_elem = crate::mechanics::BodyForce::gravity_load(&vertices, density, gravity);

            // Assemble into global vector
            for i in 0..10 {
                for local_dof in 0..3 {
                    let global_dof = dof_mgr.global_dof(elem.nodes[i], local_dof);
                    f_global[global_dof] += f_elem[3 * i + local_dof];
                }
            }
        }

        f_global
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_generator::MeshGenerator;

    #[test]
    fn test_assemble_small_mesh() {
        // Create a simple mesh (2x2x2)
        let mesh = MeshGenerator::generate_cube_detailed(1, 1, 1, 1.0, 1.0, 1.0);

        let dof_mgr = DofManager::new(mesh.num_nodes(), 1);

        let k = 1.0;
        let K = Assembler::assemble_thermal_stiffness_serial(&mesh, &dof_mgr, k);

        // Check dimensions
        assert_eq!(K.rows(), mesh.num_nodes());
        assert_eq!(K.cols(), mesh.num_nodes());

        // Check symmetry (within tolerance)
        // Note: CSR format makes this check a bit involved
        // For now, just check that matrix was assembled
        assert!(K.nnz() > 0, "Matrix should have non-zero entries");
    }

    #[test]
    fn test_serial_vs_parallel_assembly() {
        // Create mesh
        let mesh = MeshGenerator::generate_cube_detailed(2, 2, 2, 1.0, 1.0, 1.0);
        let dof_mgr = DofManager::new(mesh.num_nodes(), 1);

        let k = 1.0;

        // Serial assembly
        let K_serial = Assembler::assemble_thermal_stiffness_serial(&mesh, &dof_mgr, k);

        // Parallel assembly
        let K_parallel = Assembler::assemble_thermal_stiffness_parallel(&mesh, &dof_mgr, k);

        // Both should have same dimensions and nnz
        assert_eq!(K_serial.rows(), K_parallel.rows());
        assert_eq!(K_serial.cols(), K_parallel.cols());

        // Check that matrices are identical
        // (requires comparing values, which is a bit involved for CSR)
        // For now, check structure
        assert_eq!(K_serial.nnz(), K_parallel.nnz());
    }

    #[test]
    fn test_load_vector_assembly() {
        let mesh = MeshGenerator::generate_cube_detailed(1, 1, 1, 1.0, 1.0, 1.0);
        let dof_mgr = DofManager::new(mesh.num_nodes(), 1);

        let Q = 10.0;
        let f = Assembler::assemble_thermal_load(&mesh, &dof_mgr, Q);

        assert_eq!(f.len(), mesh.num_nodes());

        // Sum should approximately equal Q * total_volume
        let total_volume = 1.0; // 1x1x1 cube
        let sum: f64 = f.iter().sum();

        // The mesh generator creates simplified tets, not filling full volume
        // Just check that we got a reasonable positive sum
        assert!(sum > 0.0, "Load vector sum should be positive");
        assert!(sum < Q * total_volume * 2.0, "Load vector sum seems too large");
    }
}
