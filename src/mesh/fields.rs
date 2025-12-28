use std::collections::HashMap;
use nalgebra::Vector3;

/// Scalar field data on mesh nodes
#[derive(Debug, Clone)]
pub struct ScalarField {
    pub name: String,
    pub data: Vec<f64>,
}

impl ScalarField {
    pub fn new(name: &str, data: Vec<f64>) -> Self {
        Self {
            name: name.to_string(),
            data,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Vector field data on mesh nodes (3 components per node)
#[derive(Debug, Clone)]
pub struct VectorField {
    pub name: String,
    pub data: Vec<Vector3<f64>>,
}

impl VectorField {
    pub fn new(name: &str, data: Vec<Vector3<f64>>) -> Self {
        Self {
            name: name.to_string(),
            data,
        }
    }

    /// Create vector field from flat DOF vector
    ///
    /// Converts interleaved DOF array [ux0, uy0, uz0, ux1, uy1, uz1, ...]
    /// into vector of Vector3 objects
    ///
    /// # Arguments
    /// * `name` - Field name
    /// * `dof_vector` - Flat array with 3*n entries (interleaved components)
    ///
    /// # Panics
    /// Panics if dof_vector.len() is not divisible by 3
    pub fn from_dof_vector(name: &str, dof_vector: &[f64]) -> Self {
        assert_eq!(
            dof_vector.len() % 3,
            0,
            "DOF vector must have 3*n entries for vector field"
        );

        let data: Vec<_> = dof_vector
            .chunks(3)
            .map(|chunk| Vector3::new(chunk[0], chunk[1], chunk[2]))
            .collect();

        Self::new(name, data)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Collection of scalar and vector fields on the mesh
#[derive(Debug, Clone, Default)]
pub struct FieldData {
    pub scalar_fields: HashMap<String, ScalarField>,
    pub vector_fields: HashMap<String, VectorField>,
}

impl FieldData {
    pub fn new() -> Self {
        Self {
            scalar_fields: HashMap::new(),
            vector_fields: HashMap::new(),
        }
    }

    /// Add a scalar field
    pub fn add_field(&mut self, field: ScalarField) {
        self.scalar_fields.insert(field.name.clone(), field);
    }

    /// Add a vector field
    pub fn add_vector_field(&mut self, field: VectorField) {
        self.vector_fields.insert(field.name.clone(), field);
    }

    /// Get a scalar field by name
    pub fn get_field(&self, name: &str) -> Option<&ScalarField> {
        self.scalar_fields.get(name)
    }

    /// Get a vector field by name
    pub fn get_vector_field(&self, name: &str) -> Option<&VectorField> {
        self.vector_fields.get(name)
    }

    /// Get all scalar field names
    pub fn field_names(&self) -> Vec<&String> {
        self.scalar_fields.keys().collect()
    }

    /// Get all vector field names
    pub fn vector_field_names(&self) -> Vec<&String> {
        self.vector_fields.keys().collect()
    }

    /// Number of scalar fields
    pub fn num_fields(&self) -> usize {
        self.scalar_fields.len()
    }

    /// Number of vector fields
    pub fn num_vector_fields(&self) -> usize {
        self.vector_fields.len()
    }

    /// Check if empty (no fields at all)
    pub fn is_empty(&self) -> bool {
        self.scalar_fields.is_empty() && self.vector_fields.is_empty()
    }
}
