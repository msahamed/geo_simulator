/// A 10-node quadratic tetrahedral element (Tet10)
///
/// Node numbering:
/// Vertices: 0, 1, 2, 3
/// Edge midpoints:
///   4: midpoint of edge 0-1
///   5: midpoint of edge 1-2
///   6: midpoint of edge 2-0
///   7: midpoint of edge 0-3
///   8: midpoint of edge 1-3
///   9: midpoint of edge 2-3
#[derive(Debug, Clone)]
pub struct Tet10Element {
    /// Global node indices for this element (10 nodes)
    pub nodes: [usize; 10],
}

impl Tet10Element {
    pub fn new(nodes: [usize; 10]) -> Self {
        Self { nodes }
    }

    /// Get the vertex node indices (first 4 nodes)
    pub fn vertices(&self) -> [usize; 4] {
        [self.nodes[0], self.nodes[1], self.nodes[2], self.nodes[3]]
    }

    /// Get the edge midpoint node indices (last 6 nodes)
    pub fn edge_nodes(&self) -> [usize; 6] {
        [
            self.nodes[4],
            self.nodes[5],
            self.nodes[6],
            self.nodes[7],
            self.nodes[8],
            self.nodes[9],
        ]
    }

    /// Get edges as pairs of vertex indices
    pub fn edges() -> [(usize, usize); 6] {
        [
            (0, 1), // edge 0
            (1, 2), // edge 1
            (2, 0), // edge 2
            (0, 3), // edge 3
            (1, 3), // edge 4
            (2, 3), // edge 5
        ]
    }
}

/// Connectivity information for the mesh
#[derive(Debug, Clone)]
pub struct Connectivity {
    pub tet10_elements: Vec<Tet10Element>,
}

impl Connectivity {
    pub fn new() -> Self {
        Self {
            tet10_elements: Vec::new(),
        }
    }

    pub fn add_element(&mut self, element: Tet10Element) {
        self.tet10_elements.push(element);
    }

    pub fn num_elements(&self) -> usize {
        self.tet10_elements.len()
    }

    /// Get all unique vertex (corner) nodes from all elements
    pub fn corner_nodes(&self) -> Vec<usize> {
        use std::collections::HashSet;
        let mut corners = HashSet::new();
        for elem in &self.tet10_elements {
            for &node in &elem.nodes[0..4] {
                corners.insert(node);
            }
        }
        let mut sorted: Vec<usize> = corners.into_iter().collect();
        sorted.sort_unstable();
        sorted
    }
}

impl Default for Connectivity {
    fn default() -> Self {
        Self::new()
    }
}
