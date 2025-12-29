# Geodynamic Simulation Theory & Implementation

This document provides a deep dive into the physics and mathematics governing our Rust-based geodynamic simulator.

## 1. Governing Equations: The "Engine" of the Earth

### 1.1 Momentum Conservation (The Stokes Flow)
In the Earth's mantle, viscosity is so high ($10^{21}$ Pa·s) that inertia is irrelevant. Balancing forces is like pushing honey through a tube.

**Equation:**
$$\nabla \cdot \sigma + \rho \mathbf{g} = 0$$

**Intuitive Understandings:**
*   **The "Honey" Analogy:** If you drop a marble in water, it accelerates (Inertia). If you drop it in cold honey, it instantly reaches a terminal velocity where gravity is perfectly balanced by viscous drag. That "balance" is exactly what this equation represents.
*   **Stress Tensor $\sigma$:** This is a $3 \times 3$ matrix representing forces in every direction. We split it into pressure ($P$) and deviatoric stress ($\tau$):
    $$\sigma_{ij} = -P \delta_{ij} + \tau_{ij}$$

### 1.2 Mass Conservation (Incompressibility)
$$\nabla \cdot \mathbf{v} = 0$$

**Intuitive Understanding:**
Imagine squeezing a balloon. If you push one side in, another side *must* push out. In the mantle, there are no "voids" opening up; every cubic meter of rock that moves left must be replaced by another cubic meter moving in.

---

## 2. Constitutive Laws: How Rocks "Feel" Stress

Rocks are **Visco-Elasto-Plastic (VEP)**. We can understand this through a simple 1D mechanical analog.

### 2.1 The 1D "Spring-Dashpot-Slider" Model
Imagine three components in a single line:
1.  **Spring (Elasticity):** Stretches instantly under load. Stores energy.
2.  **Dashpot (Viscosity):** A piston in oil. Moves slowly over time.
3.  **Slider (Plasticity):** A heavy block that won't budge until you pull hard enough to overcome friction (the "Yield Stress").

**The Math (Strain Rate Decomposition):**
The total strain rate $\dot{\epsilon}_{total}$ is the sum of all three:
$$\dot{\epsilon}_{total} = \dot{\epsilon}_{v} + \dot{\epsilon}_{e} + \dot{\epsilon}_{p}$$

*   **Viscous:** $\dot{\epsilon}_{v} = \frac{\tau}{2\eta}$
*   **Elastic:** $\dot{\epsilon}_{e} = \frac{1}{2G} \frac{D\tau}{Dt}$ (where $G$ is shear modulus)
*   **Plastic:** $\dot{\epsilon}_{p}$ is non-zero only when $|\tau| = \tau_{yield}$.

**Intuitive Example:** Rubbing your hands together.
- **Elastic:** The initial skin stretch.
- **Viscous:** Soft skin resisting the movement slightly.
- **Plastic:** When your hands actually start sliding (frictional failure).
- **Shear Heating:** Notice your hands getting warm? That is the energy dissipated by the viscous and plastic terms being converted to heat ($ \Phi = \tau : \dot{\epsilon}_{vp} $).

---

## 3. Finite Element Method: The Tet10 Formulation

We divide the world into tiny tetrahedra (the "Finite Elements").

### 3.1 The Tet10 "Bending" Advantage
*   **Tet4 (Linear):** 4 nodes at corners. The edges stay perfectly straight.
*   **Tet10 (Quadratic):** 10 nodes (4 corners + 6 mid-edges). The edges can **curve**.

**Intuitive Understanding (Locking):**
Imagine trying to model a subducting slab bending into the mantle.
- A **Tet4** is like a Lego brick. To make a smooth curve, you need thousands of tiny bricks. If you don't have enough, the mesh "locks" because the straight edges can't physically represent the curve.
- A **Tet10** is like a flexible piece of wood. It can curve with much less effort. You get high accuracy with a much coarser (and thus potentially faster) mesh.

### 3.2 Mapping & The Jacobian
We define elements in a "Reference" space (a perfect tetrahedron) and stretch them into "Physical" space.
The **Jacobian Matrix (J)** tracks this stretching:
$$ d\mathbf{x} = \mathbf{J} d\mathbf{\xi} $$
If your element gets too distorted, the determinant of **J** becomes zero or negative, and the "physics" breaks. Our simulator tracks this to trigger **Remeshing**.

---

## 4. Multi-Physics: The "Flavor" of the Earth

### 4.1 Surface Processes (Erosion/Deposition)
The top boundary isn't fixed.
*   **Erosion:** Removes "mass" and heat.
*   **Advection:** Moving the mountains changes the stress at the base of the crust.

### 4.2 Pore Pressure
Fluids (water/melt) reduce the effective stress:
$$\sigma_{effective} = \sigma_{total} - P_{pore}$$
**Intuitive Understanding:** This is why landslides happen during heavy rain. The water "lifts" the grains apart, reducing friction (the **Slider** moves more easily), leading to catastrophic failure.

---

## 5. Summary Table: Physics to Code

| Physics | Implementation | Rust Crate |
| :--- | :--- | :--- |
| **Stokes Flow** | Implicit Sparse Solve | `faer` / `sprs` |
| **Heat Transport** | Advection + Diffusion | `ndarray` / `rayon` |
| **Material State** | Integration Points | `nalgebra` |
| **Topology** | Unstructured Mesh | `petgraph` / Custom |
| **Material Tracking**| Tracer Swarm (SoA) | `rayon` / `SearchGrid`|

---

## 6. Material Tracking: The Tracer Swarm (Marker-in-Cell)

To track material movement and history (like plastic strain or composition), we use a **Marker-in-Cell** system.

### 6.1 Tracer Advection (RK2)
We move trillion of "virtual" particles (tracers) through the velocity field computed on the mesh. To balance speed and accuracy, we use the **Midpoint Method (Runge-Kutta 2)**:

1.  **Predictor (Midpoint):**
    $$\mathbf{x}_{mid} = \mathbf{x}_n + \mathbf{v}(\mathbf{x}_n) \frac{\Delta t}{2}$$
2.  **Corrector (Final):**
    $$\mathbf{x}_{n+1} = \mathbf{x}_n + \mathbf{v}(\mathbf{x}_{mid}) \Delta t$$

**Intuitive Understanding:**
If you are rowing a boat in a swirling river, you don't just look at the water current where you *are*. You look a bit ahead, see where the current is going, and adjust. RK2 does exactly this, preventing particles from "flying off" the curves in the flow.

### 6.2 High-Performance Search Grid
Finding which of the 1,000,000 elements contains a specific tracer is an expensive search ($O(N)$). Our **SearchGrid** uses spatial binning:
1.  Divide the domain into a 3D grid of "bins".
2.  Assign elements to bins based on their bounding boxes.
3.  When a tracer moves, we only check the elements in its local bin.

**Result:** Lookup time drops from $O(N)$ to **$O(1)$ average**, enabling simulations with millions of tracers to run in seconds.

### 6.3 Marker-to-Element (M2E) Mapping
The mesh provides the velocity field, but the tracers "carry" the material properties (e.g., density, viscosity, plastic softening).
*   **Property Averaging:** Elements compute their effective properties by averaging the properties of all tracers inside them.
*   **Compositional Shifting:** As tracers move, the "material" flows across the mesh, allowing us to simulate subduction, rifting, and mountain building without the mesh itself becoming overly tangled.
