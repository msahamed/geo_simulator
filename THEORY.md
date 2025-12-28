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
