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

### 3.1 The Tet10 Element: Anatomy and Advantage

**Node Layout (10 nodes total):**
```
                  3 (apex)
                  ●
                 /│\
                / │ \
               /  │  \
            9 ●   │   ● 8
             /    │    \
            /   7 ●     \
           /    .'  '.   \
          /   .'  0   '.  \
         /  .'    ●     '. \
      0 ● ●───────●───────● ● 2
         4       5       6
          (base triangle)

Corner nodes (4):     0, 1, 2, 3
Mid-edge nodes (6):   4 (0→1), 5 (0→2), 6 (1→2)
                      7 (0→3), 8 (1→3), 9 (2→3)
```

**Comparison:**
*   **Tet4 (Linear):** 4 nodes at corners. The edges stay perfectly straight.
    - Displacement field: Linear interpolation
    - Strain field: **Constant** per element (piecewise constant)
    - Stress field: Constant per element (unrealistic jumps between elements)

*   **Tet10 (Quadratic):** 10 nodes (4 corners + 6 mid-edges). The edges can **curve**.
    - Displacement field: Quadratic interpolation
    - Strain field: **Linear** per element (smooth variation)
    - Stress field: Linear per element (much more realistic)

**Intuitive Understanding (Locking):**
Imagine trying to model a subducting slab bending into the mantle.
- A **Tet4** is like a Lego brick. To make a smooth curve, you need thousands of tiny bricks. If you don't have enough, the mesh "locks" because the straight edges can't physically represent the curve.
- A **Tet10** is like a flexible piece of wood. It can curve with much less effort. You get high accuracy with a much coarser (and thus potentially faster) mesh.

**Why Tet10 is Essential for Geodynamics:**
1. **Incompressibility:** Stokes flow requires ∇·v = 0. Tet4 elements "lock" (become overly stiff) → wrong solution. Tet10 naturally satisfies incompressibility.
2. **Bending:** Subduction, rifting, and folding involve **curvature**. Tet10 captures this with far fewer elements.
3. **Accuracy:** For the same mesh density, Tet10 is ~10x more accurate than Tet4 for stress/strain.

### 3.2 Shape Functions: The DNA of the Element

The **shape functions** $N_i(\xi, \eta, \zeta)$ define how field variables (displacement, pressure, temperature) are interpolated within the element.

**Reference Coordinates (Barycentric):**
We work in a "reference tetrahedron" with coordinates $(\xi, \eta, \zeta, \lambda)$ where:
$$\lambda = 1 - \xi - \eta - \zeta$$

These are **barycentric coordinates** (sum to 1). Each coordinate represents the "weight" of the corresponding corner node.

**Schematic:**
```
Reference Space:              Physical Space:
  ζ                              z
  ↑   3                          ↑  x₃
  │  /                           │ /
  │ /                            │/
  │/_____ η                    x₀/_____ x₂
  0      ξ                        x₁

  Perfect unit                  Distorted/stretched
  tetrahedron                   real element
```

**Tet10 Shape Functions (10 total):**

**Corner nodes (quadratic bubbles):**
$$N_0 = \lambda(2\lambda - 1)$$
$$N_1 = \xi(2\xi - 1)$$
$$N_2 = \eta(2\eta - 1)$$
$$N_3 = \zeta(2\zeta - 1)$$

**Mid-edge nodes (quadratic bridges):**
$$N_4 = 4\lambda\xi \quad \text{(edge 0→1)}$$
$$N_5 = 4\lambda\eta \quad \text{(edge 0→2)}$$
$$N_6 = 4\xi\eta \quad \text{(edge 1→2)}$$
$$N_7 = 4\lambda\zeta \quad \text{(edge 0→3)}$$
$$N_8 = 4\xi\zeta \quad \text{(edge 1→3)}$$
$$N_9 = 4\eta\zeta \quad \text{(edge 2→3)}$$

**Key Properties:**
1. **Partition of unity:** $\sum_{i=0}^{9} N_i = 1$ (always!)
2. **Kronecker delta:** $N_i(\mathbf{x}_j) = \delta_{ij}$ (equals 1 at its node, 0 at others)
3. **Quadratic:** Maximum power is 2 (e.g., $\xi^2$)

**Visualization of a Mid-Edge Function:**
```
    N₄(ξ, η, ζ) = 4λξ  (edge 0→1)

    Value along edge 0→1:
    Node 0 (λ=1, ξ=0): N₄ = 0     ●────────────●
    Node 4 (λ=½, ξ=½): N₄ = 1        (parabola)
    Node 1 (λ=0, ξ=1): N₄ = 0     0     4      1

    Peak at midpoint!
```

### 3.3 Isoparametric Mapping: Reference ↔ Physical

We use the **same** shape functions to map both **geometry** and **field variables**.

**Geometric Mapping (10 nodes define the element shape):**
$$\mathbf{x}(\xi, \eta, \zeta) = \sum_{i=0}^{9} N_i(\xi, \eta, \zeta) \, \mathbf{x}_i$$

where $\mathbf{x}_i = (x_i, y_i, z_i)$ are the **physical positions** of the 10 nodes.

**Field Variable Mapping (e.g., displacement):**
$$\mathbf{u}(\xi, \eta, \zeta) = \sum_{i=0}^{9} N_i(\xi, \eta, \zeta) \, \mathbf{u}_i$$

where $\mathbf{u}_i$ are the **displacement vectors** at the 10 nodes.

**Example in 1D (for intuition):**
```
Reference:  ξ = 0 ──●──●──●── ξ = 1
                    0  1  2  (3 nodes)

Physical:   x = ∑ Nᵢ(ξ) · xᵢ

If x₀=0, x₁=3, x₂=10 (quadratic spacing):
  ξ = 0.0 → x = 0
  ξ = 0.5 → x = 2.5  (not 5! nonlinear map)
  ξ = 1.0 → x = 10
```

### 3.4 The Jacobian Matrix: The Stretching Tensor

The **Jacobian matrix** $\mathbf{J}$ measures how the reference element is **stretched, rotated, and skewed** to fit the physical space.

**Definition:**
$$\mathbf{J} = \frac{\partial \mathbf{x}}{\partial \boldsymbol{\xi}} = \begin{bmatrix}
\frac{\partial x}{\partial \xi} & \frac{\partial x}{\partial \eta} & \frac{\partial x}{\partial \zeta} \\[0.5em]
\frac{\partial y}{\partial \xi} & \frac{\partial y}{\partial \eta} & \frac{\partial y}{\partial \zeta} \\[0.5em]
\frac{\partial z}{\partial \xi} & \frac{\partial z}{\partial \eta} & \frac{\partial z}{\partial \zeta}
\end{bmatrix}$$

**Computing J (using chain rule):**
$$\frac{\partial x}{\partial \xi} = \sum_{i=0}^{9} \frac{\partial N_i}{\partial \xi} \, x_i$$

and similarly for all 9 components.

**Example Components:**
$$\frac{\partial N_0}{\partial \xi} = \frac{\partial}{\partial \xi}[\lambda(2\lambda - 1)] = \frac{\partial}{\partial \xi}[(1-\xi-\eta-\zeta)(2-2\xi-2\eta-2\zeta)]$$
$$= -1 \cdot (2-2\xi-2\eta-2\zeta) + (1-\xi-\eta-\zeta) \cdot (-2) = -4 + 4\xi + 2\eta + 2\zeta$$

**Physical Interpretation of J:**
```
Small change in reference space:
  dξ = [dξ, dη, dζ]ᵀ

Corresponding change in physical space:
  dx = J · dξ

Example:
  Move dξ = [0.01, 0, 0] in reference
  → dx = J[:, 0] * 0.01 in physical

  First column of J = tangent vector along ξ-direction
```

**Determinant: The Volume Ratio:**
$$\det(\mathbf{J}) = \frac{dV_{physical}}{dV_{reference}}$$

**Interpretation:**
- $\det(\mathbf{J}) > 0$: Element is valid (preserves orientation)
- $\det(\mathbf{J}) = 0$: Element is **degenerate** (collapsed to a plane)
- $\det(\mathbf{J}) < 0$: Element is **inverted** (inside-out) → **FATAL**

**Geometric Meaning:**
```
det(J) = Volume scaling factor

Reference tet volume = 1/6 (unit tetrahedron)
Physical tet volume  = ∫ det(J) dV_ref

If uniform stretching by factor 2:
  det(J) = 2³ = 8  (volume increases 8x)
```

### 3.5 Strain-Displacement Matrix: Computing Physics

To compute strains (needed for stress), we need derivatives of displacements **in physical space**:
$$\boldsymbol{\epsilon} = \frac{1}{2}\left(\nabla \mathbf{u} + (\nabla \mathbf{u})^T\right)$$

But shape functions are defined in **reference space**! We need the **inverse Jacobian**:
$$\frac{\partial N_i}{\partial x} = \mathbf{J}^{-1} \frac{\partial N_i}{\partial \xi}$$

**The B-matrix (Strain-Displacement operator):**
For 3D, the strain vector is:
$$\boldsymbol{\epsilon} = \begin{bmatrix} \epsilon_{xx} \\ \epsilon_{yy} \\ \epsilon_{zz} \\ \gamma_{xy} \\ \gamma_{xz} \\ \gamma_{yz} \end{bmatrix} = \mathbf{B} \, \mathbf{u}_{elem}$$

where $\mathbf{u}_{elem} = [u_0^x, u_0^y, u_0^z, u_1^x, \ldots, u_9^z]^T$ (30 DOFs for Tet10).

**Structure of B (6 rows × 30 columns):**
Each node contributes a 6×3 block:
$$\mathbf{B}_i = \begin{bmatrix}
\frac{\partial N_i}{\partial x} & 0 & 0 \\[0.3em]
0 & \frac{\partial N_i}{\partial y} & 0 \\[0.3em]
0 & 0 & \frac{\partial N_i}{\partial z} \\[0.3em]
\frac{\partial N_i}{\partial y} & \frac{\partial N_i}{\partial x} & 0 \\[0.3em]
\frac{\partial N_i}{\partial z} & 0 & \frac{\partial N_i}{\partial x} \\[0.3em]
0 & \frac{\partial N_i}{\partial z} & \frac{\partial N_i}{\partial y}
\end{bmatrix}$$

**Complete workflow:**
```
1. Evaluate shape functions Nᵢ(ξ, η, ζ) at quadrature point
2. Compute derivatives ∂Nᵢ/∂ξ, ∂Nᵢ/∂η, ∂Nᵢ/∂ζ
3. Build Jacobian J from node positions xᵢ
4. Compute J⁻¹ (3×3 matrix inverse)
5. Transform derivatives: ∂Nᵢ/∂x = J⁻¹ · [∂Nᵢ/∂ξ, ∂Nᵢ/∂η, ∂Nᵢ/∂ζ]ᵀ
6. Assemble B-matrix from physical derivatives
7. Compute strain: ε = B · u_elem
8. Compute stress: σ = C : ε (constitutive law)
```

### 3.6 Numerical Integration: Gauss Quadrature

Integrals over the element (stiffness matrix, force vector) are evaluated numerically:
$$\int_{V_{elem}} f(\mathbf{x}) \, dV = \int_{V_{ref}} f(\boldsymbol{\xi}) \, \det(\mathbf{J}) \, dV_{ref}$$

We use **Gauss quadrature points** $\{\boldsymbol{\xi}_q, w_q\}$:
$$\int f \, dV \approx \sum_{q=1}^{N_{quad}} w_q \, f(\boldsymbol{\xi}_q) \, \det(\mathbf{J}_q)$$

**Standard Quadrature Rules for Tetrahedra:**

| Order | Points | Integration Accuracy      | Use Case                    |
|-------|--------|---------------------------|-----------------------------|
| 1     | 1      | Exact for linear          | Mass matrix (rarely used)   |
| 2     | 4      | Exact for quadratic       | Tet10 stiffness (standard)  |
| 3     | 5      | Exact for cubic           | Nonlinear problems          |
| 5     | 15     | Exact for degree 5        | Very high accuracy          |

**4-Point Gauss Rule (most common for Tet10):**
Quadrature points in barycentric coordinates:
```
α = (5 - √5) / 20 ≈ 0.1381966
β = (5 + 3√5) / 20 ≈ 0.5854102

Point 0: (α, α, α), weight = 1/4
Point 1: (β, α, α), weight = 1/4
Point 2: (α, β, α), weight = 1/4
Point 3: (α, α, β), weight = 1/4
```

**Why not just use corner nodes?**
- Corner points are **biased** (not symmetric within element)
- Gauss points are **optimal** (minimize integration error)
- For Tet10, Gauss points capture the quadratic variation accurately

**Schematic:**
```
         3
        /│\
       / │ \
      /  q₃ \        q₀, q₁, q₂, q₃ = Gauss points
     / q₀  q₁\       (NOT at corners!)
    /___●_●___\
   0   q₂      2
        1
```

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

---

## 7. Plasticity: When Rocks Break

Rocks behave elastically and viscously up to a critical stress level. Beyond that, they **yield** (permanent deformation). This is crucial for modeling faults, shear zones, and localization.

### 7.1 Mohr-Coulomb Yield Criterion

The **Mohr-Coulomb** criterion states that rocks fail when shear stress exceeds frictional resistance:

$$\tau_{yield} = c + P \tan(\phi)$$

Where:
- $\tau_{yield}$ = yield stress (Pa)
- $c$ = cohesion (initial strength, ~10-50 MPa for crust)
- $P$ = pressure (confining stress)
- $\phi$ = internal friction angle (typically 30° for crustal rocks)

**Conceptual Understanding:**
```
    τ (shear stress)
    ↑
    │     /  Failure envelope: τ = c + P·tan(φ)
    │    /
    │   /   ← Yielding (plastic deformation)
    │  /
    │ /______ Safe zone (elastic/viscous)
    │ c
    └──────────→ P (pressure)
```

**Physical Intuition:**
- **Cohesion (c):** Like glue holding grains together. Even with zero pressure, the rock has some strength.
- **Friction angle (φ):** Deeper rocks (higher P) are harder to shear. It's like stacking books – the more weight on top, the harder it is to slide the bottom book.
- **Pressure dependence:** This is why earthquakes nucleate at shallow depths (low P, easy to fail) but not in the deep mantle (high P locks things up).

### 7.2 Effective Plastic Viscosity

Rather than explicitly tracking the yield surface, we use a **regularized viscoplastic** approach. When stress exceeds yield, we replace the standard viscosity with an effective plastic viscosity:

$$\mu_{plastic} = \frac{\tau_{yield}}{2 \dot{\epsilon}_{II}}$$

Where $\dot{\epsilon}_{II} = \sqrt{\frac{1}{2} \dot{\epsilon}_{ij} \dot{\epsilon}_{ij}}$ is the strain rate invariant.

**The effective viscosity becomes:**
$$\mu_{eff} = \min(\mu_{viscous}, \mu_{plastic})$$

**Conceptual Understanding:**
```
   μ (viscosity)
    ↑
    │ μ_viscous ────────────────  Viscous flow (slow creep)
    │
    │                  /  μ_plastic drops as strain rate increases
    │                /
    │              /  ← Localization zone (shear band)
    │            /
    │__________/
    └──────────────→ ε̇_II (strain rate)
```

**Why this works:**
- **Low strain rates:** $\mu_{plastic}$ is huge → rock behaves viscously
- **High strain rates:** $\mu_{plastic}$ drops → rock weakens → **strain localizes**
- This naturally produces **shear bands** (like faults) without explicit tracking

### 7.3 Strain Softening: How Faults Form

Once yielding begins, rocks accumulate plastic strain $\epsilon_p$. As damage accumulates, cohesion **decreases**:

$$c(\epsilon_p) = c_{min} + (c_0 - c_{min}) \exp\left(-\frac{\epsilon_p}{\epsilon_{ref}}\right)$$

Where:
- $c_0$ = initial cohesion (20-50 MPa)
- $c_{min}$ = residual cohesion after complete softening (1-5 MPa)
- $\epsilon_{ref}$ = reference strain (~0.1, meaning 10% strain softens the rock)

**Evolution of plastic strain:**
$$\frac{D\epsilon_p}{Dt} = \dot{\epsilon}_{II} \quad \text{(only when yielding)}$$

**Schematic:**
```
    c (cohesion)
    ↑
    │ c₀ ●────╮
    │         │  Exponential weakening
    │         │  as plastic strain accumulates
    │         ╰─────●─────
    │               c_min
    └──────────────────→ ε_p (plastic strain)
              ε_ref
```

**Physical Intuition:**
- **Initial state:** Rock is pristine, strong ($c = c_0$)
- **First yield:** Micro-cracks form, cohesion starts dropping
- **Mature fault:** After $\epsilon_p \sim 1$, the rock is fully damaged, behaves like sand ($c \approx c_{min}$)

**Why this creates shear bands:**
1. Small perturbation (e.g., geometry, weak seed) → slightly higher strain rate
2. Higher strain → plasticity activates → cohesion drops
3. Lower cohesion → easier to deform → **more strain concentrates**
4. **Positive feedback** → narrow zone of extreme weakening = **fault**

### 7.4 Implementation: The Tracer-Element Workflow

```
┌─────────────┐
│  Tracers    │  Carry: material ID, ε_p
│  (Lagrange) │
└──────┬──────┘
       │ M2E (Marker-to-Element)
       ↓
┌─────────────┐
│  Elements   │  Compute: σ_yield(ε_p), μ_eff, ε̇_II
│  (Euler)    │
└──────┬──────┘
       │ Solve: ∇·σ + ρg = 0
       ↓
┌─────────────┐
│  Velocity   │  Advect tracers, update ε_p
│  Field      │
└─────────────┘
```

**Key steps:**
1. **Before solve:** Tracers transfer their plastic strain to elements
2. **During solve:** Elements compute $\mu_{eff}$ based on average $\epsilon_p$
3. **After solve:** Tracers advect, accumulate new plastic strain if yielding

---

## 8. Surface Processes: Hillslope Diffusion

The Earth's surface isn't static. Erosion and sediment transport reshape topography on timescales that compete with tectonics.

### 8.1 The Diffusion Model

Hillslope transport is modeled as **down-slope diffusion**:

$$\frac{\partial h}{\partial t} = \kappa \nabla^2 h$$

Where:
- $h(x,y,t)$ = surface elevation (m)
- $\kappa$ = diffusivity (m²/yr) – controls erosion rate
  - Typical values: 0.1-1 m²/yr (soil creep, bioturbation)
  - Higher for glaciated terrain (~10 m²/yr)

**Conceptual Understanding:**
```
Before:        After (diffusion):
  /\              /‾‾\
 /  \      →     /    \
/____\          /______\

Sharp peaks → Smooth hills
```

**Physical Processes Captured:**
- **Soil creep:** Gravity-driven downslope movement
- **Bioturbation:** Animals/plants mixing soil
- **Rain splash:** Sediment detachment and transport
- **NOT captured:** Channelized flow (rivers), landslides

### 8.2 Discrete Laplacian on Unstructured Mesh

On a triangulated surface, the Laplacian is:

$$\nabla^2 h_i = \frac{1}{A_i} \sum_{j \in N(i)} w_{ij} (h_j - h_i)$$

Where:
- $A_i$ = Voronoi area around node $i$
- $w_{ij}$ = cotangent weights from adjacent triangles
- $N(i)$ = neighbors of node $i$

**Algorithm:**
1. Extract top surface nodes from 3D mesh
2. Build 2D connectivity (which nodes share an edge on the surface)
3. Compute Laplacian using cotangent formula
4. Update $z$-coordinates: $h^{n+1} = h^n + \Delta t \cdot \kappa \nabla^2 h^n$

**Schematic:**
```
        j₁
        ●
       /│\
      / │ \    Node i at center
     /  │  \   Neighbors: j₁, j₂, j₃
    /   ●i  \  Δh_i ∝ Σ(h_j - h_i)
   /  /   \  \
  ● j₃     ● j₂
```

### 8.3 Coupling to Tectonics

**Time scales:**
- **Tectonics:** Extension at 1 cm/yr → significant in 1 Myr
- **Erosion:** $\kappa = 1$ m²/yr → smooths 1 km mountain in ~1 Myr

**Implementation strategy:**
- Apply diffusion **every N steps** (e.g., every 10 steps ~ 50 kyr)
- Prevents excessive smoothing between output frames
- Allows topography to develop from tectonic deformation

**Effect on simulation:**
- **Without erosion:** Mountains grow unbounded, mesh distorts severely
- **With erosion:** Topography reaches **dynamic equilibrium** (uplift ≈ erosion)
- More realistic stress distribution at base of crust

---

## 9. Isostatic Support: Winkler Foundation

When crust thickens or thins, it must be supported by the underlying mantle. The **Winkler foundation** is a simple isostatic model.

### 9.1 The "Floating Block" Analogy

Imagine a wooden block floating in water:
- Push it down → water pushes back (buoyancy)
- Lift it up → it sinks back down

The restoring force is proportional to displacement:
$$F_{restore} = -k \cdot \Delta z$$

This is **local isostasy** – each column of crust responds independently to vertical load.

### 9.2 Mathematical Formulation

The Winkler foundation applies a restoring force to the bottom boundary:

$$F_z(x,y) = -k \cdot (z - z_{ref})$$

Where:
- $k$ = foundation stiffness (Pa/m)
  - Typical value: $k = \frac{\Delta\rho \cdot g}{L}$ where $L$ is flexural wavelength
  - For $\Delta\rho = 600$ kg/m³, $g = 10$ m/s², $L = 100$ km: $k \approx 6 \times 10^7$ Pa/m
- $z_{ref}$ = reference elevation (initial configuration)
- $\Delta z = z - z_{ref}$ = vertical deflection

**Schematic:**
```
    Crust (lighter)
─────────────────────  ← Free surface (topography)
█████████████████████
█████████████████████  ← Bottom boundary
─ ─ ─ ─ ─ ─ ─ ─ ─ ─   ← Reference level (z_ref)
  ↑         ↑
  Springs (Winkler foundation)
  k·Δz    k·Δz

    Mantle (denser)
≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈
```

### 9.3 Physical Interpretation

**Airy Isostasy:**
The Winkler model approximates **Airy isostasy**, where:
$$\rho_{crust} \cdot h_{crust} = \rho_{mantle} \cdot h_{root}$$

**Example:** Tibetan Plateau
- Crust thickness: ~70 km (vs. ~30 km normal)
- Extra 40 km → surface elevation ~3 km
- Winkler force prevents unrealistic subsidence/uplift

**Limitations:**
- **Local:** Ignores lateral communication (no flexure)
- **Real Earth:** Lithosphere has finite strength → **flexural isostasy** (bending)
- For wavelengths < 100 km, Winkler is reasonable
- For larger scales (e.g., ice sheets), need full flexure equations

### 9.4 Implementation

```rust
// Initialize reference configuration
winkler.initialize_reference(&mesh);  // Stores z_ref for bottom nodes

// Each time step:
let f_winkler = winkler.compute_forces(&mesh, &dof_mgr);
for i in 0..f.len() {
    f[i] += f_winkler[i];  // Add to RHS
}
```

**Force computation:**
```rust
for &node_id in &bottom_nodes {
    let delta_z = current_z[node_id] - reference_z[node_id];
    f_z[node_id] = -k * delta_z;
}
```

**When to use:**
- ✓ Simulations with **free bottom boundary** (no fixed BC)
- ✓ Long timescales (> 1 Myr) where isostasy matters
- ✗ Short timescales (elastic rebound)
- ✗ When bottom is **fixed** (like in core complex with fixed base)

---

## 10. Mesh Quality and Smoothing

Large deformations distort elements. Bad elements → inaccurate results or solver failure. We need to **monitor** and **repair** mesh quality.

### 10.1 Mesh Quality Metrics

**Jacobian determinant (det J):**
The Jacobian maps reference coordinates to physical coordinates. For a valid element:
$$\det(\mathbf{J}) > 0$$

**What it means:**
```
Good Element (det J > 0):    Inverted Element (det J < 0):
     3                            3
    /│\                          /│\
   / │ \                        / │ \
  1──┼──2                      2──┼──1  ← Vertices "flipped"
     4                            4
```

**Quality metrics we track:**
1. **Min Jacobian:** Smallest det(J) across all quadrature points
   - Should be > 0 (positive)
   - If close to 0 → element is nearly degenerate
2. **Average Jacobian:** Overall mesh quality
3. **Inverted count:** Number of elements with det(J) ≤ 0 (FATAL)
4. **Degenerate count:** Elements with det(J) < threshold (e.g., 1% of average)

### 10.2 When Quality Degrades

**Typical evolution in geodynamic simulation:**
```
Step    0: min_J = 2.3e10  ✓ Perfect initial mesh
Step   50: min_J = 1.9e9   ✓ Mild distortion
Step  100: min_J = 5.4e8   ⚠ Getting worse
Step  150: min_J = 1.2e7   ⚠ DANGER - trigger smoothing
Step  200: min_J = -3.4e5  ✗ INVERTED - simulation failure
```

**Triggers for intervention:**
- min_J < 10% of average → **needs smoothing**
- min_J < 1% of average → **critical**
- Any inverted elements → **remeshing required**

### 10.3 Laplacian Smoothing

**Idea:** Move each node toward the average position of its neighbors.

**Algorithm:**
```
For each interior node i:
    x_new[i] = (1-λ)·x_old[i] + λ·(average of neighbor positions)
```

Where:
- $\lambda$ = relaxation factor (0 < λ < 1)
  - Typical: λ = 0.5-0.7
  - Larger λ → more aggressive smoothing

**Schematic (2D example):**
```
Before:                  After (λ=0.5):
  ●                         ●
 /│\                       /│\
● │ ●   →   Node moves    ●─┼─●  ← More regular
 \│/        toward center  \│/
  ●                         ●
```

**Boundary handling:**
- **Fixed boundaries:** Don't move (BCs must be preserved)
- **Free surface:** Move freely to improve quality
- **Internal nodes only:** Prevents changing domain shape

### 10.4 Advanced: Curvature-Aware Smoothing

Standard Laplacian can **over-smooth** curved features (like the Moho). We can add weights:

$$\mathbf{x}_i^{new} = \mathbf{x}_i + \lambda \sum_{j \in N(i)} w_{ij} (\mathbf{x}_j - \mathbf{x}_i)$$

**Weighting schemes:**
1. **Uniform:** $w_{ij} = 1/|N(i)|$ (standard Laplacian)
2. **Inverse distance:** $w_{ij} \propto 1/|\mathbf{x}_j - \mathbf{x}_i|$ (preserves local features)
3. **Cotangent:** $w_{ij} = \cot(\alpha) + \cot(\beta)$ (angle-based, best for surface meshes)

### 10.5 Implementation Strategy

```rust
// Every N steps (e.g., every 10 steps):
let quality = assess_mesh_quality(&mesh);

if quality.needs_smoothing() {
    // Apply multiple iterations (typically 5-10)
    for iter in 0..num_iterations {
        smooth_mesh_laplacian(&mut mesh, lambda, &fixed_nodes);
    }

    // Re-assess
    let new_quality = assess_mesh_quality(&mesh);

    if new_quality.num_inverted > 0 {
        panic!("Smoothing failed to fix inverted elements!");
    }
}
```

**Trade-offs:**
- ✓ **Fast:** O(N) per iteration
- ✓ **Local:** Doesn't require global solve
- ✓ **Effective:** Fixes mild distortion
- ✗ **Limited:** Can't fix severe tangling
- ✗ **May introduce errors:** Moving nodes changes solution slightly

**When smoothing isn't enough:**
- Severe distortion (min_J < 0.1% of average) → **Remeshing**
- Topological issues (self-intersecting elements) → **Adaptive refinement**
- Extreme localization (1-element-wide shear band) → **h-adaptivity**

---

## 12. Numerical Solvers: The "Brain" of the Simulator

Geodynamic problems are massive (millions of variables) and highly non-linear (viscosity depends on velocity). To solve them, we use a hierarchy of numerical methods.

### 12.1 JFNK: Jacobian-Free Newton-Krylov

Since rocks break and soften, the simulation's equations are non-linear:
$$F(\mathbf{u}) = 0$$

Where $F$ is our set of physics laws and $\mathbf{u}$ is the velocity field we want to find.

**The Newton Method:**
To solve this, we start with a guess $\mathbf{u}_k$ and find a correction $\delta \mathbf{u}$:
$$\mathbf{J} \, \delta \mathbf{u} = -F(\mathbf{u}_k)$$
$$\mathbf{u}_{k+1} = \mathbf{u}_k + \alpha \, \delta \mathbf{u}$$

Where $\mathbf{J} = \frac{\partial F}{\partial \mathbf{u}}$ is the **Jacobian** (the matrix of all possible derivatives).

**The "Jacobian-Free" Trick:**
Building the full Jacobian $\mathbf{J}$ for $10^{24}$ Pa·s viscosity is incredibly hard and memory-intensive. Instead, we notice that Krylov solvers (like BiCGSTAB) only need to know the result of a matrix-vector product $\mathbf{J} \mathbf{v}$.
We approximate this using the **Finite Difference** method:
$$\mathbf{J} \mathbf{v} \approx \frac{F(\mathbf{u} + \epsilon \mathbf{v}) - F(\mathbf{u})}{\epsilon}$$

*   **Intuition:** It's like feeling your way across a dark room. You don't need a map (the Jacobian) if you can just push in a direction ($\mathbf{v}$) and see what pushes back ($F$).

### 12.2 BiCGSTAB: The Iterative Engine

BiCGSTAB (Bi-Conjugate Gradient Stabilized) is an **iterative linear solver**. Instead of solving $\mathbf{J} \delta \mathbf{u} = -F$ in one giant step (which is too slow), it builds the solution bit-by-bit in a **Krylov Subspace**:
$$\mathcal{K}_m(\mathbf{J}, \mathbf{r}_0) = \text{span} \{ \mathbf{r}_0, \mathbf{J} \mathbf{r}_0, \mathbf{J}^2 \mathbf{r}_0, \dots \}$$

*   **Intuition:** Imagine trying to approximate a complex shape using only a few basic blocks. Each iteration of BiCGSTAB adds a new "block" (a new vector in the Krylov space) to better fit the true solution.

### 12.3 Preconditioning: "Lubricating" the Math

Large simulations are **ill-conditioned**. Viscosity contrasts ($10^{18}$ vs $10^{24}$) make the system extremely stiff. One part of the matrix might be $1,000,000\times$ larger than another.
A **Preconditioner** $\mathbf{M}$ transforms the problem:
$$\mathbf{M}^{-1} \mathbf{J} \, \delta \mathbf{u} = -\mathbf{M}^{-1} F(\mathbf{u}_k)$$

The goal is to choose $\mathbf{M}$ such that $\mathbf{M}^{-1} \mathbf{J} \approx \mathbf{I}$ (the identity matrix).

#### 12.4 Jacobi Preconditioner (The Simple Scaler)
$\mathbf{M}$ is just the **diagonal** of the matrix:
$$\mathbf{M} = \text{diag}(\mathbf{J})$$

*   **Intuition:** It scales every equation so that the "units" match. It's like converting a mix of meters, kilometers, and millimeters all into meters before doing any math. 
*   **Performance:** Very fast to set up, but "blind" to how nodes interact with their neighbors.

#### 12.5 ILU(0) Preconditioner (The Local Solver)
ILU (Incomplete LU decomposition) performs a simplified version of standard Gaussian elimination:
$$\mathbf{J} \approx \mathbf{L} \mathbf{U}$$

In **ILU(0)**, we only calculate the factors for the positions where the original matrix $\mathbf{J}$ had non-zero values.

*   **Intuition:** It captures how each node is physically coupled to its neighbors. If you push on one corner of an element, ILU(0) "knows" exactly how the other corners should move.
*   **Why it's essential:** For high-viscosity simulations ($10^{24}$), Jacobi is too weak. ILU(0) provides the numerical "stiffness" needed to bridge huge property gaps, allowing the solver to converge in 100 iterations instead of failing.

```
┌─────────────────────────────────────────────────────┐
│  FREE SURFACE (topography develops)                 │
│  ↓ Hillslope diffusion (κ∇²h)                       │
├─────────────────────────────────────────────────────┤
│  CRUST (20 km)                                      │
│  • VEP rheology (η=1e23, c=20 MPa, φ=30°)          │
│  • Strain softening (c → 2 MPa)                     │
│  • Tracers carry ε_p                                │
│  ← Extension BC (1 cm/yr) │ → Extension BC          │
├─────────────────────────────────────────────────────┤
│  MANTLE (10 km)                                     │
│  • Weaker viscosity (η=1e21)                        │
│  • Denser (ρ=3300 kg/m³ vs 2700)                   │
├═════════════════════════════════════════════════════┤
│  FIXED BOTTOM (v_z = 0)                             │
└─────────────────────────────────────────────────────┘
```

**Physics timeline:**
1. **0-0.5 Myr:** Elastic loading, stress builds
2. **0.5-2 Myr:** Plastic yielding, shear zone nucleates
3. **2-5 Myr:** Mature fault, mantle upwelling, core exhumation
4. **Throughout:**
   - Mesh quality degrades → Laplacian smoothing (every 10 steps)
   - Surface diffusion smooths topography (every 10 steps)
   - JFNK solver handles nonlinearity (plasticity)

**Expected output:**
- Narrow shear zones (localization from softening)
- Metamorphic core dome (mantle upwelling)
- Flanking basins (extension + erosion)
- Realistic topography (tectonic uplift balanced by erosion)
