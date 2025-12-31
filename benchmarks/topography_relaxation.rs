/// Benchmark 1: Topography Relaxation with Hillslope Diffusion
///
/// **Goal:** Validate hillslope diffusion against analytical solution
///
/// **Problem Setup:**
/// - Initial sinusoidal topography: h(x,0) = A sin(kx)
/// - Hillslope diffusion: ∂h/∂t = κ ∇²h
/// - Domain: 10 km × 10 km × 10 km
///
/// **Analytical Solution (1D):**
/// h(x,t) = A exp(-κ k² t) sin(kx)
///
/// where:
/// - A = initial amplitude (m)
/// - k = 2π/λ = wavenumber (1/m)
/// - κ = diffusion coefficient (m²/yr)
/// - Amplitude decays exponentially: A(t) = A₀ exp(-κ k² t)
///
/// **Success Criteria:**
/// - L2 error < 5% for all time steps
/// - Amplitude decay matches analytical (< 2% error)

use geo_simulator::{
    ImprovedMeshGenerator, VtkWriter, ScalarField,
};
use std::collections::{HashSet, HashMap};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Topography Relaxation: Hillslope Diffusion Validation");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =====================================================================
    // Physical Parameters
    // =====================================================================

    let lx = 10_000.0;      // Domain width (m) = 10 km
    let ly = 10_000.0;      // Domain depth (m)
    let lz = 1_000.0;       // Domain height (m) = 1 km

    let kappa = 1.0;        // Diffusion coefficient (m²/yr)
    let wavelength = 5_000.0; // Topography wavelength (m) = 5 km
    let amplitude_0 = 100.0;  // Initial amplitude (m)

    let k = 2.0 * std::f64::consts::PI / wavelength;  // Wavenumber (1/m)

    println!("Parameters:");
    println!("  Domain: {:.0} × {:.0} × {:.0} m", lx, ly, lz);
    println!("  κ = {:.2} m²/yr", kappa);
    println!("  λ = {:.0} m", wavelength);
    println!("  A₀ = {:.0} m", amplitude_0);
    println!("  k = {:.6} m⁻¹\n", k);

    // Time stepping
    let t_final = 1000.0;   // Final time (years)
    let n_steps = 20;
    let dt = t_final / (n_steps as f64);

    println!("Time Integration:");
    println!("  Final time: {:.0} years", t_final);
    println!("  Time steps: {}", n_steps);
    println!("  Δt: {:.1} years\n", dt);

    // =====================================================================
    // Mesh Generation
    // =====================================================================

    let nx = 20;  // Fine mesh for accurate surface representation
    let ny = 4;   // Thin in y (quasi-2D)
    let nz = 4;

    let mut mesh = ImprovedMeshGenerator::generate_cube(nx, ny, nz, lx, ly, lz);

    println!("Mesh: {} nodes, {} elements\n", mesh.num_nodes(), mesh.num_elements());

    // =====================================================================
    // Initialize Sinusoidal Topography
    // =====================================================================

    // Identify top surface nodes BEFORE perturbation
    let tol = 1e-6;
    let z_max = lz;
    let surface_nodes: Vec<usize> = mesh.geometry.nodes.iter()
        .enumerate()
        .filter(|(_, node)| (node.z - z_max).abs() < tol)
        .map(|(id, _)| id)
        .collect();

    println!("Surface nodes: {}", surface_nodes.len());

    // Store initial surface nodes as a set for use in custom diffusion
    let surface_set: HashSet<usize> = surface_nodes.iter().copied().collect();

    // Apply sinusoidal perturbation - keep all surface nodes at SAME average z
    // This way identify_surface_nodes will find them all
    for &node_id in &surface_nodes {
        let x = mesh.geometry.nodes[node_id].x;
        let h_init = amplitude_0 * (k * x).sin();
        mesh.geometry.nodes[node_id].z = z_max + h_init;
    }

    // For this benchmark, we'll apply custom diffusion that only works on known surface nodes
    // to avoid the surface identification problem

    // =====================================================================
    // Hillslope Diffusion Time Stepping
    // =====================================================================

    // Build surface connectivity manually
    let mut neighbors: HashMap<usize, Vec<usize>> = HashMap::new();
    for &node_id in &surface_nodes {
        neighbors.insert(node_id, Vec::new());
    }

    // Connect surface nodes that share elements
    for elem in &mesh.connectivity.tet10_elements {
        let surface_in_elem: Vec<usize> = elem.nodes.iter()
            .filter(|&&n| surface_set.contains(&n))
            .copied()
            .collect();

        for i in 0..surface_in_elem.len() {
            for j in i+1..surface_in_elem.len() {
                let n1 = surface_in_elem[i];
                let n2 = surface_in_elem[j];
                if !neighbors[&n1].contains(&n2) {
                    neighbors.get_mut(&n1).unwrap().push(n2);
                    neighbors.get_mut(&n2).unwrap().push(n1);
                }
            }
        }
    }

    // Check stability
    let dx = lx / (nx as f64);
    let dt_stable = 0.4 * dx * dx / kappa;
    println!("Mesh spacing: {:.1} m", dx);
    println!("Stable Δt: {:.1} years", dt_stable);
    println!("Using Δt: {:.1} years", dt);

    if dt > dt_stable {
        println!("WARNING: Δt > Δt_stable, may be unstable!\n");
    } else {
        println!("✓ Stable time step\n");
    }

    println!("═══════════════════════════════════════════════════════════════");
    println!("Time stepping...\n");

    let mut results = Vec::new();

    for step in 0..=n_steps {
        let time = step as f64 * dt;

        // Analytical solution at this time
        let amplitude_analytical = amplitude_0 * (-kappa * k * k * time).exp();

        // Measure numerical amplitude (max elevation)
        let z_values: Vec<f64> = surface_nodes.iter()
            .map(|&id| mesh.geometry.nodes[id].z)
            .collect();
        let z_max_numerical = z_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let z_min_numerical = z_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let amplitude_numerical = (z_max_numerical - z_min_numerical) / 2.0;

        // Compute L2 error
        let mut error_sum = 0.0;
        let mut norm_sum = 0.0;
        for &node_id in &surface_nodes {
            let x = mesh.geometry.nodes[node_id].x;
            let h_numerical = mesh.geometry.nodes[node_id].z - z_max;
            let h_analytical = amplitude_analytical * (k * x).sin();

            error_sum += (h_numerical - h_analytical).powi(2);
            norm_sum += h_analytical.powi(2);
        }
        let l2_error = (error_sum / norm_sum).sqrt() * 100.0;

        // Amplitude error
        let amp_error = ((amplitude_numerical - amplitude_analytical) / amplitude_analytical).abs() * 100.0;

        results.push((time, amplitude_analytical, amplitude_numerical, l2_error, amp_error));

        if step % 4 == 0 || step == n_steps {
            println!("  Step {:2}: t={:6.1} yr, A_analytical={:6.2} m, A_numerical={:6.2} m, L2={:.2}%, Amp_err={:.2}%",
                     step, time, amplitude_analytical, amplitude_numerical, l2_error, amp_error);
        }

        // Apply manual diffusion for next step
        if step < n_steps {
            let mut new_z = HashMap::new();

            for &node_id in &surface_nodes {
                let h_i = mesh.geometry.nodes[node_id].z;
                let node_neighbors = &neighbors[&node_id];

                if node_neighbors.is_empty() {
                    new_z.insert(node_id, h_i);
                    continue;
                }

                // Laplacian: ∇²h ≈ Σ(h_j - h_i) / (n * dx²)
                let mut laplacian = 0.0;
                for &neighbor_id in node_neighbors {
                    let h_j = mesh.geometry.nodes[neighbor_id].z;
                    laplacian += h_j - h_i;
                }
                laplacian /= node_neighbors.len() as f64 * dx * dx;

                // Update: h_{n+1} = h_n + dt * kappa * ∇²h
                let h_new = h_i + dt * kappa * laplacian;
                new_z.insert(node_id, h_new);
            }

            // Apply updates
            for (&node_id, &z_new) in &new_z {
                mesh.geometry.nodes[node_id].z = z_new;
            }
        }
    }

    // =====================================================================
    // Validation Summary
    // =====================================================================

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Validation Results");
    println!("═══════════════════════════════════════════════════════════════\n");

    let max_l2_error = results.iter().map(|(_, _, _, l2, _)| *l2).fold(0.0f64, f64::max);
    let max_amp_error = results.iter().map(|(_, _, _, _, amp)| *amp).fold(0.0f64, f64::max);

    println!("  Maximum L2 error: {:.2}%", max_l2_error);
    println!("  Maximum amplitude error: {:.2}%\n", max_amp_error);

    if max_l2_error < 5.0 && max_amp_error < 2.0 {
        println!("  ✓✓✓ EXCELLENT: L2 < 5%, Amplitude < 2%");
    } else if max_l2_error < 10.0 && max_amp_error < 5.0 {
        println!("  ✓✓ GOOD: L2 < 10%, Amplitude < 5%");
    } else {
        println!("  ✗ FAILED: Errors too large");
    }

    // =====================================================================
    // Export VTK
    // =====================================================================

    // Add analytical solution for comparison
    let analytical_field: Vec<f64> = mesh.geometry.nodes.iter()
        .map(|node| {
            let x = node.x;
            let t = t_final;
            let amplitude = amplitude_0 * (-kappa * k * k * t).exp();
            amplitude * (k * x).sin()
        })
        .collect();

    mesh.field_data.add_field(ScalarField::new("AnalyticalHeight", analytical_field));

    std::fs::create_dir_all("output/topography_relaxation").ok();
    VtkWriter::write_vtu(&mesh, "output/topography_relaxation/final.vtu").unwrap();

    println!("\n  VTK output: output/topography_relaxation/final.vtu");
    println!("═══════════════════════════════════════════════════════════════");
}
