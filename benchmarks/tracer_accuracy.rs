use geo_simulator::{
    ImprovedMeshGenerator, DofManager, VtkWriter, TracerSwarm, SearchGrid,
};
use nalgebra::Point3;
use std::time::Instant;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Tracer Accuracy: 3D Vortex Benchmark");
    println!("═══════════════════════════════════════════════════════════════");

    // 1. Setup Mesh (100m cube)
    let l_box = 100.0;
    let res = 10;
    let mesh = ImprovedMeshGenerator::generate_cube(res, res, res, l_box, l_box, l_box);
    println!("Mesh generated: {} elements", mesh.num_elements());

    // 2. Setup Tracer Swarm
    let mut swarm = TracerSwarm::with_capacity(5000);
    
    // Add tracers in a sphere at (25, 50, 50) with radius 10
    let center = Point3::new(25.0, 50.0, 50.0);
    let radius = 10.0;
    
    let mut n_tracers = 0;
    for ix in 0..20 {
        for iy in 0..20 {
            for iz in 0..20 {
                let p = Point3::new(
                    15.0 + ix as f64 * 1.0,
                    40.0 + iy as f64 * 1.0,
                    40.0 + iz as f64 * 1.0,
                );
                let dist = (p - center).norm();
                if dist <= radius {
                    swarm.add_tracer(p, 1);
                    n_tracers += 1;
                }
            }
        }
    }
    println!("Tracers initialized: {} particles", n_tracers);

    // 3. Setup Analytical Velocity Field (Rotation around Z axis)
    // Rotation center (50, 50)
    // v_x = -(y - 50)
    // v_y = (x - 50)
    let mut dof_mgr = DofManager::new(mesh.num_nodes(), 3);
    let mut velocity_vec = vec![0.0; dof_mgr.total_dofs()];
    
    for i in 0..mesh.num_nodes() {
        let p = mesh.geometry.nodes[i];
        let vx = -(p.y - 50.0);
        let vy = p.x - 50.0;
        
        velocity_vec[dof_mgr.global_dof(i, 0)] = vx;
        velocity_vec[dof_mgr.global_dof(i, 1)] = vy;
        velocity_vec[dof_mgr.global_dof(i, 2)] = 0.0;
    }

    // 4. Build Search Grid
    let start_grid = Instant::now();
    let grid = SearchGrid::build(&mesh, [15, 15, 15]);
    println!("Search grid built in {:?}", start_grid.elapsed());

    // 5. Advection Loop (Full circle = 2*PI seconds)
    let dt = 0.05;
    let total_time = 2.0 * std::f64::consts::PI;
    let steps = (total_time / dt).ceil() as usize;
    
    println!("Starting advection loop: {} steps (dt={})", steps, dt);
    
    std::fs::create_dir_all("output/tracer_vortex").unwrap();
    
    let start_adv = Instant::now();
    for step in 0..=steps {
        if step % 20 == 0 || step == steps {
            let filename = format!("output/tracer_vortex/tracers_{:04}.vtu", step);
            VtkWriter::write_tracers_vtu(&swarm, &filename).unwrap();
            println!("  Step {}: Tracers saved to {}", step, filename);
        }

        if step < steps {
            swarm.advect_rk2(&mesh, &grid, &dof_mgr, &velocity_vec, dt);
        }
    }
    
    let elapsed = start_adv.elapsed();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmark Completed in {:?}", elapsed);
    println!("  Avg time per step: {:?}", elapsed / steps as u32);
    
    // Check drift
    let mut max_drift = 0.0;
    for i in 0..swarm.num_tracers() {
        let p_final = Point3::new(swarm.x[i], swarm.y[i], swarm.z[i]);
        let p_initial = Point3::new(swarm.initial_x[i], swarm.initial_y[i], swarm.initial_z[i]);
        let drift = (p_final - p_initial).norm();
        if drift > max_drift {
            max_drift = drift;
        }
    }
    println!("  Maximum Drift Error: {:.6} m", max_drift);
    println!("═══════════════════════════════════════════════════════════════");
}
