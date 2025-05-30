import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize Taichi
ti.init(arch=ti.gpu)  # Use GPU, change to ti.cpu if no GPU available

# Physical parameters
rho0 = 1000.0
cs = 88.5
alpha = 0.3
g = ti.Vector([0.0, -9.8])
bounds = ti.Vector([-0.5, 0.5, -0.5, 0.5])

# Particle parameters
spacing = 0.025
radius = spacing / 2.0
h = radius * 3.0
particle_num_x = int(0.4 / spacing)
particle_num_y = int(0.7 / spacing)
n_particles = particle_num_x * particle_num_y
max_neighbors = 60

# Define Taichi fields
x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)  # Position
x_pred = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)  # Predicted position
v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)  # Velocity
m = ti.field(dtype=ti.f32, shape=n_particles)  # Mass
rho = ti.field(dtype=ti.f32, shape=n_particles)  # Density
p = ti.field(dtype=ti.f32, shape=n_particles)  # Pressure
dx = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)  # Position correction
radiuses = ti.field(dtype=ti.f32, shape=n_particles)  # Radius

# Neighbor list
neighbors = ti.field(dtype=ti.i32, shape=(n_particles, max_neighbors))
neighbor_count = ti.field(dtype=ti.i32, shape=n_particles)

# Kernel functions
@ti.func
def poly6(l, h):
    result = 0.0
    q = l / h
    if 0.0 <= l <= h:
        r = 1.0 - q * q
        result = 4.0 / (ti.math.pi * h * h) * (r * r * r)
    return result

@ti.func
def spiky_gradient(l, h):
    result = 0.0
    if 0.0 <= l <= h and l > 1e-10:
        result = -30.0 / (ti.math.pi * h**5) * (h - l)**2
    return result

@ti.func
def spiky_hessian(l, h):
    result = 0.0
    if 0.0 <= l <= h and l > 1e-10:
        result = 60.0 / (ti.math.pi * h**5) * (h - l)
    return result

# Initialize particles
@ti.kernel
def init_particles():
    for i in range(n_particles):
        idx_x = i % particle_num_x
        idx_y = i // particle_num_x
        
        # Add small perturbation
        jitter = radius * 0.05
        noise_x = (ti.random() - 0.5) * 2.0 * jitter
        noise_y = (ti.random() - 0.5) * 2.0 * jitter
        
        x[i] = ti.Vector([
            -0.4 + (idx_x + 0.5) * spacing + noise_x,
            -0.4 + (idx_y + 0.5) * spacing + noise_y
        ])
        x_pred[i] = x[i]
        v[i] = ti.Vector([0.0, 0.0])
        
        radiuses[i] = radius
        m[i] = rho0 * 4.0 * radius * radius
        rho[i] = rho0
        p[i] = 0.0
        dx[i] = ti.Vector([0.0, 0.0])

# Find neighbors
@ti.kernel
def find_neighbors():
    for i in range(n_particles):
        cnt = 0
        for j in range(n_particles):
            if i != j and (x_pred[i] - x_pred[j]).norm() < h and cnt < max_neighbors:
                neighbors[i, cnt] = j
                cnt += 1
        neighbor_count[i] = cnt

# Prediction step
@ti.kernel
def predict(dt: ti.f32):
    for i in range(n_particles):
        x_pred[i] = x[i] + v[i] * dt + g * (dt * dt)

# Update density
@ti.kernel
def update_density():
    for i in range(n_particles):
        rho[i] = m[i] * poly6(0.0, h)
        for j_idx in range(neighbor_count[i]):
            j = neighbors[i, j_idx]
            l = (x_pred[i] - x_pred[j]).norm()
            rho[i] += m[j] * poly6(l, h)

# Update pressure
@ti.kernel
def update_pressure():
    for i in range(n_particles):
        if rho[i] > rho0:
            p[i] = rho0 * cs * cs / 7.0 * (ti.pow(rho[i] / rho0, 7.0) - 1.0)
        else:
            p[i] = 0.0

# Jacobi iteration
@ti.kernel
def jacobi_pressure(dt: ti.f32):
    # Reset position corrections
    for i in range(n_particles):
        dx[i] = ti.Vector([0.0, 0.0])
    
    # Pressure update
    for i in range(n_particles):
        f = ti.Vector([0.0, 0.0])
        J = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        
        for j_idx in range(neighbor_count[i]):
            j = neighbors[i, j_idx]
            r_ij = x_pred[i] - x_pred[j]
            l = r_ij.norm()
            
            if l > 1e-8:
                n = r_ij.normalized()
                
                # Pressure force
                p_coef = -m[j] * (p[i] + p[j]) / (2.0 * rho[j])
                f += n * (p_coef * spiky_gradient(l, h))
                
                # J matrix update
                nnt = n.outer_product(n)
                I_minus_nnt = ti.Matrix.identity(ti.f32, 2) - nnt
                J += p_coef * spiky_hessian(l, h) * (I_minus_nnt / l)
        
        # Position correction
        grad_norm = f.norm()
        if grad_norm > 1e-8:
            grad_dir = f / grad_norm
            sigma = J.norm()
            dl = grad_norm / (1.0 / (dt * dt) + sigma)
            dx[i] = dx[i] + grad_dir * dl

@ti.kernel
def jacobi_viscosity(dt: ti.f32):
    # Viscosity update
    for i in range(n_particles):
        f = ti.Vector([0.0, 0.0])
        J = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        
        for j_idx in range(neighbor_count[i]):
            j = neighbors[i, j_idx]
            r_ij = x_pred[i] - x_pred[j]
            l = r_ij.norm()
            
            if l > 1e-8:
                n = r_ij.normalized()
                
                # Viscosity force
                v_ij = v[i] - v[j]
                dot_product = r_ij.dot(v_ij)
                
                if dot_product < 0:
                    mu = 2.0 * alpha * h * cs / (rho[i] + rho[j])
                    k = mu * dot_product / (l * l + 0.01 * h * h)
                    f += n * (m[j] * k * spiky_gradient(l, h))
                    
                    # J matrix update
                    nnt = n.outer_product(n)
                    I_minus_nnt = ti.Matrix.identity(ti.f32, 2) - nnt
                    J += m[j] * k * spiky_hessian(l, h) * (I_minus_nnt / l)
        
        # Position correction
        grad_norm = f.norm()
        if grad_norm > 1e-8:
            grad_dir = f / grad_norm
            sigma = J.norm()
            dl = grad_norm / (1.0 / (dt * dt) + sigma)
            dx[i] = dx[i] + grad_dir * dl

@ti.kernel
def apply_dx():
    for i in range(n_particles):
        x_pred[i] = x_pred[i] + dx[i]

# Update state
@ti.kernel
def update_state(dt: ti.f32):
    for i in range(n_particles):
        v[i] = (x_pred[i] - x[i]) / dt
        x[i] = x_pred[i]

# Update boundary
@ti.kernel
def update_boundary():
    for i in range(n_particles):
        if x[i][0] < bounds[0]: x[i][0] = bounds[0]
        if x[i][0] > bounds[1]: x[i][0] = bounds[1]
        if x[i][1] < bounds[2]: x[i][1] = bounds[2]
        if x[i][1] > bounds[3]: x[i][1] = bounds[3]

# Main simulation step
def step(dt):
    predict(dt)
    find_neighbors()
    update_density()
    update_pressure()
    jacobi_pressure(dt)
    jacobi_viscosity(dt)
    apply_dx()
    update_state(dt)
    update_boundary()

# Get particle data for visualization
def get_particle_data():
    positions = x.to_numpy()
    densities = rho.to_numpy()
    radiuses_np = radiuses.to_numpy()
    return positions, densities, radiuses_np

# Run simulation
def run_simulation(substeps=50):
    # Initialize particles
    init_particles()
    
    # Setup matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame):
        # Run multiple substeps
        dt = 1.0 / 60.0 / substeps
        for _ in range(substeps):
            step(dt)
        
        # Get data
        positions, densities, radiuses_np = get_particle_data()
        
        # Clear and redraw
        ax.clear()
        
        # Calculate colors (based on density)
        colors = np.clip((densities - rho0) / rho0, 0, 1)
        
        # Draw particles
        sizes = (radiuses_np * 600) ** 2
        scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                           c=colors, cmap='plasma', s=sizes)
        
        # Set axes
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect('equal')
        ax.set_title(f'Frame: {frame}')
        
        return ax,
    
    # Create animation
    ani = FuncAnimation(fig, update, interval=60, blit=False)
    plt.show()

if __name__ == "__main__":
    run_simulation()