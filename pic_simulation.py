import numpy as np
import plotly.graph_objects as go
from scipy.constants import epsilon_0, e, m_e

class PICSimulation:
    def __init__(self, nx=100, ny=100, n_particles=1000, dt=1e-12):
        # Grid parameters
        self.nx = nx
        self.ny = ny
        self.dx = 1e-6  # 1 μm grid spacing
        self.dy = 1e-6
        self.dt = dt
        
        # Domain size
        self.Lx = self.nx * self.dx
        self.Ly = self.ny * self.dy
        
        # Initialize grid
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize fields
        self.rho = np.zeros((nx, ny))      # Charge density
        self.phi = np.zeros((nx, ny))      # Electric potential
        self.Ex = np.zeros((nx, ny))       # Electric field x
        self.Ey = np.zeros((nx, ny))       # Electric field y
        
        # Initialize particles
        self.n_particles = n_particles
        self.initialize_particles()
        
    def initialize_particles(self):
        """Initialize particle positions and velocities."""
        # Random positions
        self.particle_x = np.random.uniform(0, self.Lx, self.n_particles)
        self.particle_y = np.random.uniform(0, self.Ly, self.n_particles)
        
        # Gaussian velocity distribution
        vth = 1e5  # Thermal velocity
        self.particle_vx = np.random.normal(0, vth, self.n_particles)
        self.particle_vy = np.random.normal(0, vth, self.n_particles)
        
        # Initialize particle charges and masses
        self.particle_q = -e * np.ones(self.n_particles)  # Electrons
        self.particle_m = m_e * np.ones(self.n_particles)
        
    def deposit_charge(self):
        """Deposit particle charges onto grid using Cloud-in-Cell method."""
        self.rho.fill(0)
        
        # Get particle cell indices
        ix = np.floor(self.particle_x / self.dx).astype(int)
        iy = np.floor(self.particle_y / self.dy).astype(int)
        
        # Get weights for CIC
        wx = (self.particle_x - ix * self.dx) / self.dx
        wy = (self.particle_y - iy * self.dy) / self.dy
        
        # Deposit charge using CIC weights
        for i in range(self.n_particles):
            if 0 <= ix[i] < self.nx-1 and 0 <= iy[i] < self.ny-1:
                self.rho[iy[i], ix[i]] += self.particle_q[i] * (1-wx[i]) * (1-wy[i])
                self.rho[iy[i], ix[i]+1] += self.particle_q[i] * wx[i] * (1-wy[i])
                self.rho[iy[i]+1, ix[i]] += self.particle_q[i] * (1-wx[i]) * wy[i]
                self.rho[iy[i]+1, ix[i]+1] += self.particle_q[i] * wx[i] * wy[i]
        
        self.rho /= (self.dx * self.dy)
        
    def solve_poisson(self):
        """Solve Poisson's equation using FFT."""
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, self.dy)
        Kx, Ky = np.meshgrid(kx, ky)
        K2 = Kx**2 + Ky**2
        K2[0, 0] = 1  # Avoid division by zero
        
        rho_k = np.fft.fft2(self.rho)
        phi_k = -rho_k / (epsilon_0 * K2)
        phi_k[0, 0] = 0  # Set mean potential to zero
        
        self.phi = np.real(np.fft.ifft2(phi_k))
        
        # Calculate E-field
        self.Ex = -np.gradient(self.phi, self.dx, axis=1)
        self.Ey = -np.gradient(self.phi, self.dy, axis=0)
        
    def interpolate_fields(self):
        """Interpolate fields to particle positions using CIC."""
        ix = np.floor(self.particle_x / self.dx).astype(int)
        iy = np.floor(self.particle_y / self.dy).astype(int)
        
        wx = (self.particle_x - ix * self.dx) / self.dx
        wy = (self.particle_y - iy * self.dy) / self.dy
        
        # Initialize particle fields
        particle_Ex = np.zeros(self.n_particles)
        particle_Ey = np.zeros(self.n_particles)
        
        # Interpolate using CIC weights
        for i in range(self.n_particles):
            if 0 <= ix[i] < self.nx-1 and 0 <= iy[i] < self.ny-1:
                particle_Ex[i] = (
                    self.Ex[iy[i], ix[i]] * (1-wx[i]) * (1-wy[i]) +
                    self.Ex[iy[i], ix[i]+1] * wx[i] * (1-wy[i]) +
                    self.Ex[iy[i]+1, ix[i]] * (1-wx[i]) * wy[i] +
                    self.Ex[iy[i]+1, ix[i]+1] * wx[i] * wy[i]
                )
                particle_Ey[i] = (
                    self.Ey[iy[i], ix[i]] * (1-wx[i]) * (1-wy[i]) +
                    self.Ey[iy[i], ix[i]+1] * wx[i] * (1-wy[i]) +
                    self.Ey[iy[i]+1, ix[i]] * (1-wx[i]) * wy[i] +
                    self.Ey[iy[i]+1, ix[i]+1] * wx[i] * wy[i]
                )
        
        return particle_Ex, particle_Ey
    
    def advance_particles(self):
        """Advance particles using leapfrog method."""
        # Get fields at particle positions
        Ex, Ey = self.interpolate_fields()
        
        # Update velocities (half step)
        self.particle_vx += 0.5 * self.particle_q * Ex * self.dt / self.particle_m
        self.particle_vy += 0.5 * self.particle_q * Ey * self.dt / self.particle_m
        
        # Update positions
        self.particle_x += self.particle_vx * self.dt
        self.particle_y += self.particle_vy * self.dt
        
        # Apply periodic boundary conditions
        self.particle_x = self.particle_x % self.Lx
        self.particle_y = self.particle_y % self.Ly
        
        # Update velocities (half step)
        Ex, Ey = self.interpolate_fields()
        self.particle_vx += 0.5 * self.particle_q * Ex * self.dt / self.particle_m
        self.particle_vy += 0.5 * self.particle_q * Ey * self.dt / self.particle_m
    
    def run_simulation(self, n_steps=100):
        """Run simulation for specified number of steps."""
        history = {
            'rho': [],
            'phi': [],
            'particle_positions': []
        }
        
        for _ in range(n_steps):
            self.deposit_charge()
            self.solve_poisson()
            self.advance_particles()
            
            # Store history
            history['rho'].append(self.rho.copy())
            history['phi'].append(self.phi.copy())
            history['particle_positions'].append(
                (self.particle_x.copy(), self.particle_y.copy())
            )
        
        return history

def create_visualization(history, step=0):
    """Create visualization of PIC simulation results."""
    fig = go.Figure()
    
    # Add charge density contour
    fig.add_trace(go.Contour(
        z=history['rho'][step],
        colorscale='RdBu',
        name='Charge Density',
        showscale=True,
        colorbar=dict(title='ρ (C/m²)')
    ))
    
    # Add particle positions
    x, y = history['particle_positions'][step]
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=2,
            color='black',
            opacity=0.5
        ),
        name='Particles'
    ))
    
    # Update layout
    fig.update_layout(
        title='Particle-in-Cell Simulation',
        xaxis_title='X (μm)',
        yaxis_title='Y (μm)',
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def run_pic_simulation(n_particles=1000, n_steps=100, nx=100, ny=100, dt=1e-12):
    """Run PIC simulation with specified parameters."""
    try:
        # Initialize and run simulation
        sim = PICSimulation(
            nx=nx,
            ny=ny,
            n_particles=n_particles,
            dt=dt
        )
        history = sim.run_simulation(n_steps=n_steps)
        
        # Create visualization
        fig = create_visualization(history)
        return fig
        
    except Exception as e:
        print(f"Error in PIC simulation: {str(e)}")
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Simulation failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig 