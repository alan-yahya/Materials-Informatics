import numpy as np
from scipy.integrate import odeint, solve_ivp
import plotly.graph_objects as go
from scipy.constants import hbar, m_e

class SemiClassicalTrajectory:
    def __init__(self, mass=m_e, energy=1.0, potential_type='barrier'):
        self.mass = mass
        self.energy = energy
        self.potential_type = potential_type
        
        # Initialize common parameters
        self.omega = 1.0  # Angular frequency for harmonic oscillator
        self.V0 = 1.5 * self.energy  # Default barrier/well height/depth
        self.a = 1.0  # Default width
        
        self.setup_potential()
        
    def setup_potential(self):
        """Define potential functions for different cases."""
        if self.potential_type == 'barrier':
            self.V0 = 1.5 * self.energy
            self.potential = lambda x: self.V0 if abs(x) < self.a/2 else 0.0
            self.turning_points = [-self.a/2, self.a/2]
            
        elif self.potential_type == 'well':
            self.V0 = -2.0 * self.energy
            self.potential = lambda x: self.V0 if abs(x) < self.a/2 else 0.0
            # Calculate classical turning points for well
            if self.energy < 0:
                self.turning_points = [-self.a/2, self.a/2]
            else:
                self.turning_points = []
                
        else:  # Harmonic oscillator
            self.potential = lambda x: 0.5 * self.mass * self.omega**2 * x**2
            # Calculate classical turning points for harmonic oscillator
            if self.energy > 0:
                amplitude = np.sqrt(2*self.energy/(self.mass*self.omega**2))
                self.turning_points = [-amplitude, amplitude]
            else:
                self.turning_points = []
    
    def wkb_wavefunction(self, x):
        """Calculate WKB approximation of the wavefunction."""
        V = np.vectorize(self.potential)(x)
        k = np.sqrt(2 * self.mass * (self.energy - V + 0j)) / hbar
        
        # WKB amplitude and phase
        amplitude = 1 / np.sqrt(abs(k))
        phase = np.zeros_like(x, dtype=complex)
        
        # Integrate k to get phase
        for i in range(1, len(x)):
            phase[i] = phase[i-1] + k[i-1] * (x[i] - x[i-1])
        
        # Apply connection formulas near turning points
        if self.turning_points:
            for tp in self.turning_points:
                mask = x > tp
                phase[mask] += np.pi/4
        
        return amplitude * np.exp(1j * phase)
    
    def bohm_velocity_field(self, x, psi):
        """Calculate Bohmian velocity field from wavefunction."""
        dx = x[1] - x[0]
        j = hbar/(2j*self.mass) * (np.gradient(psi, dx) * np.conj(psi) - 
                                  psi * np.gradient(np.conj(psi), dx))
        rho = abs(psi)**2
        # Avoid division by zero
        rho = np.where(rho < 1e-10, 1e-10, rho)
        v = j/rho
        return np.real(v)
    
    def bohm_trajectory(self, x0, t_span, x_grid):
        """Calculate Bohmian trajectory."""
        psi = self.wkb_wavefunction(x_grid)
        v_field = self.bohm_velocity_field(x_grid, psi)
        
        def v_interp(t, x):
            idx = np.searchsorted(x_grid, x)
            if idx == 0:
                return v_field[0]
            elif idx == len(x_grid):
                return v_field[-1]
            else:
                # Linear interpolation
                alpha = (x - x_grid[idx-1])/(x_grid[idx] - x_grid[idx-1])
                return (1-alpha)*v_field[idx-1] + alpha*v_field[idx]
        
        try:
            sol = solve_ivp(v_interp, t_span, [x0], dense_output=True)
            return sol.t, sol.y[0]
        except:
            # Return constant position if integration fails
            t = np.linspace(t_span[0], t_span[1], 100)
            return t, np.full_like(t, x0)
    
    def calculate_trajectories(self, n_trajectories=5):
        """Calculate both WKB and Bohmian trajectories."""
        # Set up spatial and temporal grids
        x = np.linspace(-5, 5, 1000)
        t = np.linspace(0, 10, 200)
        
        try:
            # Calculate WKB wavefunction
            psi_wkb = self.wkb_wavefunction(x)
            
            # Calculate Bohmian trajectories
            bohm_trajectories = []
            x0_values = np.linspace(-4, 4, n_trajectories)
            for x0 in x0_values:
                t_traj, x_traj = self.bohm_trajectory(x0, (t[0], t[-1]), x)
                bohm_trajectories.append((t_traj, x_traj))
            
            return {
                'x': x,
                't': t,
                'psi_wkb': psi_wkb,
                'bohm_trajectories': bohm_trajectories,
                'potential': np.vectorize(self.potential)(x)
            }
        except Exception as e:
            print(f"Error in calculate_trajectories: {str(e)}")
            # Return empty results
            return {
                'x': x,
                't': t,
                'psi_wkb': np.zeros_like(x, dtype=complex),
                'bohm_trajectories': [],
                'potential': np.vectorize(self.potential)(x)
            }

def create_visualization(results, title="Semi-Classical Trajectories"):
    """Create visualization of WKB and Bohmian trajectories."""
    fig = go.Figure()
    
    # Plot potential
    fig.add_trace(go.Scatter(
        x=results['x'],
        y=results['potential'],
        name='Potential',
        line=dict(color='black', dash='dash'),
    ))
    
    # Plot WKB probability density
    psi_wkb = results['psi_wkb']
    prob_density = np.abs(psi_wkb)**2
    if np.any(prob_density > 0):  # Only plot if we have non-zero values
        fig.add_trace(go.Scatter(
            x=results['x'],
            y=prob_density/np.max(prob_density),  # Normalize for visualization
            name='WKB Probability',
            line=dict(color='blue'),
        ))
    
    # Plot Bohmian trajectories
    for i, (t, x) in enumerate(results['bohm_trajectories']):
        fig.add_trace(go.Scatter(
            x=x,
            y=np.zeros_like(x) + i*0.1,  # Offset for visibility
            name=f'Trajectory {i+1}',
            line=dict(color='red'),
            showlegend=(i == 0)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Position',
        yaxis_title='Energy / Probability',
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def run_semiclassical_simulation(mass=m_e, energy=1.0, potential_type='barrier'):
    """Run semi-classical trajectory simulation."""
    try:
        model = SemiClassicalTrajectory(mass=mass, energy=energy, potential_type=potential_type)
        results = model.calculate_trajectories()
        fig = create_visualization(results)
        return fig
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Simulation failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig 