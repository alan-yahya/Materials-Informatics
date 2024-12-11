import numpy as np
import qutip as qt
import plotly.graph_objects as go

class QuantumDynamics:
    def __init__(self, n_levels=3, gamma=0.1, omega=1.0):
        print(f"Initializing QuantumDynamics with n_levels={n_levels}, gamma={gamma}, omega={omega}")  # Debug log
        self.n_levels = n_levels
        self.gamma = gamma  # Decay rate
        self.omega = omega  # Rabi frequency
        
    def create_hamiltonian(self):
        print("Creating Hamiltonian")  # Debug log
        # Create annihilation operator
        a = qt.destroy(self.n_levels)
        # Create Hamiltonian
        H = self.omega * (a + a.dag())
        return H
        
    def create_collapse_operators(self):
        print("Creating collapse operators")  # Debug log
        # Create annihilation operator
        a = qt.destroy(self.n_levels)
        # Create collapse operators for decay
        c_ops = [np.sqrt(self.gamma) * a]
        return c_ops
        
    def initial_state(self, excited_level=1):
        print(f"Creating initial state with excited_level={excited_level}")  # Debug log
        # Start in a specific excited state
        psi0 = qt.basis(self.n_levels, excited_level)
        return psi0
        
    def run_simulation(self, t_max=10.0, n_steps=200):
        print(f"Running simulation with t_max={t_max}, n_steps={n_steps}")  # Debug log
        # Time points
        times = np.linspace(0, t_max, n_steps)
        
        # Create operators and initial state
        H = self.create_hamiltonian()
        c_ops = self.create_collapse_operators()
        psi0 = self.initial_state()
        
        print("Starting quantum evolution")  # Debug log
        # Run quantum dynamics
        result = qt.sesolve(H, psi0, times, c_ops)
        
        print("Calculating expectation values")  # Debug log
        # Calculate expectation values
        expect_n = []
        n_op = qt.num(self.n_levels)
        
        for state in result.states:
            expect_n.append(qt.expect(n_op, state))
            
        return times, expect_n, result.states

def create_visualization(times, expect_n, states, n_levels):
    print("Creating visualization")  # Debug log
    # Create main figure
    fig = go.Figure()
    
    # Plot expectation value of number operator
    fig.add_trace(go.Scatter(
        x=times,
        y=expect_n,
        mode='lines',
        name='Excitation Number'
    ))
    
    # Plot population of each level
    for i in range(n_levels):
        populations = [abs(state[i])**2 for state in states]
        fig.add_trace(go.Scatter(
            x=times,
            y=populations,
            mode='lines',
            name=f'Level {i}'
        ))
    
    # Update layout
    fig.update_layout(
        title='Quantum State Evolution',
        xaxis_title='Time',
        yaxis_title='Population/Expectation',
        showlegend=True,
        width=800,
        height=600
    )
    
    print("Visualization created")  # Debug log
    return fig

def run_qutip_simulation(n_levels=3, gamma=0.1, omega=1.0, t_max=10.0, n_steps=200):
    """Run quantum dynamics simulation using QuTiP."""
    try:
        print(f"Starting QuTiP simulation with parameters: n_levels={n_levels}, gamma={gamma}, omega={omega}, t_max={t_max}, n_steps={n_steps}")  # Debug log
        
        # Initialize simulator
        simulator = QuantumDynamics(n_levels=n_levels, gamma=gamma, omega=omega)
        
        # Run simulation
        times, expect_n, states = simulator.run_simulation(t_max=t_max, n_steps=n_steps)
        
        # Create visualization
        fig = create_visualization(times, expect_n, states, n_levels)
        
        print("QuTiP simulation completed successfully")  # Debug log
        return fig
        
    except Exception as e:
        print(f"Error in QuTiP simulation: {str(e)}")  # Debug log
        # Create error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Simulation failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig 