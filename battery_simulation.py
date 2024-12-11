import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint

class BatteryModel:
    def __init__(self, initial_capacity=900, cycles=100):
        # Material parameters for S/PCNF/CNT
        self.initial_capacity = initial_capacity  # mAh/g
        self.decay_rate = 0.002  # Capacity fade per cycle
        self.cnt_content = 0.15  # CNT content ratio
        self.sulfur_loading = 0.65  # Sulfur loading ratio
        self.cycles = cycles
        
        # Electrochemical parameters
        self.voltage_range = (1.7, 2.8)  # V
        self.c_rate = 0.2  # C-rate
        self.temperature = 298  # K
        
    def capacity_fade_model(self, cycle):
        """Model capacity fade over cycles."""
        # Basic capacity fade model with CNT stabilization effect
        base_fade = self.initial_capacity * np.exp(-self.decay_rate * cycle)
        cnt_stabilization = 1 + 0.1 * self.cnt_content * (1 - np.exp(-cycle/20))
        return base_fade * cnt_stabilization
    
    def voltage_profile(self, state_of_charge):
        """Calculate voltage profile based on state of charge."""
        # Simplified voltage profile for Li-S battery
        v_high = self.voltage_range[1]
        v_low = self.voltage_range[0]
        
        # Two-plateau behavior characteristic of Li-S batteries
        plateau1 = 2.3 + 0.1 * np.sin(state_of_charge * np.pi)
        plateau2 = 2.1 + 0.05 * np.sin(state_of_charge * np.pi)
        
        # Combine plateaus based on state of charge
        voltage = np.where(state_of_charge > 0.5, plateau1, plateau2)
        return np.clip(voltage, v_low, v_high)
    
    def simulate_cycle(self, cycle_number):
        """Simulate a single charge-discharge cycle."""
        # Time points for a single cycle
        t = np.linspace(0, 1, 100)
        
        # Discharge curve (state of charge)
        soc_discharge = 1 - t
        # Charge curve (state of charge)
        soc_charge = t
        
        # Calculate capacity for this cycle
        current_capacity = self.capacity_fade_model(cycle_number)
        
        # Calculate voltage profiles
        voltage_discharge = self.voltage_profile(soc_discharge)
        voltage_charge = self.voltage_profile(soc_charge)
        
        return {
            'soc_discharge': soc_discharge,
            'soc_charge': soc_charge,
            'voltage_discharge': voltage_discharge,
            'voltage_charge': voltage_charge,
            'capacity': current_capacity
        }
    
    def run_simulation(self):
        """Run complete battery simulation."""
        cycles = np.arange(self.cycles)
        capacities = np.array([self.capacity_fade_model(c) for c in cycles])
        
        # Simulate specific cycles for voltage profiles
        cycle_data = self.simulate_cycle(0)  # First cycle
        mid_cycle_data = self.simulate_cycle(self.cycles // 2)  # Middle cycle
        final_cycle_data = self.simulate_cycle(self.cycles - 1)  # Last cycle
        
        return {
            'cycles': cycles,
            'capacities': capacities,
            'first_cycle': cycle_data,
            'mid_cycle': mid_cycle_data,
            'final_cycle': final_cycle_data
        }

def create_visualization(simulation_data):
    """Create visualization of battery performance."""
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add capacity fade curve
    fig.add_trace(
        go.Scatter(
            x=simulation_data['cycles'],
            y=simulation_data['capacities'],
            name='Capacity Fade',
            line=dict(color='blue')
        )
    )
    
    # Add voltage profiles for first, middle, and last cycles
    cycles_to_plot = {
        'First Cycle': simulation_data['first_cycle'],
        'Mid Cycle': simulation_data['mid_cycle'],
        'Final Cycle': simulation_data['final_cycle']
    }
    
    colors = ['red', 'green', 'orange']
    for (cycle_name, data), color in zip(cycles_to_plot.items(), colors):
        # Discharge curve
        fig.add_trace(
            go.Scatter(
                x=data['capacity'] * data['soc_discharge'],
                y=data['voltage_discharge'],
                name=f'{cycle_name} Discharge',
                line=dict(color=color, dash='solid'),
                visible='legendonly'
            )
        )
        
        # Charge curve
        fig.add_trace(
            go.Scatter(
                x=data['capacity'] * data['soc_charge'],
                y=data['voltage_charge'],
                name=f'{cycle_name} Charge',
                line=dict(color=color, dash='dot'),
                visible='legendonly'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Battery Performance: S/PCNF/CNT Composite',
        xaxis_title='Capacity (mAh/g)',
        yaxis_title='Voltage (V) / Capacity (mAh/g)',
        hovermode='x',
        showlegend=True
    )
    
    return fig

def run_battery_simulation(initial_capacity=900, cycles=100):
    """Run battery simulation and create visualization."""
    # Create and run simulation
    model = BatteryModel(initial_capacity=initial_capacity, cycles=cycles)
    simulation_data = model.run_simulation()
    
    # Create visualization
    fig = create_visualization(simulation_data)
    
    return fig 