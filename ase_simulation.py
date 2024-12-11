import numpy as np
from ase import Atoms, units
from ase.io import write
from io import StringIO
from ase.build import bulk, molecule, surface
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.constraints import FixAtoms
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AtomisticSimulation:
    def __init__(self, structure_type='bulk', material='Cu', size=(2,2,2), vacuum=10.0):
        self.structure_type = structure_type
        self.material = material
        self.size = size
        self.vacuum = vacuum
        self.atoms = None
        self.trajectory = []
        self.energies = []
        self.temperatures = []
        
    def create_structure(self):
        """Create atomic structure based on type."""
        if self.structure_type == 'bulk':
            self.atoms = bulk(self.material) * self.size
        elif self.structure_type == 'surface':
            slab = surface(self.material, (1,1,1), self.size[0])
            slab.center(vacuum=self.vacuum, axis=2)
            self.atoms = slab
        elif self.structure_type == 'nanoparticle':
            # Create a simple cubic nanoparticle
            atoms = bulk(self.material, cubic=True) * self.size
            atoms.center(vacuum=self.vacuum)
            self.atoms = atoms
        elif self.structure_type == 'molecule':
            self.atoms = molecule(self.material)
        
        # Set calculator
        self.atoms.calc = EMT()
        
    def optimize_geometry(self, fmax=0.01):
        """Optimize atomic positions."""
        print("Starting geometry optimization")
        opt = BFGS(self.atoms)
        opt.run(fmax=fmax)
        print(f"Final energy: {self.atoms.get_potential_energy()} eV")
        return self.atoms.get_positions()
        
    def run_dynamics(self, temperature=300, timestep=1.0, steps=100):
        """Run molecular dynamics simulation."""
        print(f"Starting MD simulation at {temperature}K")
        
        # Set initial temperature
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature)
        
        # Set up dynamics
        dyn = VelocityVerlet(self.atoms, timestep * units.fs)
        
        # Run dynamics
        self.trajectory = []
        self.energies = []
        self.temperatures = []
        
        for i in range(steps):
            dyn.run(1)
            self.trajectory.append(self.atoms.get_positions().copy())
            self.energies.append(self.atoms.get_potential_energy())
            self.temperatures.append(self.atoms.get_temperature())
            
        print("MD simulation completed")
        return self.trajectory, self.energies, self.temperatures

def create_visualization(trajectory, energies, temperatures, atoms):
    """Create interactive visualization of simulation results."""
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=('Atomic Positions', 'Energy and Temperature'),
                       specs=[[{'type': 'scatter3d'}],
                             [{'secondary_y': True}]])
    
    # Plot final atomic positions
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Add atoms as scatter3d
    fig.add_trace(
        go.Scatter3d(
            x=positions[:,0],
            y=positions[:,1],
            z=positions[:,2],
            mode='markers',
            marker=dict(
                size=10,
                color=[{'Cu': 'orange', 'Au': 'gold', 'Ag': 'silver'}.get(s, 'blue') for s in symbols]
            ),
            text=symbols,
            name='Atoms'
        ),
        row=1, col=1
    )
    
    # Plot energy evolution
    fig.add_trace(
        go.Scatter(
            x=list(range(len(energies))),
            y=energies,
            name='Potential Energy',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    # Plot temperature evolution on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=list(range(len(temperatures))),
            y=temperatures,
            name='Temperature',
            line=dict(color='red')
        ),
        row=2, col=1,
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title='Atomistic Simulation Results',
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)'
        ),
        height=1000
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='Step', row=2, col=1)
    fig.update_yaxes(title_text='Energy (eV)', row=2, col=1)
    fig.update_yaxes(title_text='Temperature (K)', secondary_y=True, row=2, col=1)
    
    return fig

def run_ase_simulation(structure_type='bulk', material='Cu', size=(2,2,2),
                      vacuum=10.0, temperature=300, timestep=1.0, steps=100):
    """Run atomistic simulation using ASE."""
    try:
        print(f"Starting ASE simulation with parameters: structure_type={structure_type}, material={material}")
        
        # Initialize simulation
        sim = AtomisticSimulation(
            structure_type=structure_type,
            material=material,
            size=size,
            vacuum=vacuum
        )
        
        # Create structure
        sim.create_structure()
        print("Structure created")
        
        # Optimize geometry
        sim.optimize_geometry()
        print("Geometry optimized")
        
        # Run molecular dynamics
        trajectory, energies, temperatures = sim.run_dynamics(
            temperature=temperature,
            timestep=timestep,
            steps=steps
        )
        print("Molecular dynamics completed")
        
        # Create visualization
        fig = create_visualization(trajectory, energies, temperatures, sim.atoms)
        print("Visualization created")
        
        return fig
        
    except Exception as e:
        print(f"Error in ASE simulation: {str(e)}")
        # Create error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Simulation failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig 