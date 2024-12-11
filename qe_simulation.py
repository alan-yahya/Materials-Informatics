import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ase import Atoms
from ase.io import write
from ase.calculators.espresso import Espresso
import os
import subprocess
import platform

def check_qe_installation():
    """Check if Quantum ESPRESSO is installed and accessible."""
    try:
        if platform.system() == 'Windows':
            result = subprocess.run(['where', 'pw.x'], capture_output=True, text=True)
        else:
            result = subprocess.run(['which', 'pw.x'], capture_output=True, text=True)
        return bool(result.stdout.strip())
    except:
        return False

class QESimulator:
    def __init__(self):
        self.structure = None
        self.calculator = None
        self.results = None
        
    def create_structure(self, structure_type='bulk', **kwargs):
        """Create atomic structure based on type."""
        if structure_type == 'bulk':
            # Create bulk crystal structure (e.g., FCC Cu)
            lattice_constant = kwargs.get('lattice_constant', 3.5)
            self.structure = Atoms('Cu',
                                 positions=[[0, 0, 0]],
                                 cell=[[lattice_constant, 0, 0],
                                       [0, lattice_constant, 0],
                                       [0, 0, lattice_constant]],
                                 pbc=True)
        return self.structure
        
    def setup_calculator(self, calculation_type='scf', **kwargs):
        """Setup QE calculator with given parameters."""
        if not check_qe_installation():
            raise RuntimeError(
                "Quantum ESPRESSO (pw.x) is not installed or not in PATH.\n"
                "Please install Quantum ESPRESSO:\n"
                "- Windows: Download from http://www.qe-forge.org/gf/project/q-e/frs/\n"
                "- Linux: Use package manager (apt-get install quantum-espresso)\n"
                "- macOS: Use Homebrew (brew install quantum-espresso)\n"
                "After installation, make sure pw.x is in your system PATH."
            )
            
        # QE input parameters
        input_data = {
            'control': {
                'calculation': calculation_type,
                'restart_mode': 'from_scratch',
                'prefix': 'qe',
                'pseudo_dir': '.',
                'outdir': '.',
                'verbosity': 'high'
            },
            'system': {
                'ecutwfc': kwargs.get('ecutwfc', 40.0),
                'ecutrho': kwargs.get('ecutrho', 320.0),
                'occupations': 'smearing',
                'smearing': 'gaussian',
                'degauss': 0.01
            },
            'electrons': {
                'conv_thr': 1.0e-6,
                'mixing_beta': 0.7
            }
        }
        
        # K-points
        kpts = kwargs.get('kpts', (4, 4, 4))
        
        self.calculator = Espresso(
            input_data=input_data,
            pseudopotentials={'Cu': 'Cu.pbe-dn-kjpaw_psl.1.0.0.UPF'},
            kpts=kpts
        )
        
        self.structure.calc = self.calculator
        
    def run_calculation(self):
        """Run the calculation."""
        try:
            # Calculate energy
            self.results = {
                'energy': self.structure.get_potential_energy(),
                'forces': self.structure.get_forces(),
                'stress': self.structure.get_stress()
            }
            return True
        except Exception as e:
            print(f"Error in calculation: {str(e)}")
            # Create mock results for visualization when QE is not available
            self.results = {
                'energy': 0.0,
                'forces': np.zeros((len(self.structure), 3)),
                'stress': np.zeros(6),
                'is_mock': True
            }
            return False
            
    def create_visualization(self, plot_type='scf'):
        """Create visualization of calculation results."""
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Structure', 'Analysis'),
                           specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]])
        
        # Plot structure
        positions = self.structure.get_positions()
        
        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='circle'
                ),
                name='Atoms'
            ),
            row=1, col=1
        )
        
        # Plot analysis
        if self.results:
            if 'is_mock' in self.results:
                # Show message about QE not being available
                fig.add_annotation(
                    text=(
                        "Quantum ESPRESSO not installed.\n"
                        "This is a visualization mock-up.\n"
                        "Please install QE for actual calculations."
                    ),
                    xref="x2", yref="y2",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
            else:
                # Plot real results
                fig.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[self.results['energy']],
                        mode='markers',
                        name='Total Energy'
                    ),
                    row=1, col=2
                )
        
        # Update layout
        fig.update_layout(
            title='Quantum ESPRESSO Calculation',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)'
            ),
            height=600,
            showlegend=True
        )
        
        return fig

def run_qe_simulation(structure_type='bulk', calculation_type='scf', plot_type='scf', **kwargs):
    """Run QE simulation."""
    try:
        print(f"Starting QE simulation with structure type: {structure_type}, calculation: {calculation_type}")
        
        # Initialize simulator
        simulator = QESimulator()
        
        # Create structure
        simulator.create_structure(structure_type, **kwargs)
        print("Structure created")
        
        try:
            # Setup calculator and run
            simulator.setup_calculator(calculation_type, **kwargs)
            simulator.run_calculation()
        except RuntimeError as e:
            print(f"QE setup error: {str(e)}")
        
        # Create visualization
        fig = simulator.create_visualization(plot_type)
        print("QE simulation completed successfully")
        
        return fig
        
    except Exception as e:
        print(f"Error in QE simulation: {str(e)}")
        # Create error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Simulation failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig 