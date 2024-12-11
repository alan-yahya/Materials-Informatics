import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import MDAnalysis as mda
from MDAnalysis.analysis import rdf, contacts, rms, align
import tempfile
import os

class TrajectoryAnalyzer:
    def __init__(self):
        self.universe = None
        self.temp_dir = tempfile.mkdtemp()
        self.results = {}
        
    def load_trajectory(self, topology_data=None, trajectory_data=None, format='PDB'):
        """Load molecular dynamics trajectory."""
        try:
            if format == 'PDB':
                # Create temporary files
                top_file = os.path.join(self.temp_dir, 'topology.pdb')
                with open(top_file, 'w') as f:
                    f.write(topology_data)
                    
                if trajectory_data:
                    traj_file = os.path.join(self.temp_dir, 'trajectory.dcd')
                    with open(traj_file, 'w') as f:
                        f.write(trajectory_data)
                    self.universe = mda.Universe(top_file, traj_file)
                else:
                    self.universe = mda.Universe(top_file)
                    
            return True
        except Exception as e:
            print(f"Error loading trajectory: {str(e)}")
            return False
            
    def calculate_rdf(self, selection1='name NA', selection2='name CL', nbins=100, range=(0, 15)):
        """Calculate radial distribution function."""
        if self.universe is None:
            return None
            
        try:
            # Select atoms
            group1 = self.universe.select_atoms(selection1)
            group2 = self.universe.select_atoms(selection2)
            
            # Calculate RDF
            rdf_analyzer = rdf.InterRDF(group1, group2, nbins=nbins, range=range)
            rdf_analyzer.run()
            
            self.results['rdf'] = {
                'bins': rdf_analyzer.bins,
                'rdf': rdf_analyzer.rdf
            }
            return True
        except Exception as e:
            print(f"Error calculating RDF: {str(e)}")
            return False
            
    def calculate_rmsd(self, reference=None, select='protein'):
        """Calculate RMSD over trajectory."""
        if self.universe is None:
            return None
            
        try:
            # Select atoms
            mobile = self.universe.select_atoms(select)
            
            if reference is None:
                reference = mobile.positions.copy()
                
            # Calculate RMSD
            rmsd_analyzer = rms.RMSD(mobile, reference=reference)
            rmsd_analyzer.run()
            
            self.results['rmsd'] = {
                'time': np.arange(len(rmsd_analyzer.rmsd)),
                'rmsd': rmsd_analyzer.rmsd[:, 2]  # Column 2 contains RMSD values
            }
            return True
        except Exception as e:
            print(f"Error calculating RMSD: {str(e)}")
            return False
            
    def analyze_contacts(self, selection1='protein', selection2='resname LIG', cutoff=3.5):
        """Analyze contacts between selections."""
        if self.universe is None:
            return None
            
        try:
            # Select atoms
            group1 = self.universe.select_atoms(selection1)
            group2 = self.universe.select_atoms(selection2)
            
            # Calculate contacts
            contacts_analyzer = contacts.Contacts(group1, group2, cutoff=cutoff)
            contacts_analyzer.run()
            
            self.results['contacts'] = {
                'time': np.arange(len(contacts_analyzer.timeseries)),
                'n_contacts': contacts_analyzer.timeseries[:, 1]
            }
            return True
        except Exception as e:
            print(f"Error analyzing contacts: {str(e)}")
            return False
            
    def calculate_radius_of_gyration(self, selection='protein'):
        """Calculate radius of gyration over trajectory."""
        if self.universe is None:
            return None
            
        try:
            # Select atoms
            group = self.universe.select_atoms(selection)
            
            # Calculate Rg over trajectory
            rg = []
            times = []
            
            for ts in self.universe.trajectory:
                rg.append(group.radius_of_gyration())
                times.append(ts.time)
                
            self.results['rg'] = {
                'time': np.array(times),
                'rg': np.array(rg)
            }
            return True
        except Exception as e:
            print(f"Error calculating radius of gyration: {str(e)}")
            return False
            
    def create_visualization(self, plot_type='all'):
        """Create visualization of analysis results."""
        if not self.results:
            return None
            
        try:
            n_plots = len(self.results)
            fig = make_subplots(rows=n_plots, cols=1,
                              subplot_titles=list(self.results.keys()),
                              vertical_spacing=0.1)
            
            current_row = 1
            
            # Plot RDF if available
            if 'rdf' in self.results:
                fig.add_trace(
                    go.Scatter(
                        x=self.results['rdf']['bins'],
                        y=self.results['rdf']['rdf'],
                        mode='lines',
                        name='RDF'
                    ),
                    row=current_row, col=1
                )
                fig.update_xaxes(title_text='Distance (Å)', row=current_row)
                fig.update_yaxes(title_text='g(r)', row=current_row)
                current_row += 1
                
            # Plot RMSD if available
            if 'rmsd' in self.results:
                fig.add_trace(
                    go.Scatter(
                        x=self.results['rmsd']['time'],
                        y=self.results['rmsd']['rmsd'],
                        mode='lines',
                        name='RMSD'
                    ),
                    row=current_row, col=1
                )
                fig.update_xaxes(title_text='Time', row=current_row)
                fig.update_yaxes(title_text='RMSD (Å)', row=current_row)
                current_row += 1
                
            # Plot contacts if available
            if 'contacts' in self.results:
                fig.add_trace(
                    go.Scatter(
                        x=self.results['contacts']['time'],
                        y=self.results['contacts']['n_contacts'],
                        mode='lines',
                        name='Contacts'
                    ),
                    row=current_row, col=1
                )
                fig.update_xaxes(title_text='Time', row=current_row)
                fig.update_yaxes(title_text='Number of Contacts', row=current_row)
                current_row += 1
                
            # Plot radius of gyration if available
            if 'rg' in self.results:
                fig.add_trace(
                    go.Scatter(
                        x=self.results['rg']['time'],
                        y=self.results['rg']['rg'],
                        mode='lines',
                        name='Rg'
                    ),
                    row=current_row, col=1
                )
                fig.update_xaxes(title_text='Time', row=current_row)
                fig.update_yaxes(title_text='Radius of Gyration (Å)', row=current_row)
                
            # Update layout
            fig.update_layout(
                title='Trajectory Analysis Results',
                height=300 * n_plots,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None

def run_mdanalysis(topology_data=None, trajectory_data=None, analysis_type='all', **kwargs):
    """Run molecular dynamics trajectory analysis."""
    try:
        print("Starting MD trajectory analysis")
        
        # Initialize analyzer
        analyzer = TrajectoryAnalyzer()
        
        # Load trajectory
        if not analyzer.load_trajectory(topology_data, trajectory_data):
            raise ValueError("Failed to load trajectory")
        print("Trajectory loaded")
        
        # Run analyses based on type
        if analysis_type in ['all', 'rdf']:
            if analyzer.calculate_rdf(
                selection1=kwargs.get('selection1', 'name NA'),
                selection2=kwargs.get('selection2', 'name CL'),
                nbins=kwargs.get('nbins', 100),
                range=kwargs.get('range', (0, 15))
            ):
                print("RDF calculated")
                
        if analysis_type in ['all', 'rmsd']:
            if analyzer.calculate_rmsd(
                select=kwargs.get('rmsd_selection', 'protein')
            ):
                print("RMSD calculated")
                
        if analysis_type in ['all', 'contacts']:
            if analyzer.analyze_contacts(
                selection1=kwargs.get('contact_sel1', 'protein'),
                selection2=kwargs.get('contact_sel2', 'resname LIG'),
                cutoff=kwargs.get('cutoff', 3.5)
            ):
                print("Contacts analyzed")
                
        if analysis_type in ['all', 'rg']:
            if analyzer.calculate_radius_of_gyration(
                selection=kwargs.get('rg_selection', 'protein')
            ):
                print("Radius of gyration calculated")
                
        # Create visualization
        fig = analyzer.create_visualization(analysis_type)
        print("Visualization created")
        
        return fig
        
    except Exception as e:
        print(f"Error in MD analysis: {str(e)}")
        # Create error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Analysis failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig 