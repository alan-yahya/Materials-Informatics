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
            
    def calculate_basic_properties(self, selection='all'):
        """Calculate basic structural properties."""
        if self.universe is None:
            return None
            
        try:
            # Select atoms
            atoms = self.universe.select_atoms(selection)
            
            # Calculate basic properties
            self.results['structure'] = {
                'n_atoms': len(atoms),
                'n_residues': len(atoms.residues),
                'residue_types': list(set(atoms.residues.resnames)),
                'atom_types': list(set(atoms.names)),
                'total_mass': atoms.total_mass(),
                'center_of_mass': atoms.center_of_mass(),
                'radius_of_gyration': atoms.radius_of_gyration(),
                'principal_axes': atoms.principal_axes(),
                'dimensions': self.universe.dimensions[:3].tolist()  # Box dimensions
            }
            
            # Calculate bonds if available
            if hasattr(atoms, 'bonds'):
                self.results['structure']['n_bonds'] = len(atoms.bonds)
                
            # Calculate secondary structure if protein
            protein = self.universe.select_atoms('protein')
            if len(protein) > 0:
                try:
                    from MDAnalysis.analysis.secondary_structure import SecondaryStructureAnalysis
                    ssa = SecondaryStructureAnalysis(protein).run()
                    self.results['structure']['secondary_structure'] = {
                        'alpha': float(ssa.alpha_helix_content),
                        'beta': float(ssa.beta_sheet_content),
                        'coil': float(ssa.coil_content)
                    }
                except:
                    print("Could not calculate secondary structure")
            
            return True
            
        except Exception as e:
            print(f"Error calculating basic properties: {str(e)}")
            return False
            
    def create_visualization(self, plot_type='all'):
        """Create visualization of analysis results."""
        if not self.results:
            return None
            
        try:
            if plot_type == 'structure':
                # Create structure analysis visualization
                structure_data = self.results.get('structure', {})
                if not structure_data:
                    raise ValueError("No structure data available")
                
                # Determine if we need the secondary structure plot
                has_ss_data = 'secondary_structure' in structure_data
                
                # Create figure with appropriate subplots
                if has_ss_data:
                    fig = make_subplots(rows=2, cols=2,
                                      subplot_titles=('Atom Distribution', 'Residue Types',
                                                    'Secondary Structure', 'Properties'),
                                      specs=[[{'type': 'xy'}, {'type': 'xy'}],
                                            [{'type': 'domain'}, {'type': 'table'}]])
                else:
                    # No secondary structure data, use simpler layout
                    fig = make_subplots(rows=2, cols=2,
                                      subplot_titles=('Atom Distribution', 'Residue Types',
                                                    'Properties', None),
                                      specs=[[{'type': 'xy'}, {'type': 'xy'}],
                                            [{'type': 'table', 'colspan': 2}, None]])
                
                # Atom types distribution
                atom_types = structure_data.get('atom_types', [])
                if atom_types:
                    atoms = self.universe.select_atoms('all')
                    atom_counts = {atype: len(atoms.select_atoms(f'name {atype}')) 
                                 for atype in atom_types}
                    fig.add_trace(
                        go.Bar(
                            x=list(atom_counts.keys()),
                            y=list(atom_counts.values()),
                            name='Atom Types'
                        ),
                        row=1, col=1
                    )
                    fig.update_xaxes(title_text='Atom Type', row=1, col=1)
                    fig.update_yaxes(title_text='Count', row=1, col=1)
                
                # Residue types distribution
                residue_types = structure_data.get('residue_types', [])
                if residue_types:
                    residues = self.universe.select_atoms('all').residues
                    res_counts = {rtype: len([r for r in residues if r.resname == rtype]) 
                                for rtype in residue_types}
                    fig.add_trace(
                        go.Bar(
                            x=list(res_counts.keys()),
                            y=list(res_counts.values()),
                            name='Residue Types'
                        ),
                        row=1, col=2
                    )
                    fig.update_xaxes(title_text='Residue Type', row=1, col=2)
                    fig.update_yaxes(title_text='Count', row=1, col=2)
                
                # Secondary structure pie chart if available
                if has_ss_data:
                    ss_data = structure_data['secondary_structure']
                    fig.add_trace(
                        go.Pie(
                            labels=['α-Helix', 'β-Sheet', 'Coil'],
                            values=[ss_data.get('alpha', 0),
                                   ss_data.get('beta', 0),
                                   ss_data.get('coil', 0)],
                            name='Secondary Structure'
                        ),
                        row=2, col=1
                    )
                
                # Properties table
                properties = {
                    'Number of Atoms': structure_data.get('n_atoms', 'N/A'),
                    'Number of Residues': structure_data.get('n_residues', 'N/A'),
                    'Total Mass (Da)': f"{structure_data.get('total_mass', 0):.2f}",
                    'Radius of Gyration (Å)': f"{structure_data.get('radius_of_gyration', 0):.2f}"
                }
                
                # Add box dimensions if available
                dimensions = structure_data.get('dimensions')
                if dimensions and len(dimensions) >= 3:
                    properties['Box Dimensions (Å)'] = ' × '.join([f"{x:.1f}" for x in dimensions[:3]])
                
                # Add number of bonds if available
                if 'n_bonds' in structure_data:
                    properties['Number of Bonds'] = structure_data['n_bonds']
                
                table_trace = go.Table(
                    header=dict(
                        values=['Property', 'Value'],
                        fill_color='paleturquoise',
                        align='left'
                    ),
                    cells=dict(
                        values=[list(properties.keys()),
                               list(properties.values())],
                        fill_color='lavender',
                        align='left'
                    )
                )
                
                if has_ss_data:
                    fig.add_trace(table_trace, row=2, col=2)
                else:
                    fig.add_trace(table_trace, row=2, col=1)
                
                # Update layout
                fig.update_layout(
                    title='Structure Analysis Results',
                    height=800,
                    showlegend=True
                )
                
                return fig
                
            else:
                # Handle other plot types using existing visualization code
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
                
                # Update layout
                fig.update_layout(
                    title='Analysis Results',
                    height=300 * n_plots,
                    showlegend=True
                )
                
                return fig
                
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None

def run_mdanalysis(topology_data=None, trajectory_data=None, analysis_type='structure', 
                  selection='all', topology_format='pdb', trajectory_format='none', **kwargs):
    """Run molecular dynamics trajectory analysis."""
    try:
        print("Starting MD trajectory analysis")
        
        # Initialize analyzer
        analyzer = TrajectoryAnalyzer()
        
        # Load trajectory
        if not analyzer.load_trajectory(topology_data, trajectory_data, format=topology_format):
            raise ValueError("Failed to load trajectory")
        print("Trajectory loaded")
        
        # Run analyses based on type
        if analysis_type == 'structure':
            # Basic structure analysis
            if analyzer.calculate_basic_properties(selection):
                print("Structure analysis completed")
                
        elif analysis_type == 'contacts':
            if analyzer.analyze_contacts(
                selection1=selection,
                selection2=kwargs.get('contact_selection', 'all'),
                cutoff=kwargs.get('cutoff', 3.5)
            ):
                print("Contact analysis completed")
                
        elif analysis_type == 'rmsd':
            if analyzer.calculate_rmsd(
                select=selection
            ):
                print("RMSD analysis completed")
                
        elif analysis_type == 'rdf':
            if analyzer.calculate_rdf(
                selection1=selection,
                selection2=kwargs.get('rdf_selection', 'all'),
                nbins=kwargs.get('nbins', 100),
                range=kwargs.get('range', (0, 15))
            ):
                print("RDF analysis completed")
                
        elif analysis_type == 'density':
            if analyzer.calculate_density(
                selection=selection,
                delta=kwargs.get('delta', 1.0)
            ):
                print("Density analysis completed")
        
        # Create visualization
        fig = analyzer.create_visualization(plot_type=analysis_type)
        if fig is None:
            raise ValueError("Failed to create visualization")
            
        return fig
        
    except Exception as e:
        print(f"Error in MD analysis: {str(e)}")
        raise 