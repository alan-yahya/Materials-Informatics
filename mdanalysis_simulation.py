import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import rms, align, contacts
# Try to import secondary structure analysis, but don't fail if not available
try:
    from MDAnalysis.analysis.secondary_structure import SecondaryStructureAnalysis
    HAS_SECONDARY_STRUCTURE = True
except ImportError:
    HAS_SECONDARY_STRUCTURE = False
    print("Warning: Secondary structure analysis not available. Install MDAnalysis with full dependencies for this feature.")
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run_basic_analysis(pdb_file, options=None):
    """
    Perform structural analysis on a PDB file with customizable options.
    
    Parameters
    ----------
    pdb_file : str
        Path to the PDB file to analyze
    options : dict
        Analysis options including:
        - analyze_secondary: bool
        - analyze_contacts: bool
        - analyze_rmsd: bool
        - contact_cutoff: float
        - show_backbone: bool
        - show_sidechains: bool
        - show_hydrogens: bool
        - color_scheme: str
        
    Returns
    -------
    dict
        Results containing analysis data and visualizations
    """
    if options is None:
        options = {}
        
    # Set default options
    default_options = {
        'analyze_secondary': True,
        'analyze_contacts': True,
        'analyze_rmsd': True,
        'contact_cutoff': 6.0,
        'show_backbone': True,
        'show_sidechains': False,
        'show_hydrogens': False,
        'color_scheme': 'element'
    }
    
    options = {**default_options, **options}
    
    try:
        # Create a Universe from the PDB file
        universe = mda.Universe(pdb_file)
        
        results = {
            'basic_info': get_basic_info(universe),
            'plots': [],
            'warnings': [],
            'errors': []
        }
        
        # Select atoms based on visualization options
        selection = build_atom_selection(universe, options)
        
        # Perform requested analyses
        if options['analyze_secondary']:
            try:
                ss_results = analyze_secondary_structure(universe)
                results['secondary_structure'] = ss_results
                results['plots'].append(create_ss_plot(ss_results))
            except Exception as e:
                results['errors'].append(f"Secondary structure analysis failed: {str(e)}")
        
        if options['analyze_contacts']:
            try:
                contact_results = analyze_contacts(universe, cutoff=options['contact_cutoff'])
                results['contacts'] = contact_results
                results['plots'].append(create_contact_plot(contact_results))
            except Exception as e:
                results['errors'].append(f"Contact analysis failed: {str(e)}")
        
        if options['analyze_rmsd']:
            try:
                rmsd_results = calculate_rmsd(universe)
                results['rmsd'] = rmsd_results
                results['plots'].append(create_rmsd_plot(rmsd_results))
            except Exception as e:
                results['errors'].append(f"RMSD calculation failed: {str(e)}")
        
        # Create structure visualization
        try:
            structure_plot = create_structure_visualization(
                selection, 
                color_scheme=options['color_scheme']
            )
            results['plots'].append(structure_plot)
        except Exception as e:
            results['errors'].append(f"Structure visualization failed: {str(e)}")
        
        return results
        
    except Exception as e:
        return {
            'error': str(e),
            'plots': [],
            'warnings': [],
            'errors': [str(e)]
        }

def get_basic_info(universe):
    """Get basic system information."""
    # Convert numpy types to native Python types
    atom_types = dict(zip(*np.unique(universe.atoms.names, return_counts=True)))
    atom_types = {str(k): int(v) for k, v in atom_types.items()}
    
    # Get residue information
    residue_types = dict(zip(*np.unique(universe.residues.resnames, return_counts=True)))
    residue_types = {str(k): int(v) for k, v in residue_types.items()}
    
    # Calculate chain information
    chain_info = {}
    for segid in np.unique(universe.segments.segids):
        chain = universe.select_atoms(f'segid {segid}')
        chain_info[str(segid)] = {
            'n_residues': int(len(chain.residues)),
            'n_atoms': int(len(chain)),
            'residue_range': f"{chain.residues.resids[0]}-{chain.residues.resids[-1]}"
        }
    
    # Calculate structure dimensions
    bbox = universe.atoms.bbox()
    dimensions = {
        'x': float(bbox[1][0] - bbox[0][0]),
        'y': float(bbox[1][1] - bbox[0][1]),
        'z': float(bbox[1][2] - bbox[0][2])
    }
    
    # Try to get charge information, default to 0 if not available
    try:
        total_charge = float(sum(universe.atoms.charges))
    except (AttributeError, ValueError):
        total_charge = 0.0
    
    return {
        'n_atoms': int(len(universe.atoms)),
        'n_residues': int(len(universe.residues)),
        'n_segments': int(len(universe.segments)),
        'atom_types': atom_types,
        'residue_types': residue_types,
        'chain_info': chain_info,
        'com': [float(x) for x in universe.atoms.center_of_mass()],
        'radius_of_gyration': float(universe.atoms.radius_of_gyration()),
        'dimensions': dimensions,
        'total_mass': float(universe.atoms.total_mass()),
        'total_charge': total_charge,
        'is_protein': len(universe.select_atoms('protein')) > 0,
        'has_nucleic': len(universe.select_atoms('nucleic')) > 0,
        'has_water': len(universe.select_atoms('resname WAT or resname HOH or resname SOL')) > 0,
        'n_bonds': int(len(universe.bonds)),
        'sequence': [str(r) for r in universe.residues.resnames]
    }

def build_atom_selection(universe, options):
    """Build atom selection based on visualization options."""
    selection_terms = []
    
    if options['show_backbone']:
        selection_terms.append("backbone")
    if options['show_sidechains']:
        selection_terms.append("(not backbone) and (not name H*)")
    if options['show_hydrogens']:
        selection_terms.append("name H*")
    
    selection_string = " or ".join(selection_terms) if selection_terms else "all"
    return universe.select_atoms(selection_string)

def analyze_secondary_structure(universe):
    """Analyze protein secondary structure."""
    if not HAS_SECONDARY_STRUCTURE:
        return {
            'composition': {},
            'per_residue': [],
            'error': 'Secondary structure analysis not available. Install MDAnalysis with full dependencies.'
        }
    
    protein = universe.select_atoms("protein")
    ssa = SecondaryStructureAnalysis(protein).run()
    return {
        'composition': dict(ssa.results.keys()),
        'per_residue': ssa.results.assign.tolist()
    }

def analyze_contacts(universe, cutoff=6.0):
    """Analyze atomic contacts."""
    protein = universe.select_atoms("protein")
    contact_matrix = contacts.contact_matrix(protein.positions, radius=cutoff)
    return {
        'matrix': [[int(x) for x in row] for row in contact_matrix],
        'n_contacts': int(np.sum(contact_matrix) // 2),
        'cutoff': cutoff
    }

def calculate_rmsd(universe):
    """Calculate RMSD between frames if trajectory available."""
    protein = universe.select_atoms("protein and name CA")
    ref_coords = protein.positions.copy()
    
    if len(universe.trajectory) > 1:
        rmsd_values = []
        for ts in universe.trajectory:
            rmsd = rms.rmsd(protein.positions, ref_coords)
            rmsd_values.append(float(rmsd))
        return {'rmsd_values': rmsd_values}
    else:
        return {'rmsd_values': [0.0]}

def create_structure_visualization(selection, color_scheme='element'):
    """Create 3D structure visualization."""
    # Color mappings for different schemes
    color_maps = {
        'element': {
            'C': 'gray', 'N': 'blue', 'O': 'red', 'S': 'yellow',
            'P': 'orange', 'H': 'white'
        },
        'chain': {},  # Will be populated dynamically
        'residue': {},  # Will be populated dynamically
        'secondary': {
            'H': 'red', 'B': 'yellow', 'E': 'blue',
            'G': 'orange', 'I': 'pink', 'T': 'green',
            'C': 'gray'
        },
        'bfactor': None  # Will use a continuous color scale
    }
    
    fig = go.Figure()
    
    # Add atoms as markers
    colors = get_colors(selection, color_scheme, color_maps)
    
    fig.add_trace(go.Scatter3d(
        x=selection.positions[:, 0],
        y=selection.positions[:, 1],
        z=selection.positions[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
            symbol='circle'
        ),
        text=[f"{a.name} {a.resname} {a.resid}" for a in selection],
        name='Atoms'
    ))
    
    # Add bonds
    for bond in selection.bonds:
        start = bond.atoms[0].position
        end = bond.atoms[1].position
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        ))
    
    fig.update_layout(
        title='Structure Visualization',
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)'
        ),
        showlegend=True
    )
    
    return fig

def create_ss_plot(ss_results):
    """Create secondary structure plot."""
    fig = go.Figure(data=[
        go.Bar(
            x=list(ss_results['composition'].keys()),
            y=list(ss_results['composition'].values()),
            name='Secondary Structure Composition'
        )
    ])
    
    fig.update_layout(
        title='Secondary Structure Composition',
        xaxis_title='Structure Type',
        yaxis_title='Count'
    )
    
    return fig

def create_contact_plot(contact_results):
    """Create contact map plot."""
    fig = go.Figure(data=[
        go.Heatmap(
            z=contact_results['matrix'],
            colorscale='Viridis',
            name='Contact Map'
        )
    ])
    
    fig.update_layout(
        title=f'Contact Map (cutoff: {contact_results["cutoff"]} Å)',
        xaxis_title='Residue Index',
        yaxis_title='Residue Index'
    )
    
    return fig

def create_rmsd_plot(rmsd_results):
    """Create RMSD plot."""
    fig = go.Figure(data=[
        go.Scatter(
            x=list(range(len(rmsd_results['rmsd_values']))),
            y=rmsd_results['rmsd_values'],
            mode='lines+markers',
            name='RMSD'
        )
    ])
    
    fig.update_layout(
        title='RMSD over Frames',
        xaxis_title='Frame',
        yaxis_title='RMSD (Å)'
    )
    
    return fig

def get_colors(selection, scheme, color_maps):
    """Get colors based on the selected color scheme."""
    if scheme == 'element':
        return [color_maps['element'].get(a.element, 'gray') for a in selection]
    elif scheme == 'chain':
        unique_chains = np.unique(selection.segids)
        chain_colors = dict(zip(unique_chains, 
                              [f'rgb({r},{g},{b})' for r,g,b in np.random.randint(0, 255, (len(unique_chains), 3))]))
        return [chain_colors[a.segid] for a in selection]
    elif scheme == 'bfactor':
        bfactors = selection.tempfactors
        return bfactors
    else:
        return ['gray'] * len(selection)