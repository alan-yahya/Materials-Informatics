import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import rms, align

def run_basic_analysis(pdb_file):
    """
    Perform basic structural analysis on a PDB file.
    
    Parameters
    ----------
    pdb_file : str
        Path to the PDB file to analyze
        
    Returns
    -------
    None
        Results are printed to stdout
    """
    # Create a Universe from the PDB file
    universe = mda.Universe(pdb_file)
    
    # Print basic system information
    print("=== Basic System Information ===")
    print(f"System contains {len(universe.atoms)} atoms")
    print(f"Number of residues: {len(universe.residues)}")
    print(f"Number of segments: {len(universe.segments)}")
    
    # Get atom types
    atom_types = np.unique(universe.atoms.names)
    print("\n=== Atom Composition ===")
    print("Atom types present:", ', '.join(atom_types))
    for atype in atom_types:
        count = len(universe.select_atoms(f'name {atype}'))
        print(f"{atype}: {count} atoms")
    
    # Select protein atoms and check if any were found
    protein = universe.select_atoms("protein")
    if len(protein) == 0:
        print("\nNo protein atoms found. Analyzing all atoms instead.")
        protein = universe.atoms
    
    # Get basic structural properties
    try:
        print("\n=== Structural Properties ===")
        # Center of mass
        com = protein.center_of_mass()
        print(f"Center of mass: {com}")
        
        # Radius of gyration
        rog = protein.radius_of_gyration()
        print(f"Radius of gyration: {rog:.2f} Å")
        
        # Calculate distances between all atoms
        if len(protein) >= 2:
            from MDAnalysis.analysis.distances import distance_array
            distances = distance_array(protein.positions, protein.positions)
            # Get unique distances (excluding self-distances)
            unique_distances = distances[np.triu_indices_from(distances, k=1)]
            print(f"\nDistance Analysis:")
            print(f"Minimum distance between atoms: {np.min(unique_distances):.2f} Å")
            print(f"Maximum distance between atoms: {np.max(unique_distances):.2f} Å")
            print(f"Average distance between atoms: {np.mean(unique_distances):.2f} Å")
    except Exception as e:
        print(f"\nError calculating structural properties: {str(e)}")
    
    # Calculate distances between specific atoms (example with CA atoms)
    ca_atoms = universe.select_atoms("name CA")
    if len(ca_atoms) > 1:
        distances = np.linalg.norm(ca_atoms.positions[1:] - ca_atoms.positions[:-1], axis=1)
        print(f"\nAverage CA-CA distance: {np.mean(distances):.2f} Å")
    else:
        print("\nNo CA atoms found for distance calculation")
    
    # Basic structure analysis
    try:
        from MDAnalysis.analysis.secondary_structure import SecondaryStructureAnalysis
        ssa = SecondaryStructureAnalysis(protein)
        ssa.run()
        print("\n=== Secondary Structure Analysis ===")
        print("Secondary structure composition:")
        print(ssa.results.keys())
    except ImportError:
        print("\nDSSP not available for secondary structure analysis")
    except Exception as e:
        print(f"\nError in secondary structure analysis: {str(e)}")