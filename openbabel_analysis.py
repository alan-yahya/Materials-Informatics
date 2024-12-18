import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openbabel import openbabel as ob
from openbabel import pybel
import tempfile
import os
import networkx as nx

class MoleculeHandler:
    def __init__(self):
        self.mol = None
        self.format = None
        self.temp_dir = tempfile.mkdtemp()
        self.graph = None
        
    def read_molecule(self, data, input_format='smiles'):
        """Read molecular data in various formats and generate 3D coordinates."""
        try:
            if input_format == 'smiles':
                # Read SMILES and create molecule
                self.mol = pybel.readstring('smi', data)
                
                # Generate 3D coordinates
                self.mol.make3D(forcefield='mmff94', steps=100)
                print("Generated 3D coordinates")
                
            elif input_format == 'xyz':
                self.mol = pybel.readstring('xyz', data)
            elif input_format == 'pdb':
                self.mol = pybel.readstring('pdb', data)
            else:
                raise ValueError(f"Unsupported input format: {input_format}")
                
            self.format = input_format
            return True
        except Exception as e:
            print(f"Error reading molecule: {str(e)}")
            return False
            
    def convert_format(self, output_format='xyz'):
        """Convert molecule to different format."""
        if self.mol is None:
            return None
            
        try:
            return self.mol.write(output_format)
        except Exception as e:
            print(f"Error converting format: {str(e)}")
            return None
            
    def optimize_geometry(self, force_field='mmff94', steps=500):
        """Optimize molecular geometry."""
        if self.mol is None:
            return False
            
        try:
            # Set up force field
            if force_field == 'mmff94':
                ff = pybel._forcefields["mmff94"]
            elif force_field == 'uff':
                ff = pybel._forcefields["uff"]
            else:
                raise ValueError(f"Unsupported force field: {force_field}")
                
            # Setup force field for molecule
            if not ff.Setup(self.mol.OBMol):
                print("Could not setup force field")
                return False
                
            # Optimize geometry
            ff.ConjugateGradients(steps)
            ff.GetCoordinates(self.mol.OBMol)
            
            return True
        except Exception as e:
            print(f"Error optimizing geometry: {str(e)}")
            return False
            
    def calculate_descriptors(self):
        """Calculate molecular descriptors."""
        if self.mol is None:
            return None
            
        try:
            # Count bonds using OBMol
            mol = self.mol.OBMol
            n_bonds = mol.NumBonds()
            
            # Basic descriptors
            descriptors = {
                'molecular_weight': self.mol.molwt,
                'exact_mass': self.mol.exactmass,
                'formula': self.mol.formula,
                'n_atoms': len(self.mol.atoms),
                'n_bonds': n_bonds,
                'n_rotatable_bonds': len([b for b in ob.OBMolBondIter(mol) if b.IsRotor()]),
                'hbd': len([a for a in self.mol.atoms if a.OBAtom.IsHbondDonor()]),
                'hba': len([a for a in self.mol.atoms if a.OBAtom.IsHbondAcceptor()])
            }
            
            # Ring analysis
            ring_data = self._analyze_rings(mol)
            descriptors.update(ring_data)
            
            # Topological descriptors
            descriptors.update(self._calculate_topological_descriptors(mol))
            
            # Volume and surface descriptors
            descriptors.update(self._calculate_volume_descriptors(mol))
            
            # Additional descriptors
            descriptors.update(self._calculate_additional_descriptors(mol))
            
            return descriptors
            
        except Exception as e:
            print(f"Error calculating descriptors: {str(e)}")
            return None
            
    def _analyze_rings(self, mol):
        """Analyze ring systems in the molecule."""
        try:
            # Get ring information
            ring_data = {}
            
            # Count different types of rings
            ring_data['n_rings_total'] = mol.GetSSSR()
            ring_data['n_aromatic_rings'] = len([ring for ring in mol.GetSSSR() if ring.IsAromatic()])
            
            # Analyze ring sizes
            ring_sizes = [ring.Size() for ring in mol.GetSSSR()]
            ring_data['ring_sizes'] = ring_sizes
            ring_data['max_ring_size'] = max(ring_sizes) if ring_sizes else 0
            ring_data['min_ring_size'] = min(ring_sizes) if ring_sizes else 0
            
            # Count rings by size
            ring_counts = {}
            for size in ring_sizes:
                ring_counts[size] = ring_counts.get(size, 0) + 1
            ring_data['ring_size_distribution'] = ring_counts
            
            return ring_data
            
        except Exception as e:
            print(f"Error analyzing rings: {str(e)}")
            return {}
            
    def _calculate_topological_descriptors(self, mol):
        """Calculate topological descriptors."""
        try:
            descriptors = {}
            
            # TPSA calculation
            tpsa = 0
            try:
                tpsa_filter = ob.OBDescriptor.FindType("TPSA")
                if tpsa_filter is not None:
                    tpsa = tpsa_filter.Predict(mol)
            except:
                print("Warning: Could not calculate TPSA")
            descriptors['tpsa'] = tpsa
            
            # Calculate basic counts
            descriptors.update({
                'n_atoms_heavy': mol.NumHvyAtoms(),
                'n_bonds_rotatable': len([b for b in ob.OBMolBondIter(mol) if b.IsRotor()]),
                'n_bonds_total': mol.NumBonds()
            })
            
            # Count different bond types
            bond_types = {'single': 0, 'double': 0, 'triple': 0, 'aromatic': 0}
            for bond in ob.OBMolBondIter(mol):
                if bond.IsAromatic():
                    bond_types['aromatic'] += 1
                elif bond.GetBondOrder() == 1:
                    bond_types['single'] += 1
                elif bond.GetBondOrder() == 2:
                    bond_types['double'] += 1
                elif bond.GetBondOrder() == 3:
                    bond_types['triple'] += 1
            
            descriptors['bond_types'] = bond_types
            
            return descriptors
            
        except Exception as e:
            print(f"Error calculating topological descriptors: {str(e)}")
            return {}
            
    def _calculate_volume_descriptors(self, mol):
        """Calculate volume and surface-related descriptors."""
        try:
            descriptors = {}
            
            # Calculate molecular volume using grid method
            try:
                mol.Center()
                box = mol.GetBox()
                volume = box[0] * box[1] * box[2]
                descriptors['molecular_volume'] = volume
            except:
                print("Warning: Could not calculate molecular volume")
            
            # Approximate atomic contributions
            total_volume = 0
            total_surface = 0
            
            # Typical atomic radii in Angstroms
            atomic_radii = {
                1: 1.20,  # H
                6: 1.70,  # C
                7: 1.55,  # N
                8: 1.52,  # O
                9: 1.47,  # F
                15: 1.80, # P
                16: 1.80, # S
                17: 1.75, # Cl
                35: 1.85, # Br
                53: 1.98  # I
            }
            
            for atom in ob.OBMolAtomIter(mol):
                atomic_num = atom.GetAtomicNum()
                radius = atomic_radii.get(atomic_num, 1.70)  # Default to carbon radius
                
                # Volume = 4/3 * pi * r^3
                atom_volume = (4/3) * np.pi * (radius ** 3)
                total_volume += atom_volume
                
                # Surface = 4 * pi * r^2
                atom_surface = 4 * np.pi * (radius ** 2)
                total_surface += atom_surface
            
            descriptors['approximate_volume'] = total_volume
            descriptors['approximate_surface'] = total_surface
            
            return descriptors
            
        except Exception as e:
            print(f"Error calculating volume descriptors: {str(e)}")
            return {}
            
    def _calculate_additional_descriptors(self, mol):
        """Calculate additional molecular descriptors."""
        try:
            descriptors = {}
            
            # Atom-based counts
            atom_counts = {}
            for atom in ob.OBMolAtomIter(mol):
                atomic_num = atom.GetAtomicNum()
                symbol = ob.GetSymbol(atomic_num)
                atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
            descriptors['atom_counts'] = atom_counts
            
            # Hybridization counts
            hybridization = {
                'sp3': 0,
                'sp2': 0,
                'sp': 0,
                'other': 0
            }
            
            for atom in ob.OBMolAtomIter(mol):
                hyb = atom.GetHyb()
                if hyb == 3:
                    hybridization['sp3'] += 1
                elif hyb == 2:
                    hybridization['sp2'] += 1
                elif hyb == 1:
                    hybridization['sp'] += 1
                else:
                    hybridization['other'] += 1
                    
            descriptors['hybridization'] = hybridization
            
            # Chirality
            n_chiral = 0
            for atom in ob.OBMolAtomIter(mol):
                if atom.IsChiral():
                    n_chiral += 1
            descriptors['n_chiral_centers'] = n_chiral
            
            # Formal charge
            total_charge = 0
            for atom in ob.OBMolAtomIter(mol):
                total_charge += atom.GetFormalCharge()
            descriptors['total_formal_charge'] = total_charge
            
            # Aromaticity
            n_aromatic = 0
            for atom in ob.OBMolAtomIter(mol):
                if atom.IsAromatic():
                    n_aromatic += 1
            descriptors['n_aromatic_atoms'] = n_aromatic
            if mol.NumAtoms() > 0:
                descriptors['fraction_aromatic_atoms'] = n_aromatic / mol.NumAtoms()
            
            return descriptors
            
        except Exception as e:
            print(f"Error calculating additional descriptors: {str(e)}")
            return {}
            
    def predict_properties(self, descriptors):
        """Predict molecular properties using descriptors."""
        if descriptors is None:
            return None
            
        try:
            # Get number of heavy atoms
            n_heavy = descriptors.get('n_atoms_heavy', 0)
            
            # Get bond type information
            bond_types = descriptors.get('bond_types', {})
            n_rotatable = descriptors.get('n_bonds_rotatable', 0)
            
            # Get topological properties
            tpsa = descriptors.get('tpsa', 0)
            
            # Estimate LogP based on structure
            logp = self._estimate_logp(
                n_heavy=n_heavy,
                n_aromatic=descriptors.get('n_aromatic_atoms', 0),
                n_rotatable=n_rotatable,
                atom_counts=descriptors.get('atom_counts', {})
            )
            
            # Estimate solubility using modified Yalkowsky equation
            # logS = 0.5 - 0.01 * (MP - 25) - logP
            # Using estimated melting point based on molecular weight
            mw = descriptors.get('molecular_weight', 0)
            est_mp = 0.1 * mw + 100
            solubility = 0.5 - 0.01 * (est_mp - 25) - logp
            
            # Calculate Lipinski's Rule of 5 parameters
            lipinski = {
                'molecular_weight_ok': mw <= 500,
                'logp_ok': -0.4 <= logp <= 5.6,
                'hbd_ok': descriptors.get('hbd', 0) <= 5,
                'hba_ok': descriptors.get('hba', 0) <= 10,
                'rotatable_bonds_ok': n_rotatable <= 10
            }
            
            # Calculate bioavailability score (0-1)
            bioavailability = sum(1 for v in lipinski.values() if v) / len(lipinski)
            
            # Estimate toxicity risk based on structural features
            toxicity_risk = self._estimate_toxicity(
                mw=mw,
                logp=logp,
                n_aromatic=descriptors.get('n_aromatic_atoms', 0),
                tpsa=tpsa,
                atom_counts=descriptors.get('atom_counts', {})
            )
            
            # Estimate biological activity based on physicochemical properties
            activity_score = self._estimate_biological_activity(
                mw=mw,
                logp=logp,
                tpsa=tpsa,
                n_rotatable=n_rotatable,
                hbd=descriptors.get('hbd', 0),
                hba=descriptors.get('hba', 0)
            )
            
            predictions = {
                'logp': logp,
                'solubility': solubility,
                'bioavailability': bioavailability,
                'lipinski_rules': lipinski,
                'toxicity_risk': toxicity_risk,
                'activity_score': activity_score
            }
            
            return predictions
            
        except Exception as e:
            print(f"Error predicting properties: {str(e)}")
            return None
            
    def _estimate_logp(self, n_heavy=0, n_aromatic=0, n_rotatable=0, atom_counts=None):
        """Estimate LogP based on molecular features."""
        if atom_counts is None:
            atom_counts = {}
            
        # Base contribution
        logp = 0.0
        
        # Atom type contributions
        atom_contributions = {
            'C': 0.5,    # Hydrophobic
            'N': -0.5,   # Hydrophilic
            'O': -1.0,   # Hydrophilic
            'F': 0.5,    # Hydrophobic
            'Cl': 1.0,   # Hydrophobic
            'Br': 1.2,   # Hydrophobic
            'I': 1.5,    # Hydrophobic
            'S': 0.2,    # Slightly hydrophobic
            'P': 0.2     # Slightly hydrophobic
        }
        
        # Add atom contributions
        for atom, count in atom_counts.items():
            logp += atom_contributions.get(atom, 0) * count
            
        # Adjust for aromaticity
        logp += 0.3 * n_aromatic
        
        # Adjust for rotatable bonds
        logp += 0.1 * n_rotatable
        
        return logp
        
    def _estimate_toxicity(self, mw=0, logp=0, n_aromatic=0, tpsa=0, atom_counts=None):
        """Estimate toxicity risk based on structural features."""
        if atom_counts is None:
            atom_counts = {}
            
        risk_score = 0
        
        # High molecular weight increases risk
        if mw > 800:
            risk_score += 1
            
        # Very high or low LogP increases risk
        if logp > 5 or logp < -2:
            risk_score += 1
            
        # High number of aromatic rings increases risk
        if n_aromatic > 12:  # Assuming aromatic atoms, not rings
            risk_score += 1
            
        # High TPSA might indicate permeability issues
        if tpsa > 140:
            risk_score += 1
            
        # Check for potentially toxic elements
        toxic_elements = {'Cl', 'Br', 'I', 'P', 'S'}
        if any(elem in toxic_elements for elem in atom_counts.keys()):
            risk_score += 1
            
        return {
            'score': risk_score,
            'level': 'Low' if risk_score <= 1 else 'Medium' if risk_score <= 2 else 'High'
        }
        
    def _estimate_biological_activity(self, mw=0, logp=0, tpsa=0, n_rotatable=0, hbd=0, hba=0):
        """Estimate potential biological activity based on drug-likeness rules."""
        score = 0
        max_score = 6
        
        # Molecular weight between 160 and 500
        if 160 <= mw <= 500:
            score += 1
            
        # LogP between -0.4 and 5.6 (extended Lipinski range)
        if -0.4 <= logp <= 5.6:
            score += 1
            
        # Topological polar surface area between 20 and 140
        if 20 <= tpsa <= 140:
            score += 1
            
        # Number of rotatable bonds <= 10
        if n_rotatable <= 10:
            score += 1
            
        # Hydrogen bond donors <= 5
        if hbd <= 5:
            score += 1
            
        # Hydrogen bond acceptors <= 10
        if hba <= 10:
            score += 1
            
        return {
            'score': score / max_score,
            'level': 'Low' if score <= 2 else 'Medium' if score <= 4 else 'High'
        }
        
    def create_molecular_graph(self):
        """Create a NetworkX graph representation of the molecule."""
        if self.mol is None:
            return None
            
        try:
            # Create new graph
            G = nx.Graph()
            
            # Add nodes (atoms)
            for atom in self.mol.atoms:
                # Get atom properties
                atomic_num = atom.atomicnum
                symbol = ob.GetSymbol(atomic_num)
                coords = atom.coords
                
                # Add node with properties
                G.add_node(atom.idx,
                          symbol=symbol,
                          atomic_num=atomic_num,
                          coords=coords,
                          formal_charge=atom.formalcharge,
                          is_aromatic=atom.OBAtom.IsAromatic())
            
            # Add edges (bonds)
            for bond in ob.OBMolBondIter(self.mol.OBMol):
                # Get bond properties
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                bond_order = bond.GetBondOrder()
                is_aromatic = bond.IsAromatic()
                
                # Add edge with properties
                G.add_edge(begin_idx, end_idx,
                          bond_order=bond_order,
                          is_aromatic=is_aromatic)
            
            self.graph = G
            return True
            
        except Exception as e:
            print(f"Error creating molecular graph: {str(e)}")
            return False
            
    def get_graph_properties(self):
        """Calculate graph-theoretic properties of the molecular graph."""
        if self.graph is None:
            return None
            
        try:
            properties = {
                'n_nodes': self.graph.number_of_nodes(),
                'n_edges': self.graph.number_of_edges(),
                'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
                'density': nx.density(self.graph),
                'is_connected': nx.is_connected(self.graph),
                'clustering_coefficient': nx.average_clustering(self.graph),
                'shortest_paths': dict(nx.all_pairs_shortest_path_length(self.graph)),
                'centrality': {
                    'degree': dict(nx.degree_centrality(self.graph)),
                    'betweenness': dict(nx.betweenness_centrality(self.graph)),
                    'closeness': dict(nx.closeness_centrality(self.graph))
                }
            }
            
            # Calculate ring information
            cycles = nx.cycle_basis(self.graph)
            properties['rings'] = {
                'count': len(cycles),
                'sizes': [len(cycle) for cycle in cycles],
                'cycles': cycles
            }
            
            return properties
            
        except Exception as e:
            print(f"Error calculating graph properties: {str(e)}")
            return None
            
    def create_visualization(self, plot_type='3d'):
        """Create visualization of molecular structure and graph."""
        if self.mol is None:
            return None
            
        try:
            # Calculate descriptors and predictions
            descriptors = self.calculate_descriptors()
            predictions = self.predict_properties(descriptors)
            
            # Create molecular graph if not already created
            if self.graph is None:
                self.create_molecular_graph()
            
            # Get graph properties
            graph_props = self.get_graph_properties() if self.graph else None
            
            # Create subplots: 3D structure, graph visualization, and properties
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
                      [{'type': 'table', 'colspan': 2}, None]],
                subplot_titles=('3D Structure', 'Molecular Graph', 'Properties & Analysis')
            )
            
            # Add 3D structure visualization
            coords = np.array([[a.coords[i] for i in range(3)] for a in self.mol.atoms])
            elements = [a.type for a in self.mol.atoms]
            
            # Element-color mapping
            element_colors = {
                'C': 'gray', 'H': 'white', 'O': 'red', 'N': 'blue',
                'S': 'yellow', 'P': 'orange', 'F': 'green', 'Cl': 'green',
                'Br': 'brown', 'I': 'purple'
            }
            
            # Plot atoms
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=[element_colors.get(el, 'gray') for el in elements],
                        symbol='circle'
                    ),
                    text=elements,
                    name='Atoms'
                ),
                row=1, col=1
            )
            
            # Add bonds
            for bond in ob.OBMolBondIter(self.mol.OBMol):
                begin_idx = bond.GetBeginAtomIdx() - 1
                end_idx = bond.GetEndAtomIdx() - 1
                begin_coords = coords[begin_idx]
                end_coords = coords[end_idx]
                
                fig.add_trace(
                    go.Scatter3d(
                        x=[begin_coords[0], end_coords[0]],
                        y=[begin_coords[1], end_coords[1]],
                        z=[begin_coords[2], end_coords[2]],
                        mode='lines',
                        line=dict(color='gray', width=2),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Add graph visualization
            if self.graph:
                # Create spring layout
                pos = nx.spring_layout(self.graph)
                
                # Add edges
                edge_x = []
                edge_y = []
                for edge in self.graph.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                fig.add_trace(
                    go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        mode='lines',
                        name='Bonds',
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                # Add nodes
                node_x = []
                node_y = []
                node_text = []
                node_color = []
                
                for node in self.graph.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    symbol = self.graph.nodes[node]['symbol']
                    node_text.append(f"{symbol}{node}")
                    node_color.append(element_colors.get(symbol, 'gray'))
                
                fig.add_trace(
                    go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        marker=dict(
                            size=10,
                            color=node_color,
                            line_width=1
                        ),
                        text=node_text,
                        textposition="top center",
                        name='Atoms',
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Create property table
            if descriptors and predictions and graph_props:
                table_data = {
                    'Property': [],
                    'Value': []
                }
                
                # Basic properties
                table_data['Property'].extend([
                    'Formula', 'Molecular Weight', 'Number of Atoms',
                    'Number of Bonds', 'Number of Rings'
                ])
                table_data['Value'].extend([
                    descriptors['formula'],
                    f"{descriptors['molecular_weight']:.2f}",
                    str(descriptors['n_atoms']),
                    str(descriptors['n_bonds']),
                    str(graph_props['rings']['count'])
                ])
                
                # Graph properties
                table_data['Property'].extend([
                    'Average Degree',
                    'Graph Density',
                    'Clustering Coefficient'
                ])
                table_data['Value'].extend([
                    f"{graph_props['average_degree']:.2f}",
                    f"{graph_props['density']:.3f}",
                    f"{graph_props['clustering_coefficient']:.3f}"
                ])
                
                # Predictions
                table_data['Property'].extend([
                    'LogP', 'Solubility (logS)', 'Bioavailability Score'
                ])
                table_data['Value'].extend([
                    f"{predictions['logp']:.2f}",
                    f"{predictions['solubility']:.2f}",
                    f"{predictions['bioavailability']:.2f}"
                ])
                
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=list(table_data.keys()),
                            fill_color='paleturquoise',
                            align='left'
                        ),
                        cells=dict(
                            values=list(table_data.values()),
                            fill_color='lavender',
                            align='left'
                        )
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title='Molecular Analysis with Graph Representation',
                scene=dict(
                    xaxis_title='X (Å)',
                    yaxis_title='Y (Å)',
                    zaxis_title='Z (Å)'
                ),
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None
            
    def convert_to_pdb(self):
        """Convert molecule to PDB format."""
        if self.mol is None:
            return None
        try:
            # Convert to PDB format
            return self.mol.write("pdb")
        except Exception as e:
            print(f"Error converting to PDB: {str(e)}")
            return None
            
    def calculate_similarity(self, other_mol, fp_type='fp2'):
        """Calculate molecular similarity using fingerprints."""
        if self.mol is None or other_mol is None:
            return None
        
        try:
            # Get fingerprints for both molecules
            fp1 = self._get_fingerprint(self.mol, fp_type)
            fp2 = self._get_fingerprint(other_mol, fp_type)
            
            if fp1 is None or fp2 is None:
                return None
            
            # Calculate Tanimoto coefficient
            tanimoto = self._calculate_tanimoto(fp1, fp2)
            
            return tanimoto
            
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return None
            
    def _get_fingerprint(self, mol, fp_type):
        """Generate molecular fingerprint."""
        try:
            if fp_type == 'fp2':
                return mol.calcfp(fptype='FP2')
            elif fp_type == 'fp3':
                return mol.calcfp(fptype='FP3')
            elif fp_type == 'fp4':
                return mol.calcfp(fptype='FP4')
            elif fp_type == 'maccs':
                return mol.calcfp(fptype='MACCS')
            else:
                raise ValueError(f"Unsupported fingerprint type: {fp_type}")
        except Exception as e:
            print(f"Error generating fingerprint: {str(e)}")
            return None
            
    def _calculate_tanimoto(self, fp1, fp2):
        """Calculate Tanimoto coefficient between two fingerprints."""
        try:
            return fp1 | fp2
        except Exception as e:
            print(f"Error calculating Tanimoto coefficient: {str(e)}")
            return None

def _get_similarity_description(mol1, mol2, similarity, fp_type):
    """Generate detailed description of molecular similarity comparison."""
    try:
        # Get basic molecular properties
        props1 = {
            'formula': mol1.formula,
            'weight': mol1.molwt,
            'n_atoms': len(mol1.atoms),
            'n_bonds': mol1.OBMol.NumBonds()
        }
        
        props2 = {
            'formula': mol2.formula,
            'weight': mol2.molwt,
            'n_atoms': len(mol2.atoms),
            'n_bonds': mol2.OBMol.NumBonds()
        }
        
        # Generate description
        description = f"""
<b>Molecular Similarity Analysis</b><br>
<br>
<b>Molecule 1:</b><br>
• Formula: {props1['formula']}<br>
• Molecular Weight: {props1['weight']:.2f} g/mol<br>
• Number of Atoms: {props1['n_atoms']}<br>
• Number of Bonds: {props1['n_bonds']}<br>
<br>
<b>Molecule 2:</b><br>
• Formula: {props2['formula']}<br>
• Molecular Weight: {props2['weight']:.2f} g/mol<br>
• Number of Atoms: {props2['n_atoms']}<br>
• Number of Bonds: {props2['n_bonds']}<br>
<br>
<b>Similarity Analysis:</b><br>
• Fingerprint Type: {fp_type.upper()}<br>
• Tanimoto Coefficient: {similarity:.3f}<br>
<br>
<b>Interpretation:</b><br>
• The Tanimoto coefficient ranges from 0 (completely different) to 1 (identical)<br>
• Values > 0.85 indicate high structural similarity<br>
• Values 0.5-0.85 indicate moderate similarity<br>
• Values < 0.5 indicate low similarity<br>
<br>
<b>Method Details:</b><br>
• {fp_type.upper()} fingerprint type was used to encode molecular structure<br>
• Similarity was calculated using the Tanimoto coefficient (Tc)<br>
• Tc = (features in common) / (total unique features)<br>
"""
        
        # Add similarity interpretation
        if similarity > 0.85:
            description += "• <b>Result:</b> These molecules show high structural similarity"
        elif similarity > 0.5:
            description += "• <b>Result:</b> These molecules show moderate structural similarity"
        else:
            description += "• <b>Result:</b> These molecules show low structural similarity"
            
        return description
        
    except Exception as e:
        print(f"Error generating similarity description: {str(e)}")
        return "Error generating detailed description"

def compare_molecules(smiles1, smiles2, fp_type='fp2'):
    """Compare two molecules using fingerprint similarity."""
    try:
        # Initialize handlers
        handler1 = MoleculeHandler()
        handler2 = MoleculeHandler()
        
        # Read molecules
        if not handler1.read_molecule(smiles1, 'smiles') or not handler2.read_molecule(smiles2, 'smiles'):
            return None, None, None
            
        # Calculate similarity
        similarity = handler1.calculate_similarity(handler2.mol, fp_type)
        
        # Generate detailed description
        description = _get_similarity_description(handler1.mol, handler2.mol, similarity, fp_type)
        
        # Create visualization with 2D structure comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Molecule 1', 'Molecule 2', 'Analysis Details'),
            specs=[
                [{'type': 'scene'}, {'type': 'scene'}],
                [{'type': 'table', 'colspan': 2}, None]
            ],
            vertical_spacing=0.15
        )
        
        # Add 3D structures
        for i, handler in enumerate([handler1, handler2], 1):
            coords = np.array([[a.coords[i] for i in range(3)] for a in handler.mol.atoms])
            elements = [a.type for a in handler.mol.atoms]
            
            # Plot atoms
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=[{'C': 'gray', 'H': 'white', 'O': 'red', 'N': 'blue'}.get(el, 'gray') 
                               for el in elements],
                        symbol='circle'
                    ),
                    text=elements,
                    name=f'Atoms (Mol {i})'
                ),
                row=1, col=i
            )
            
            # Add bonds
            for bond in ob.OBMolBondIter(handler.mol.OBMol):
                begin_idx = bond.GetBeginAtomIdx() - 1
                end_idx = bond.GetEndAtomIdx() - 1
                begin_coords = coords[begin_idx]
                end_coords = coords[end_idx]
                
                fig.add_trace(
                    go.Scatter3d(
                        x=[begin_coords[0], end_coords[0]],
                        y=[begin_coords[1], end_coords[1]],
                        z=[begin_coords[2], end_coords[2]],
                        mode='lines',
                        line=dict(color='gray', width=2),
                        showlegend=False
                    ),
                    row=1, col=i
                )
        
        # Add description as a table
        fig.add_trace(
            go.Table(
                cells=dict(
                    values=[[description]],
                    align=['left'],
                    font=dict(size=12),
                    height=30
                )
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Molecular Comparison (Similarity: {similarity:.3f})',
            height=800,  # Increased height to accommodate description
            showlegend=True,
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)'
            ),
            scene2=dict(  # Settings for second 3D scene
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)'
            )
        )
        
        return similarity, description, fig
        
    except Exception as e:
        print(f"Error comparing molecules: {str(e)}")
        return None, None, None

def run_openbabel_analysis(input_format='smiles', data=None, **kwargs):
    """Run molecular analysis using OpenBabel."""
    try:
        # Validate input data
        if not data or data.strip() == '':
            # If no data provided, use example molecules
            examples = {
                'acetic_acid': 'CC(=O)O',  # Acetic acid
                'cumic_acid': 'CC(C)C1=CC=C(C=C1)C(=O)O'  # Cumic acid
            }
            data = examples['cumic_acid']  # Using cumic acid as default
            print(f"No input provided. Using example molecule: {data}")
            
        print(f"Starting OpenBabel analysis with format: {input_format}")
        
        # Initialize handler
        handler = MoleculeHandler()
        
        # Read molecule
        if not handler.read_molecule(data, input_format):
            # Create error figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Failed to read molecular structure. Please check your input.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title='Error: Invalid Input',
                height=400
            )
            return fig, None
            
        print("Molecule read successfully")
        
        # Optimize geometry if requested
        if kwargs.get('optimize', False):
            force_field = kwargs.get('force_field', 'mmff94')
            steps = kwargs.get('steps', 500)
            if handler.optimize_geometry(force_field, steps):
                print("Geometry optimized")
            else:
                print("Geometry optimization failed")
        
        # Create visualization
        fig = handler.create_visualization()
        print("Visualization created")
        
        # Convert to PDB format
        pdb_data = handler.convert_to_pdb()
        
        return fig, pdb_data
        
    except Exception as e:
        print(f"Error in OpenBabel analysis: {str(e)}")
        # Create error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Analysis failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title='Error',
            height=400
        )
        return fig, None 

def analyze_reaction(reaction_smiles, validation_options=None):
    """Analyze and validate a chemical reaction."""
    try:
        if validation_options is None:
            validation_options = {
                'check_mass_balance': True,
                'check_atom_mapping': True,
                'check_valence': True
            }
            
        # Parse reaction SMILES
        parts = reaction_smiles.split('>')
        if len(parts) == 2:
            reactants_smiles = parts[0]
            products_smiles = parts[1]
            agent_smiles = ""
        elif len(parts) == 3:
            reactants_smiles = parts[0]
            agent_smiles = parts[1]
            products_smiles = parts[2]
        else:
            raise ValueError("Invalid reaction SMILES format")

        # Create reactant and product molecules
        reactants = []
        products = []
        
        # Parse reactants
        for smiles in reactants_smiles.split('.'):
            if smiles.strip():
                mol = pybel.readstring('smi', smiles)
                mol.make3D()
                reactants.append(mol)
                
        # Parse products
        for smiles in products_smiles.split('.'):
            if smiles.strip():
                mol = pybel.readstring('smi', smiles)
                mol.make3D()
                products.append(mol)
                
        validation_results = []
        warnings = []
        errors = []
        
        # Check mass balance
        if validation_options.get('check_mass_balance'):
            reactant_mass = sum(mol.molwt for mol in reactants)
            product_mass = sum(mol.molwt for mol in products)
            mass_diff = abs(reactant_mass - product_mass)
            
            if mass_diff < 0.01:
                validation_results.append(("Mass Balance", True, f"Mass is conserved (Δm = {mass_diff:.3f} g/mol)"))
            else:
                warnings.append(f"Mass balance issue: Δm = {mass_diff:.3f} g/mol")
        
        # Check atom mapping
        if validation_options.get('check_atom_mapping'):
            reactant_atoms = sum(len(mol.atoms) for mol in reactants)
            product_atoms = sum(len(mol.atoms) for mol in products)
            
            if reactant_atoms == product_atoms:
                validation_results.append(("Atom Count", True, f"Atom count matches ({reactant_atoms} atoms)"))
            else:
                errors.append(f"Atom count mismatch: {reactant_atoms} (reactants) vs {product_atoms} (products)")
        
        # Check valence rules
        if validation_options.get('check_valence'):
            valence_issues = []
            
            # Define typical valences for common elements
            typical_valences = {
                'C': 4, 'N': 3, 'O': 2, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1,
                'S': [2, 4, 6], 'P': [3, 5], 'B': 3
            }
            
            for i, mol in enumerate(reactants + products):
                for atom in mol.atoms:
                    symbol = atom.type
                    if symbol in typical_valences:
                        # Count bonds and formal charge
                        total_bonds = sum(bond.GetBondOrder() 
                                        for bond in ob.OBAtomBondIter(atom.OBAtom))
                        formal_charge = atom.OBAtom.GetFormalCharge()
                        
                        # Check if valence is valid
                        expected_valences = typical_valences[symbol]
                        if isinstance(expected_valences, list):
                            valid_valence = total_bonds + abs(formal_charge) in expected_valences
                        else:
                            valid_valence = total_bonds + abs(formal_charge) == expected_valences
                            
                        if not valid_valence:
                            valence_issues.append(
                                f"Unusual valence for {symbol} atom in "
                                f"{'reactant' if i < len(reactants) else 'product'} "
                                f"{i % len(reactants) + 1} "
                                f"(bonds: {total_bonds}, charge: {formal_charge})"
                            )
            
            if not valence_issues:
                validation_results.append(("Valence Rules", True, "All atoms have typical valence states"))
            else:
                warnings.extend(valence_issues)  # Make these warnings instead of errors
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Reactants', 'Products'),
            specs=[[{'type': 'scene'}], [{'type': 'scene'}]],
            vertical_spacing=0.2
        )
        
        # Plot reactants
        _add_molecules_to_plot(fig, reactants, row=1, col=1)
        
        # Plot products
        _add_molecules_to_plot(fig, products, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)'
            ),
            scene2=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)'
            )
        )
        
        # Add reaction equation to title
        equation = f"{reactants_smiles}"
        if agent_smiles:
            equation += f" ({agent_smiles})"
        equation += f" → {products_smiles}"
        fig.update_layout(
            title=f"Reaction Analysis: {equation}"
        )
        
        # Generate HTML for validation results
        validation_html = _generate_validation_html(validation_results, warnings, errors)
        
        return {
            'is_valid': len(errors) == 0,
            'validation_results': validation_results,
            'warnings': warnings,
            'errors': errors,
            'validation_html': validation_html,
            'plot': fig
        }
        
    except Exception as e:
        print(f"Error analyzing reaction: {str(e)}")
        return None

def _add_molecules_to_plot(fig, molecules, row, col):
    """Add molecules to the reaction plot."""
    spacing = 5  # Spacing between molecules
    current_offset = 0
    
    for mol in molecules:
        # Get coordinates from atoms
        coords = np.array([[atom.coords[i] for i in range(3)] for atom in mol.atoms])
        
        # Offset molecule position
        coords[:, 0] += current_offset
        
        # Plot atoms
        elements = [atom.type for atom in mol.atoms]
        
        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode='markers',
                marker=dict(
                    size=10,
                    color=[{'C': 'gray', 'H': 'white', 'O': 'red', 'N': 'blue'}.get(el, 'gray') 
                           for el in elements],
                    symbol='circle'
                ),
                text=elements,
                name=f'Molecule {current_offset//spacing + 1}'
            ),
            row=row, col=col
        )
        
        # Add bonds
        for atom in mol.atoms:
            for bond in ob.OBAtomBondIter(atom.OBAtom):
                begin_atom = mol.atoms[bond.GetBeginAtomIdx() - 1]
                end_atom = mol.atoms[bond.GetEndAtomIdx() - 1]
                begin_coords = np.array(begin_atom.coords) + [current_offset, 0, 0]
                end_coords = np.array(end_atom.coords) + [current_offset, 0, 0]
                
                fig.add_trace(
                    go.Scatter3d(
                        x=[begin_coords[0], end_coords[0]],
                        y=[begin_coords[1], end_coords[1]],
                        z=[begin_coords[2], end_coords[2]],
                        mode='lines',
                        line=dict(color='gray', width=2),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        current_offset += spacing

def _generate_validation_html(validation_results, warnings, errors):
    """Generate HTML for validation results."""
    html = []
    
    # Add successful validations
    for name, success, message in validation_results:
        html.append(f"""
            <div class="validation-result validation-success">
                <strong>{name}:</strong> {message}
            </div>
        """)
    
    # Add warnings
    for warning in warnings:
        html.append(f"""
            <div class="validation-result validation-warning">
                <strong>Warning:</strong> {warning}
            </div>
        """)
    
    # Add errors
    for error in errors:
        html.append(f"""
            <div class="validation-result validation-error">
                <strong>Error:</strong> {error}
            </div>
        """)
    
    return "".join(html) 