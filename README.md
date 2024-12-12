# Materials Informatics

A suite of molecular simulation tools to cross-validate data from literature mining. Deployed as a flask app using multiple scientific Python libraries with cross-talk for materials informatics.

## Implemented Modules

### 1. OpenBabel

- Molecular property calculation
- Similarity comparison
- Reaction parsing and validation

### 2. ChEMBL

- Molecule identification
- Property retrieval
- Drug-likeness analysis

### 3. ChemML

- Molecular descriptor calculation
- Property visualization
- Structure rendering (2D/3D)

### 4. MDAnalysis

- Structure analysis
- Trajectory processing
- Property calculation

### 5. Atomic Simulation Environment (ASE)

- Structure creation
- Geometry optimization
- Property analysis

### 6. Pymatgen

- Structure creation (bulk, surface)
- Band structure calculation
- Property analysis

### 7. SciPy Quantum Orbitals

- Atomic orbital visualization
- Electron density plots
- Basis set selection

## Installation

### Using Conda
1. Create conda environment:
```bash
conda env create -f environment.yml
```

2. Activate environment:
```bash
conda activate material-informatics
```

3. Install additional dependencies:
```bash
pip install -r requirements.txt
```

### Using Docker
1. Pull the Docker image:
```bash
docker pull alanyahya/materialsinformatics:latest
```

2. Run the Docker container:
```bash
docker run -it --rm alanyahya/materialsinformatics:latest
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open web browser and navigate to:
```
http://localhost:5000
```

3. Select analysis type from the available tabs
4. Input required parameters
5. View interactive results and visualizations

## Dependencies

### Core Libraries
- Flask: Web framework
- Plotly: Interactive visualizations
- NumPy/SciPy: Scientific computing
- OpenBabel: Chemical toolbox
- ChEMBL API: Chemical database
- MDAnalysis: Molecular analysis
- ASE: Atomic simulations
- Pymatgen: Materials analysis

### Additional Tools
- RDKit: Cheminformatics
- scikit-learn: Machine learning
- pandas: Data manipulation

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Deployment

### Local
1. Start the Flask development server:
```bash
python app.py
```
Access the dev deployment at `http://localhost:5000`

Access the prod deployment at `http://localhost:8000` 
