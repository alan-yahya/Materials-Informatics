# Material Informatics Suite

In order to cross-validate data from literature mining methods, it is important to have access to a robust suite of simulation/screening tools.

This flask app demonstrates multiple scientific Python libraries with cross-talk for materials/cheminformatics.

## Implemented Modules

### 1. OpenBabel Analysis

- Structure visualization and manipulation
- Molecular property calculation
- Similarity comparison
- Reaction parsing and validation
- 3D structure generation
- Geometry optimization

### 2. ChEMBL Integration

- Molecule identification
- Property retrieval
- Structure visualization
- Drug-likeness analysis

### 3. ChemML Analysis

- Molecular descriptor calculation
- Property visualization
- Structure rendering (2D/3D)
- Basic ML-based predictions

### 4. MDAnalysis Tools

- Structure analysis
- Trajectory processing
- Property calculation
- Visualization options
- Secondary structure analysis

### 5. Atomic Simulation Environment (ASE)

- Structure creation
- Geometry optimization
- Property analysis
- Visualization tools

### 6. Pymatgen Analysis

- Structure creation (bulk, surface)
- Band structure calculation
- Property analysis
- Visualization tools

### 7. Quantum Orbital Visualization

- Atomic orbital visualization
- Electron density plots
- Basis set selection
- Interactive 3D rendering

## Installation

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

### Local Development
1. Start the Flask development server:
```bash
python app.py
```
2. Open browser at `http://localhost:5000`

### Production Deployment (Coolify)
1. Configure Coolify:
   - Source: GitHub repository
   - Build Command: `docker build -t material-informatics .`
   - Port: 8000
   - Environment Variables:
     ```
     FLASK_ENV=production
     GUNICORN_CMD_ARGS="--bind=0.0.0.0:8000 --workers=4 --thread=2 --timeout=120"
     ```

2. Deploy:
   - Push changes to GitHub
   - Coolify will automatically build and deploy

### Docker
```bash
# Build image
docker build -t material-informatics .

# Run container
docker run -p 8000:8000 material-informatics
```

Access the application at `http://localhost:8000` 