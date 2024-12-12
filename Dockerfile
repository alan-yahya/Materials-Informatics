# Use miniconda as base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Create conda environment with exact packages
RUN conda create -n orbitals python=3.9 numpy scipy flask plotly chembl_webresource_client mdanalysis ase pymatgen rdkit psutil -c conda-forge && \
    conda clean -afy

# Install additional packages in specific order
SHELL ["conda", "run", "-n", "orbitals", "/bin/bash", "-c"]
RUN pip install scikit-image gunicorn && \
    conda install -c openbabel openbabel

# Copy application files
COPY . .

# Create uploads directory
RUN mkdir -p uploads && \
    chmod 777 uploads

# Expose port
EXPOSE 8000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:8000 --workers=4 --thread=2 --timeout=120"

# Run the application with gunicorn
CMD ["conda", "run", "--no-capture-output", "-n", "orbitals", "gunicorn", "app:app"] 