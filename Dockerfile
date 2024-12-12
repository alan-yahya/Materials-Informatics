# Use miniconda as base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment files
COPY environment.yml requirements.txt ./

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Copy application files
COPY . .

# Create uploads directory
RUN mkdir -p uploads && \
    chmod 777 uploads

# Activate conda environment and install additional requirements
SHELL ["conda", "run", "-n", "material-informatics", "/bin/bash", "-c"]
RUN pip install -r requirements.txt gunicorn

# Expose port
EXPOSE 8000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:8000 --workers=4 --thread=2 --timeout=120"

# Run the application with gunicorn
CMD ["conda", "run", "--no-capture-output", "-n", "material-informatics", "gunicorn", "app:app"] 