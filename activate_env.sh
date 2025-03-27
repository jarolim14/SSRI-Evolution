#!/bin/bash

# Load environment variables
source .env

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate $PYTHON_ENV

# Verify environment
echo "Using Python environment: $PYTHON_ENV"
python --version
which python

# Install dependencies if needed
pip install -e ".[dev]" 