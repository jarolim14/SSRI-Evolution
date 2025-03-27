# Bibliometrics Analysis Project

A Python-based project for analyzing academic papers using the Scopus API. This project provides tools for fetching, processing, and analyzing academic paper data, including citations and references.

## Features

- Fetch article data from Scopus API
- Retrieve citation and reference information
- Process and clean academic paper data
- Network analysis of paper citations
- Data visualization capabilities

## Project Structure

```
bibliometrics/
├── data/               # Raw and processed data files
├── notebooks/          # Jupyter notebooks for analysis
├── output/            # Generated outputs and visualizations
├── src/               # Source code
│   └── data_fetching/ # Scopus API interaction modules
├── .env               # Environment variables (not in git)
├── .env.example       # Example environment variables
└── pyproject.toml     # Project dependencies and configuration
```

## Prerequisites

- Python 3.11 or higher
- Scopus API keys (multiple keys supported for rate limit management)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bibliometrics
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -e ".[dev]"  # For development with all tools
# or
pip install -e .  # For just the main dependencies
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Scopus API keys and other settings
```

## Configuration

The project uses environment variables for configuration. Copy `.env.example` to `.env` and set the following variables:

- `PYTHONPATH`: Set to "src" for module imports
- `DATA_DIR`: Directory for data files
- `OUTPUT_DIR`: Directory for output files
- `LOG_LEVEL`: Logging verbosity (INFO, DEBUG, etc.)
- `SCOPUS_API_KEY_*`: Your Scopus API keys

### API Rate Limits
- `api_key_A`: 40,000 requests per week
- Other keys: 10,000 requests per week

## Environment Setup

1. Create and activate the conda environment:
```bash
conda create -n bibliometrics python=3.11
conda activate bibliometrics
```

2. Set up the environment:
```bash
# Run this script to activate the environment and install dependencies
./activate_env.sh
```

3. Configure PYTHONPATH:
   - The project uses the `src` directory as a module
   - Add the project root to PYTHONPATH in `.env`:
     ```
     PYTHONPATH=/path/to/Study-1-Bibliometrics
     ```
   - This allows importing from `src` directly:
     ```python
     from src.data_fetching.ScopusProcessor import ScopusRefFetcherPrep
     ```

4. For Jupyter notebooks:
```bash
# Install and register the environment as a Jupyter kernel
python -m ipykernel install --user --name=bibliometrics --display-name="Python (bibliometrics)"
```

5. Verify the environment:
```bash
# Should show the path to your bibliometrics environment
which python
```

## Environment Management

### Best Practices
1. Always activate the environment before running code:
```bash
conda activate bibliometrics
```

2. When adding new dependencies:
   - Add them to `pyproject.toml` first
   - Run `pip install -e ".[dev]"` to install them
   - Never install packages directly with `pip install package_name`

3. IDE/Editor Setup:
   - In VS Code/Cursor, select the Python interpreter from your conda environment
   - This ensures your editor uses the same environment as your terminal

### Troubleshooting
If you encounter `ModuleNotFoundError`:
1. Verify you're in the correct environment:
```bash
which python  # Should point to your bibliometrics environment
```

2. Check if the package is installed:
```bash
pip list | grep package_name
```

3. If needed, reinstall dependencies:
```bash
pip install -e ".[dev]"
```

## Usage

Always start your work by running:
```bash
./activate_env.sh
```

This ensures you're using the correct Python environment with all required dependencies.

1. Data Collection:
   - Use notebooks in the `notebooks/` directory to fetch data from Scopus
   - Start with `01-RetrieveScopusData.ipynb` for initial data collection
   - Use `02-RetrieveCitationData.ipynb` for citation data

2. Data Processing:
   - Use `01.1-CleanScopusData.ipynb` for data cleaning
   - Additional processing scripts are available in `src/`

3. Analysis:
   - Network analysis of citations
   - Visualization of paper relationships
   - Statistical analysis of publication patterns

## Contact

Lukas Westphal
lukas.westphal@sund.ku.dk
