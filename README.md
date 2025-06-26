# BibliometricAnalysis Project

A Python-based project for analyzing academic papers using the Scopus API. This project provides tools for fetching, processing, and analyzing academic paper data, including citations and references.

## Features

- Fetch article data from Scopus API
- Retrieve citation and reference information
- Process and clean academic paper data
- Network analysis of paper citations
- Data visualization capabilities

## Project Structure

```
BibliometricAnalysis/
├── data/               # Raw and processed data files
├── notebooks/          # Jupyter notebooks for analysis
├── output/             # Generated outputs and visualizations
├── src/                # Source code
│   └── data_fetching/  # Scopus API interaction modules
├── .env                # Environment variables (not in git)
├── .env.example        # Example environment variables
└── pyproject.toml      # Project dependencies and configuration
```

## Prerequisites

- Python 3.11 or higher
- Scopus API keys (multiple keys supported for rate limit management)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BibliometricAnalysis
```

2. Create and activate a virtual environment (recommended: conda):
```bash
conda create -n bibliometrics python=3.11
conda activate bibliometrics
```

3. Install dependencies:
```bash
pip install -e .[dev]  # For development with all tools
# or
pip install -e .       # For just the main dependencies
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

## Environment Management

- Only `pyproject.toml` is used for dependency management. Do not use `requirements.txt` or `setup.cfg`.
- Always activate your environment before running code:
```bash
conda activate bibliometrics
```
- When adding new dependencies:
  - Add them to `pyproject.toml` first
  - Run `pip install -e .[dev]` to install them

## Coding Standards

- Follow PEP 8 and Black formatting (see `.cursorrules` for project-specific rules)
- Use descriptive variable names and function names
- Prefer vectorized operations and method chaining in pandas
- Add docstrings to all public functions and classes
- Structure notebooks with clear markdown sections and explanations

## Usage

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

## Reproducibility & Contribution

- Ensure all notebooks can be run from start to finish in a clean environment
- Document all data sources, assumptions, and methodologies
- Use version control for all code and notebooks
- See `.cursorrules` for coding and analysis standards
- Contributions are welcome! Please open an issue or pull request for discussion

## Contact

Lukas Westphal
lukas.westphal@sund.ku.dk
