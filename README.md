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



## Contact

Lukas Westphal
lukas.westphal@sund.ku.dk
