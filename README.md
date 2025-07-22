# SSRI Bibliometric Project

This is a Python-based project for analyzing academic papers using the Scopus API. This project provides tools for fetching, processing, and analyzing academic paper data.
This repository contains the data analysis code for the manuscript:

**_The Evolution of SSRI Research: Trajectories of Knowledge Domains Across Four Decades_**

> The repository for the accompanying interactive visualization can be found here: [Immersive-SSRI-Evolution-Viz](https://github.com/jarolim14/Immersive-SSRI-Evolution-Viz)




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
├── notebooks/          # Jupyter notebooks for analysis (see below for details)
├── output/             # Generated outputs and visualizations
├── src/                # Source code
│   ├── data_fetching/  # Scopus API interaction and data cleaning modules
│   ├── main_path/      # Main path analysis and plotting tools
│   ├── nlp/            # Text processing and embedding creation
│   ├── visualization/  # Visualization utilities (edge bundling, tree hierarchy, etc.)
│   └── network/        # Network creation, analysis, and community detection
├── .env                # Environment variables (not in git)
├── .env.example        # Example environment variables
└── pyproject.toml      # Project dependencies and configuration
```

### Source Code Modules
- **data_fetching/**: Fetches and cleans data from Scopus, manages API keys, and processes references.
- **main_path/**: Implements main path analysis and plotting for citation networks.
- **nlp/**: Handles text processing and generates paper embeddings using NLP models.
- **visualization/**: Contains utilities for advanced network visualization (e.g., edge bundling, tree/dendrogram visualizations).
- **network/**: Tools for network creation, analysis, descriptive statistics, and community detection.

## Main Jupyter Notebooks

The `notebooks/` directory contains the main analysis workflow. Key notebooks include:

- **00-Introduction.ipynb**: Overview and introduction to the project and dataset.
- **01-RetrieveScopusData.ipynb**: Fetches publication data from the Scopus API.
- **01.1-CleanScopusData.ipynb**: Cleans and preprocesses the raw Scopus data.
- **02-RetrieveCitationData.ipynb**: Retrieves reference/citation data for each publication.
- **02.1-CleanScopusRetrievePubmed.ipynb**: Additional cleaning and PubMed data retrieval.
- **03-ConnectPapers.ipynb**: Merges article and reference data, prepares for network construction.
- **04-CreateTextEmbeddings.ipynb**: Processes text and generates embeddings using NLP models.
- **05-CreateNetworks.ipynb**: Constructs citation and semantic similarity networks.
- **06-DescriptiveStatistics.ipynb**: Computes and visualizes descriptive statistics of the dataset and networks.
- **07-CommunityDetection.ipynb**: Detects communities in the citation/semantic networks using clustering algorithms.
- **08-AnalysisDatasetCreation.ipynb**: Prepares the final dataset for downstream analysis and visualization.
- **09-ClusterProgressions.ipynb**: Analyzes and visualizes the evolution of research clusters over time.
- **10-PrepareThreeJs.ipynb**: Prepares data for interactive 3D visualization (Three.js).
- **11-CreatePajekNetwork.ipynb**: Exports networks for use in Pajek (main path analysis software).

Each notebook is structured to be run sequentially, but you can jump to specific steps as needed. See markdown cells in each notebook for detailed instructions and explanations.

## Environment Variables

The project uses environment variables for configuration. Copy `.env.example` to `.env` and set the following variables as needed:

- `PYTHONPATH`: Path to the `src` directory (usually set to `src`)
- `DATA_DIR`: Directory for data files (e.g., `data`)
- `OUTPUT_DIR`: Directory for output files (e.g., `output`)
- `SRC_DIR`: Path to the source code directory (e.g., `src`)
- `THREEJS_OUTPUT_DIR`: (Optional) Directory for Three.js visualization outputs
- `LOG_LEVEL`: Logging verbosity (e.g., `INFO`, `DEBUG`)
- `SCOPUS_API_KEY_*`: One or more Scopus API keys (e.g., `SCOPUS_API_KEY_A`, `SCOPUS_API_KEY_B`, ...)

If you encounter missing variable errors, check that your `.env` file contains all required keys. Refer to the top of each notebook for the specific variables used.

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
