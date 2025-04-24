"""
Cluster Interconnectivity Analysis Script

This script analyzes and visualizes the interconnectivity between academic paper clusters.
It processes citation networks to show how different research clusters reference each other,
creating matrices and heatmap visualizations.

The script:
1. Loads cluster and reference data from pickle and JSON files
2. Extracts safety and indication clusters from a predefined legend
3. Creates interconnectivity matrices showing citation patterns between clusters
4. Analyzes internal vs. external reference patterns for each cluster
5. Generates heatmap visualizations of the interconnectivity

Requirements:
- Environment variables in .env file: DATA_DIR, OUTPUT_DIR
- Input data:
  - Pickled DataFrame with cluster assignments and reference data
  - JSON files with cluster labels and legend information

Author: Lukas Westphal
Date: April 16, 2025
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv


class ClusterAnalyzer:
    """
    Class to handle cluster analysis operations.

    This class provides methods for analyzing interconnectivity between
    academic paper clusters based on citation networks. It loads data,
    extracts cluster categories, creates interconnectivity matrices,
    analyzes reference patterns, and generates visualizations.

    The analysis focuses on how different research clusters reference each other,
    measuring both internal (within-cluster) and external (between-cluster) citation
    patterns to identify knowledge flow and research community structures.
    """

    def __init__(self, data_dir: str, output_dir: str, parameter: str = "alpha0.3_k10_res0.002"):
        """Initialize with paths and parameters."""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.parameter = parameter
        self.cluster_col = f"cluster_{parameter}"
        self.df = None
        self.cluster_label_dict = None
        self.legend = None
        self.safety_clusters = None
        self.indication_clusters = None

    def load_data(self):
        """Load the dataset and reference files."""
        # Load DataFrame
        pdf = os.path.join(self.data_dir, "08-analysis-data/2025/df_analysis.pkl")
        self.df = pd.read_pickle(pdf)
        print(f"DataFrame loaded with columns: {self.df.columns}")

        # Load cluster labels
        labels_path = os.path.join(
            self.output_dir,
            "cluster-qualifications_2025/cluster-label-tree/cluster_labels_filtered.json"
        )
        with open(labels_path, "r") as f:
            self.cluster_label_dict = json.load(f)

        # Load legend
        legend_path = os.path.join(
            self.output_dir,
            "cluster-qualifications_2025/cluster-label-tree/legend_labels_2025.json"
        )
        with open(legend_path, "r") as f:
            self.legend = json.load(f)

        return self

    def extract_cluster_categories(self):
        """Extract safety and indication clusters from legend."""
        # Get all safety cluster numbers
        self.safety_clusters = sorted(set(
            int(k) for k in self._get_dict_keys(self.legend['Safety']) if k.isdigit()
        ))
        print(f'Number of safety clusters: {len(self.safety_clusters)}')

        # Get all indication cluster numbers
        self.indication_clusters = sorted(
            int(k) for k in self._get_dict_keys(self.legend['Indications']) if k.isdigit()
        )
        print(f'Number of indication clusters: {len(self.indication_clusters)}')

        return self

    @staticmethod
    def _get_dict_keys(obj):
        """Extract all dictionary keys from nested structures."""
        if isinstance(obj, dict):
            # Get keys from current level and recurse for nested structures
            return list(obj.keys()) + [k for v in obj.values() for k in ClusterAnalyzer._get_dict_keys(v)]
        elif isinstance(obj, list):
            # Recurse for each item in the list
            return [k for item in obj for k in ClusterAnalyzer._get_dict_keys(item)]
        else:
            # Base case: not a dict or list
            return []

    def create_cluster_interconnectivity_matrix(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create matrices showing the interconnectivity between clusters using filtered_reference_eids.
        """
        # Get unique clusters and create mappings
        clusters = sorted(self.df[self.cluster_col].unique(), key=int)
        n_clusters = len(clusters)

        if n_clusters == 0:
            raise ValueError("No clusters found in the data")

        # Create a mapping of eid to cluster for faster lookups
        eid_to_cluster = self.df.set_index('eid')[self.cluster_col].to_dict()

        # Create a mapping of cluster to index for faster lookups
        cluster_to_idx = {cluster: idx for idx, cluster in enumerate(clusters)}

        # Initialize edge count matrix
        edge_counts = np.zeros((n_clusters, n_clusters))

        # Pre-process and organize all references by source cluster
        references_by_cluster = defaultdict(list)

        # Build a list of all references with their source clusters
        for _, row in self.df.iterrows():
            cluster_idx = cluster_to_idx[row[self.cluster_col]]
            if isinstance(row["filtered_reference_eids"], list):
                references_by_cluster[cluster_idx].extend(row["filtered_reference_eids"])
            else:
                print(f"Unexpected reference type: {type(row['filtered_reference_eids'])}")

        # Process all references at once per cluster
        for src_cluster_idx, reference_eids in references_by_cluster.items():
            # Filter references to only those in our dataset
            valid_refs = [eid for eid in reference_eids if eid in eid_to_cluster]

            if valid_refs:
                # Get the target clusters for all references
                target_clusters = [eid_to_cluster[eid] for eid in valid_refs]

                # Convert target clusters to indices
                target_indices = [cluster_to_idx[c] for c in target_clusters]

                # Count occurrences of each target cluster
                for tgt_idx in target_indices:
                    edge_counts[src_cluster_idx, tgt_idx] += 1

        # Create DataFrames with sorted cluster labels
        cluster_labels = [f"Cluster {c}" for c in clusters]
        edge_counts_df = pd.DataFrame(
            edge_counts,
            index=cluster_labels,
            columns=cluster_labels
        )

        # Calculate cluster sizes once (more efficient)
        cluster_sizes = self.df[self.cluster_col].value_counts()
        # Sort the index numerically
        cluster_sizes = cluster_sizes.reindex(sorted(cluster_sizes.index, key=int)).values

        # Create normalized matrix using broadcasting
        normalized_counts = np.zeros((n_clusters, n_clusters))
        size_matrix = np.outer(cluster_sizes, cluster_sizes)
        mask = size_matrix > 0
        normalized_counts[mask] = edge_counts[mask] / size_matrix[mask]

        normalized_df = pd.DataFrame(
            normalized_counts,
            index=cluster_labels,
            columns=cluster_labels
        )

        return edge_counts_df, normalized_df

    def analyze_cluster_interconnectivity(self) -> Dict:
        """
        Analyze the interconnectivity between clusters using filtered_reference_eids.
        """
        # Get unique clusters
        clusters = sorted(self.df[self.cluster_col].unique())

        # Create a mapping of eid to cluster for faster lookups
        eid_to_cluster = self.df.set_index('eid')[self.cluster_col].to_dict()

        # Pre-compute all edge information at once
        edge_data = defaultdict(lambda: {"internal": 0, "external": 0})

        # Group by cluster for more efficient processing
        cluster_groups = self.df.groupby(self.cluster_col)

        for cluster, cluster_df in cluster_groups:
            # Process all rows in a cluster at once
            for _, row in cluster_df.iterrows():
                if isinstance(row["filtered_reference_eids"], list) and row["filtered_reference_eids"]:
                    # Filter to valid references
                    valid_refs = [eid for eid in row["filtered_reference_eids"] if eid in eid_to_cluster]

                    if valid_refs:
                        # Get target clusters for all references at once
                        target_clusters = [eid_to_cluster[eid] for eid in valid_refs]

                        # Count internal vs external references
                        internal_count = sum(1 for c in target_clusters if c == cluster)
                        external_count = len(target_clusters) - internal_count

                        edge_data[cluster]["internal"] += internal_count
                        edge_data[cluster]["external"] += external_count

        # Calculate metrics
        cluster_metrics = {}
        for cluster in clusters:
            internal_edges = edge_data[cluster]["internal"]
            external_edges = edge_data[cluster]["external"]
            total_edges = internal_edges + external_edges

            internal_ratio = internal_edges / total_edges if total_edges > 0 else 0
            external_ratio = external_edges / total_edges if total_edges > 0 else 0

            cluster_metrics[cluster] = {
                "size": len(self.df[self.df[self.cluster_col] == cluster]),
                "internal_edges": internal_edges,
                "external_edges": external_edges,
                "internal_ratio": internal_ratio,
                "external_ratio": external_ratio,
            }

        return cluster_metrics

    def print_metrics_summary(self, metrics_df):
        """Print summary statistics from metrics dataframe."""
        print("\nCluster Interconnectivity Summary:")
        print("-" * 50)
        print(f"Total number of clusters: {len(metrics_df)}")
        print(f"Average cluster size: {metrics_df['size'].mean():.2f}")
        print(f"Average internal edge ratio: {metrics_df['internal_ratio'].mean():.3f}")
        print(f"Average external edge ratio: {metrics_df['external_ratio'].mean():.3f}")

    def plot_heatmap(self, df_to_plot, cluster_type="All", diag_excluded=True, cap_max=True):
        """Plot heatmap for cluster interconnectivity."""
        # Set figure size based on matrix dimensions
        figsize = (25, 25) if cluster_type == "All" else (20, 15)

        # Create labels for axes
        clusters = [cluster.replace('Cluster ', '') for cluster in df_to_plot.index.tolist()]
        x_labels = [self.cluster_label_dict[cluster] for cluster in clusters]
        y_labels = [self.cluster_label_dict[cluster] for cluster in clusters]

        # Exclude diagonal if specified
        if diag_excluded:
            np.fill_diagonal(df_to_plot.values, np.nan)

        # Set maximum value for color scale
        if cap_max:
            vmax = np.nanpercentile(df_to_plot.values, 95)
        else:
            vmax = df_to_plot.max().max()

        # Create plot
        plt.figure(figsize=figsize)
        sns.heatmap(df_to_plot, annot=False, xticklabels=x_labels, yticklabels=y_labels,
                    cmap="viridis", vmin=0, vmax=vmax)

        title = f"Edge Counts Between {cluster_type} Clusters"
        plt.title(title)
        plt.tight_layout()

        # Create output directory if it doesn't exist
        output_path = os.path.join(self.output_dir, "cluster-interconnectivity")
        os.makedirs(output_path, exist_ok=True)

        # Save figure
        plt.savefig(os.path.join(output_path, f"{title.replace(' ', '_')}.png"), dpi=400)
        plt.show()
        plt.close()



"""
Main function to run the cluster interconnectivity analysis.

This function:
1. Loads environment variables from .env file
2. Initializes the ClusterAnalyzer with appropriate paths
3. Loads data and extracts cluster categories
4. Creates interconnectivity matrices
5. Analyzes cluster interconnectivity metrics
6. Generates visualizations for all clusters and safety clusters

The function orchestrates the entire analysis pipeline and produces
both numerical metrics and visual heatmaps showing citation patterns.
"""
# Load environment variables
load_dotenv()

# Access environment variables
data_dir = os.getenv("DATA_DIR")
output_dir = os.getenv("OUTPUT_DIR")

if not data_dir or not output_dir:
    raise ValueError("DATA_DIR and OUTPUT_DIR must be set in .env file")

# Initialize analyzer
analyzer = ClusterAnalyzer(data_dir, output_dir)

# Load data and extract cluster categories
analyzer.load_data().extract_cluster_categories()

# Create output directory
os.makedirs(os.path.join(output_dir, "cluster-interconnectivity"), exist_ok=True)

# Create interconnectivity matrices
edge_counts_df, normalized_df = analyzer.create_cluster_interconnectivity_matrix()
print("Edge counts matrix created successfully")

# Analyze cluster interconnectivity
cluster_metrics = analyzer.analyze_cluster_interconnectivity()

# Convert cluster metrics to DataFrame for easier analysis
metrics_df = pd.DataFrame.from_dict(cluster_metrics, orient="index")
metrics_df.index.name = "cluster"
metrics_df = metrics_df.sort_index()

# Print summary statistics
analyzer.print_metrics_summary(metrics_df)

# Plot heatmap for all clusters
analyzer.plot_heatmap(edge_counts_df, "All")

# Plot heatmap for safety clusters
safety_clusters_df = edge_counts_df.loc[
    [f'Cluster {i}' for i in analyzer.safety_clusters],
    [f'Cluster {i}' for i in analyzer.safety_clusters]
]
analyzer.plot_heatmap(safety_clusters_df, "Safety", diag_excluded=True, cap_max=True)
