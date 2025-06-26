import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class StaticProportionsPlotter:
    """
    A class to create static visualizations of cluster proportions over time.

    Features:
    - Plot with or without rolling means
    - Optimized color selection for better visual distinction
    - Cluster merging functionality
    - Flexible legend placement
    - Customizable styling options
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cluster_label_dict: Dict[str, str],
        cluster_col: str = "cluster_alpha0.3_k10_res0.002",
        year_col: str = "year",
        font_name: str = "Arial",
        font_size: int = 26,
        cluster_mergers: Optional[List[Tuple[str, str]]] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the plotter with the necessary data.

        Args:
            df: DataFrame containing the cluster data
            cluster_label_dict: Dictionary mapping cluster IDs to readable labels
            cluster_col: Column name for the cluster assignments
            year_col: Column name for the year data
            font_name: Font family to use for the plot
            font_size: Base font size for the plot
            cluster_mergers: List of tuples (target_cluster, source_cluster) to merge
            output_dir: Directory to save output figures (None for no default directory)
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Store initial parameters
        self.df = df.copy()
        self.cluster_label_dict = cluster_label_dict.copy()
        self.cluster_col = cluster_col
        self.year_col = year_col
        self.font_name = font_name
        self.font_size = font_size
        self.output_dir = Path(output_dir) if output_dir else None

        # Ensure cluster column is string type for consistent comparisons
        if not pd.api.types.is_string_dtype(self.df[self.cluster_col]):
            self.df[self.cluster_col] = self.df[self.cluster_col].apply(
                lambda x: (
                    str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)
                )
            )
            self.logger.info(
                f"Converted {self.cluster_col} to string type for consistent comparisons"
            )

        # Merge clusters if needed
        if cluster_mergers:
            self.merge_clusters(cluster_mergers)

        # Initialize cluster counts as None (will be calculated when needed)
        self.cluster_counts = None

        # Verify font availability
        self._check_font_availability()

    def _check_font_availability(self):
        """Check if the specified font is available, otherwise use a fallback."""
        available_fonts = set(f.name for f in fm.fontManager.ttflist)
        if self.font_name not in available_fonts:
            self.logger.warning(
                f"Font '{self.font_name}' not found. Using 'sans-serif' instead."
            )
            self.font_name = "sans-serif"

    def merge_clusters(self, cluster_mergers: List[Tuple[str, str]]):
        """
        Merge clusters based on the specified list of tuples.

        Args:
            cluster_mergers: List of tuples (target_cluster, source_cluster)
                             where source_cluster will be replaced by target_cluster
        """
        for target, source in cluster_mergers:
            # Ensure both are strings for consistent comparison
            target_str = str(target)
            source_str = str(source)

            # Replace in DataFrame
            self.df[self.cluster_col] = self.df[self.cluster_col].replace(
                {source_str: target_str}
            )

            # Update label dictionary if needed
            if (
                source_str in self.cluster_label_dict
                and target_str not in self.cluster_label_dict
            ):
                self.cluster_label_dict[target_str] = self.cluster_label_dict[
                    source_str
                ]

        # Reset cluster counts since data has changed
        self.cluster_counts = None
        self.logger.info(f"Merged {len(cluster_mergers)} cluster pairs")

    def _calculate_cluster_proportions_per_year(
        self, year_range: Optional[Tuple[int, int]] = None
    ):
        """
        Calculate counts and proportions of papers per cluster per year.

        Args:
            year_range: Optional tuple of (start_year, end_year) to filter the data
        """
        # Filter by year range if specified
        filtered_df = self.df
        if year_range:
            filtered_df = filtered_df[
                (filtered_df[self.year_col] >= year_range[0])
                & (filtered_df[self.year_col] <= year_range[1])
            ]

        # Group by year and cluster to get counts
        cluster_counts = (
            filtered_df.groupby([self.year_col, self.cluster_col])
            .size()
            .reset_index(name="count")
        )

        # Calculate yearly proportions
        cluster_counts["proportion"] = cluster_counts.groupby(self.year_col)[
            "count"
        ].transform(lambda x: x / x.sum())

        self.cluster_counts = cluster_counts
        return cluster_counts

    def _calculate_subset_cluster_proportions_per_year(
        self, clusters_to_plot: List[str]
    ):
        """
        Calculate proportions for a subset of clusters relative to that subset's total.

        Args:
            clusters_to_plot: List of cluster IDs to include

        Returns:
            DataFrame with proportions calculated relative to the subset total
        """
        # Ensure all cluster IDs are strings
        clusters_as_str = [str(c) for c in clusters_to_plot]

        # Filter to only the requested clusters
        filtered_counts = self.cluster_counts[
            self.cluster_counts[self.cluster_col].isin(clusters_as_str)
        ]

        if filtered_counts.empty:
            available_clusters = sorted(self.cluster_counts[self.cluster_col].unique())
            raise ValueError(
                f"No data found for specified clusters. Available clusters: {available_clusters}"
            )

        # Recalculate proportions based on the subset total
        filtered_counts["proportion"] = filtered_counts.groupby(self.year_col)[
            "count"
        ].transform(lambda x: x / x.sum())

        return filtered_counts

    def _set_x_axis_ticks_and_limits(self, ax, years, rotation=45):
        """
        Set x-axis ticks and limits for better visualization.

        Args:
            ax: Matplotlib axis object
            years: Series or array of years in the data
            rotation: Rotation angle for x-tick labels
        """
        min_year = min(years)
        max_year = max(years)

        # Calculate an appropriate tick interval
        year_range = max_year - min_year
        if year_range <= 10:
            tick_interval = 1
        elif year_range <= 30:
            tick_interval = 5
        else:
            tick_interval = 10

        # Generate tick positions
        start_tick = (min_year // tick_interval) * tick_interval
        end_tick = ((max_year // tick_interval) + 1) * tick_interval
        ticks = np.arange(start_tick, end_tick + 1, tick_interval)

        # Ensure the min and max years are included
        if min_year not in ticks:
            ticks = np.insert(ticks, 0, min_year)
        if max_year not in ticks:
            ticks = np.append(ticks, max_year)

        # Set ticks and limits
        ax.set_xticks(ticks)
        buffer = year_range * 0.02  # Add 2% buffer on each side
        ax.set_xlim(min_year - buffer, max_year + buffer)

        # Rotate x labels if specified
        plt.xticks(rotation=rotation)

    def _generate_distinct_colors(self, num_colors: int) -> List:
        """
        Generate a list of perceptually distinct colors.

        Uses the HSV color space with evenly spaced hues and carefully selected
        saturation and value to ensure maximum visual distinction.

        Args:
            num_colors: Number of distinct colors to generate

        Returns:
            List of RGB colors as tuples
        """
        colors = []

        # Use golden ratio to generate well-spaced hues
        golden_ratio_conjugate = 0.618033988749895

        # Start at a random point (but fixed for reproducibility)
        h = 0.1  # Starting hue

        # Generate colors with varying hue but consistent saturation and value
        for i in range(num_colors):
            # Shift the hue by the golden ratio
            h += golden_ratio_conjugate
            h %= 1.0

            # HSV (hue, saturation, value) - vary saturation and value slightly
            # to enhance distinguishability
            if i % 3 == 0:
                # Use high saturation and value
                s = 0.8
                v = 0.9
            elif i % 3 == 1:
                # Use high saturation but lower value (darker)
                s = 0.85
                v = 0.7
            else:
                # Use lower saturation but high value (lighter, pastel-like)
                s = 0.65
                v = 0.85

            # Convert HSV to RGB
            rgb = self._hsv_to_rgb(h, s, v)
            colors.append(rgb)

        return colors

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """
        Convert HSV color to RGB.

        Args:
            h: Hue (0-1)
            s: Saturation (0-1)
            v: Value (0-1)

        Returns:
            RGB tuple with values in range 0-1
        """
        if s == 0.0:
            return (v, v, v)

        i = int(h * 6)  # Sector 0 to 5
        f = (h * 6) - i  # Factorial part of h
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        if i % 6 == 0:
            return (v, t, p)
        elif i % 6 == 1:
            return (q, v, p)
        elif i % 6 == 2:
            return (p, v, t)
        elif i % 6 == 3:
            return (p, q, v)
        elif i % 6 == 4:
            return (t, p, v)
        else:
            return (v, p, q)

    def _get_optimized_colors(
        self, num_colors: int, color_palette: Optional[str] = None
    ) -> List:
        """
        Get optimized colors for the plot.

        Args:
            num_colors: Number of colors needed
            color_palette: Optional matplotlib colormap name

        Returns:
            List of RGB color tuples
        """
        if color_palette and hasattr(plt.cm, color_palette):
            # Use the specified colormap
            cmap = getattr(plt.cm, color_palette)
            return [cmap(i / max(1, num_colors - 1)) for i in range(num_colors)]

        # Use qualitative colormaps for better distinction
        if num_colors <= 9:
            return plt.cm.Set1.colors[:num_colors]
        elif num_colors <= 12:
            return plt.cm.Paired.colors[:num_colors]
        elif num_colors <= 20:
            return plt.cm.tab20.colors[:num_colors]
        else:
            # For larger sets, generate algorithmically
            return self._generate_distinct_colors(num_colors)

    def plot_cluster_proportions(
        self,
        clusters_to_plot: List[Union[str, int, float]],
        rolling_window: Optional[int] = None,
        proportions_total: Literal["total", "subset"] = "total",
        figsize: Tuple[int, int] = (20, 8),
        color_palette: Optional[str] = None,
        title: Optional[str] = None,
        legend_loc: str = "right",
        legend_ncol: int = 1,
        year_range: Optional[Tuple[int, int]] = None,
        x_rotation: int = 45,
        show_grid: bool = True,
        save: bool = False,
        filename: Optional[str] = None,
        dpi: int = 300,
        format: str = "png",
        background_alpha: float = 0.5,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot cluster proportions over time with optional rolling means.

        Args:
            clusters_to_plot: List of cluster IDs to include in the plot
            rolling_window: Size of the rolling window for smoothing (None for no smoothing)
            proportions_total: Whether to calculate proportions relative to "total" data or "subset" only
            figsize: Size of the figure as (width, height) in inches
            color_palette: Name of a matplotlib colormap
            title: Custom title for the plot
            legend_loc: Location for the legend ('right', 'bottom', 'inside')
            legend_ncol: Number of columns in the legend
            year_range: Optional tuple of (start_year, end_year) to filter the data
            x_rotation: Rotation angle for x-axis labels
            show_grid: Whether to show grid lines
            save: Whether to save the figure
            filename: Custom filename for the saved figure
            dpi: Resolution in dots per inch for saved figure
            format: File format for saved figure
            background_alpha: Alpha transparency for background stackplot (if using rolling means)

        Returns:
            Tuple of (Figure, Axes) objects
        """
        # Completely isolate this figure from any previous state
        plt.close("all")  # Close all existing figures
        plt.rcParams.update(plt.rcParamsDefault)  # Reset all rcParams to default

        # Create a completely new figure with a clear backend state
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)  # Create a single axis

        # Ensure clusters are strings
        clusters_to_plot = [str(cluster) for cluster in clusters_to_plot]

        # Calculate proportions if not already done
        if self.cluster_counts is None:
            self._calculate_cluster_proportions_per_year(year_range)

        # Determine which proportions to use based on mode
        if proportions_total == "total":
            filtered_cluster_counts = self.cluster_counts.copy()
            yaxis_label = "Proportion (Overall)"
        else:  # "subset"
            filtered_cluster_counts = (
                self._calculate_subset_cluster_proportions_per_year(clusters_to_plot)
            )
            yaxis_label = "Proportion (Selected Clusters)"

        # Pivot to get clusters as columns
        pivot_df = filtered_cluster_counts.pivot(
            index=self.year_col, columns=self.cluster_col, values="proportion"
        ).fillna(0)

        # Filter to only include requested clusters and handle missing
        available_clusters = [c for c in clusters_to_plot if c in pivot_df.columns]
        if not available_clusters:
            raise ValueError(
                f"None of the requested clusters found in data. Available: {list(pivot_df.columns)}"
            )

        pivot_df = pivot_df[available_clusters]

        # Calculate rolling means if requested
        if rolling_window and rolling_window > 1:
            pivot_df_rolling = pivot_df.rolling(
                window=rolling_window, min_periods=1
            ).mean()
        else:
            pivot_df_rolling = pivot_df  # No rolling mean

        # Get human-readable labels
        cluster_labels = [
            self.cluster_label_dict.get(cluster, f"Cluster {cluster}")
            for cluster in available_clusters
        ]

        # Get optimized colors
        colors = self._get_optimized_colors(len(available_clusters), color_palette)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Convert all of the data needed for the plot to ensure no shared state
        pivot_df_data = pivot_df.copy()

        if rolling_window and rolling_window > 1:
            pivot_df_rolling_data = pivot_df_rolling.copy()
        else:
            pivot_df_rolling_data = pivot_df_data  # No rolling mean

        # Make local copies of all data to ensure complete isolation
        available_clusters_local = list(available_clusters)
        cluster_labels_local = list(cluster_labels)
        colors_local = list(colors)

        # Plot the data - different approaches for with/without rolling means
        if rolling_window and rolling_window > 1:
            # For rolling means, show both raw and smoothed data
            # Background stackplot shows raw data with transparency
            ax.stackplot(
                pivot_df_data.index,
                [pivot_df_data[col] for col in pivot_df_data.columns],
                labels=[],  # No labels for background
                colors=colors_local,
                alpha=background_alpha,
            )

            # Foreground stackplot shows smoothed data
            ax.stackplot(
                pivot_df_rolling_data.index,
                [pivot_df_rolling_data[col] for col in pivot_df_rolling_data.columns],
                labels=cluster_labels_local,
                colors=colors_local,
                alpha=1.0,  # Full opacity
            )

            if title is None:
                title = f"Cluster Proportions Over Time ({rolling_window}-Year Rolling Mean)"
        else:
            # Standard stackplot for non-smoothed data
            ax.stackplot(
                pivot_df_data.index,
                [pivot_df_data[col] for col in pivot_df_data.columns],
                labels=cluster_labels_local,
                colors=colors_local,
            )

            if title is None:
                title = "Cluster Proportions Over Time"

        # Set title if provided
        if title:
            ax.set_title(
                title,
                fontsize=self.font_size + 2,
                fontname=self.font_name,
                pad=20,
            )

        # Set axis labels
        ax.set_ylabel(yaxis_label, fontsize=self.font_size, fontname=self.font_name)
        ax.set_xlabel("Year", fontsize=self.font_size, fontname=self.font_name)

        # Configure x-axis
        self._set_x_axis_ticks_and_limits(ax, pivot_df.index, rotation=x_rotation)

        # Add grid if requested
        if show_grid:
            ax.grid(True, linestyle="--", alpha=0.7)

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

        # Position the legend based on specified location
        if legend_loc == "right":
            legend = ax.legend(
                title="Legend",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                fontsize=self.font_size - 4,
                title_fontsize=self.font_size,
                ncol=legend_ncol,
                frameon=True,
                fancybox=True,
            )
            # Adjust layout for right legend
            plt.tight_layout()
            plt.subplots_adjust(right=0.8 if len(available_clusters) > 3 else 0.85)

        elif legend_loc == "bottom":
            legend = ax.legend(
                title="Legend",
                bbox_to_anchor=(0.5, -0.15),
                loc="upper center",
                fontsize=self.font_size - 4,
                title_fontsize=self.font_size,
                ncol=legend_ncol or min(3, len(available_clusters)),
                frameon=True,
                fancybox=True,
            )
            # Adjust layout for bottom legend
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2 if len(available_clusters) > 3 else 0.15)

        else:  # "inside" or any other value
            legend = ax.legend(
                title="Legend",
                loc="best",
                fontsize=self.font_size - 4,
                title_fontsize=self.font_size,
                ncol=legend_ncol,
                frameon=True,
                fancybox=True,
                facecolor="white",
                alpha=0.8,
            )
            # Use standard tight layout
            plt.tight_layout()

        # Set font for tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(self.font_name)
            label.set_fontsize(self.font_size - 4)

        # Save the figure if requested
        if save:
            save_path = self._get_save_path(filename, format)
            try:
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=format)
                self.logger.info(f"Figure saved to {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save figure: {e}")

        return fig, ax

    def _get_save_path(
        self, filename: Optional[str] = None, format: str = "png"
    ) -> str:
        """
        Get the full path for saving a figure.

        Args:
            filename: Optional custom filename
            format: File format extension

        Returns:
            Path to save the figure
        """
        if filename and Path(filename).is_absolute():
            # If an absolute path is provided, use it directly
            return filename

        if self.output_dir is None:
            if filename is None:
                # Default to current directory with timestamp
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                return f"cluster_proportions_{timestamp}.{format}"
            # Use filename relative to current directory
            return filename

        # Make sure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate default filename if none provided
        if filename is None:
            # Get the max year from the data for the filename
            if hasattr(self, "cluster_counts") and self.cluster_counts is not None:
                max_year = self.cluster_counts[self.year_col].max()
                filename = f"cluster_proportions_{max_year}"
            else:
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cluster_proportions_{timestamp}"

        # Add extension if not present
        if not filename.lower().endswith(f".{format.lower()}"):
            filename = f"{filename}.{format.lower()}"

        # Return full path
        return str(self.output_dir / filename)
