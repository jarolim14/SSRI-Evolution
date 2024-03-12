import ast
import warnings

import matplotlib.pyplot as plt


class DrugUsePlotter:
    def __init__(
        self,
        df,
        year_col,
        measure_col,
        color_col,
        group_col,
        class_col,
        min_measure_value=None,
    ):
        if min_measure_value is not None:
            # Filter DataFrame to include only groups with at least min_measure_value of measure_col in any year
            valid_groups = df.groupby(group_col).apply(
                lambda group: group[measure_col].max() >= min_measure_value
            )
            valid_groups = valid_groups[valid_groups].index.tolist()
            df = df[df[group_col].isin(valid_groups)]

        self.df = df
        self.year_col = year_col
        self.measure_col = measure_col
        self.color_col = color_col
        self.group_col = group_col
        self.class_col = class_col

        self.y_min = df[measure_col].min()
        self.y_max = df[measure_col].max()

    def create_subplot(self, ax, group_name, group_df):
        colors = [ast.literal_eval(color) for color in group_df[self.color_col]]
        MARKERS = [
            "o",
            ",",
            "s",
            "+",
            "D",
            "p",
            "*",
            "h",
            "H",
            "+",
            "x",
            "d",
            "|",
            "_",
            "v",
            "^",
            "<",
            ">",
        ]
        unique_groups = group_df[self.group_col].unique()

        group_markers = {
            group: MARKERS[i % len(MARKERS)] for i, group in enumerate(unique_groups)
        }
        group_colors = {
            group: colors[i % len(colors)] for i, group in enumerate(unique_groups)
        }

        for group in unique_groups:
            subset = group_df[group_df[self.group_col] == group]
            ax.plot(
                subset[self.year_col],
                subset[self.measure_col],
                marker=group_markers[group],
                color=group_colors[group],
                label=group,
                linestyle="-",
            )

        ax.set_ylim(self.y_min, self.y_max)
        ax.set_title(f"{self.class_col}: {group_name}")

    def plot(self, separate_plots=True, save_path=None):
        warnings.simplefilter(action="ignore", category=FutureWarning)
        grouped = self.df.groupby(self.class_col)

        if separate_plots:
            num_rows = len(grouped)
            fig, axes = plt.subplots(num_rows, 1, figsize=(8, 6 * num_rows))

            for idx, (group_name, group_df) in enumerate(grouped):
                ax = axes[
                    idx if num_rows > 1 else 0
                ]  # Handle case where there's only one group
                self.create_subplot(ax, group_name, group_df)
                ax.grid(True, linestyle="--", alpha=0.6, color="lightgrey")
                ax.set_xlabel(self.year_col if idx == num_rows - 1 else "")
                ax.legend(title=self.group_col, loc="upper left")

            fig.text(
                -0.014,
                0.5,
                f"Number of {self.measure_col}",
                va="center",
                rotation="vertical",
            )

        else:
            fig, ax = plt.subplots(figsize=(15, 6))
            for group_name, group_df in grouped:
                self.create_subplot(ax, group_name, group_df)
            ax.grid(True, linestyle="--", alpha=0.6, color="lightgrey")
            ax.set_title("Combined Plot")
            ax.set_xlabel(self.year_col)
            ax.set_ylabel(f"Number of {self.measure_col}")
            ax.legend(title=self.group_col, loc="upper left")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
