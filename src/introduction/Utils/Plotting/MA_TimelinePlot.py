import ast

import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt


class BirthYearTimelinePlotter:
    def __init__(self, df, style=1):
        self.style = style
        self.df = df

    def _assign_colors(self):
        """
        Assign colors to data points based on a 'color' column in the DataFrame if it exists.

        Args:
            df (DataFrame): The DataFrame containing data to be plotted.

        Returns:
            colors (list): List of colors for data points.
        """
        if "Color" not in self.df.columns:
            self.color_column = "Color"
            self.df[self.color_column] = ["#31a8ca"] * len(self.df)
        else:
            self.color_column = "Color"

    def _customize_timeline_appearance(self):
        plt.yticks([])
        plt.xlabel("Year")
        plt.title("Timeline of Birth Years and Drug Names")
        min_year, max_year = self.df["BirthYear"].min(), self.df["BirthYear"].max()
        plt.xlim(min_year - 2, max_year + 5)
        if self.style == 1:
            plt.xticks(range(min_year, max_year + 1), rotation=45)
            plt.xticks(
                range(max(min_year - 2, 1960), max_year + 2, 5),
                labels=range(max(min_year - 2, 1960), max_year + 2, 5),
                rotation=45,
            )
        elif self.style == 2:
            plt.xticks(
                self.df["BirthYear"].unique(),
                labels=self.df["BirthYear"].unique(),
                rotation=45,
            )
        plt.grid(which="major", axis="x", linestyle="-", linewidth=0.8, color="#CCCCCC")
        plt.grid(
            which="minor", axis="x", linestyle="--", linewidth=0.2, color="#CCCCCC"
        )
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))

    def _add_note(self, note_text):
        y_min = plt.gca().get_ylim()[0]
        plt.text(
            self.df["BirthYear"].min() - 2,
            y_min - 3,
            rf"$\mathit{{Note:}}$ {note_text}",
            fontsize=10,
            ha="left",
            va="top",
        )

    def plot_birth_year_timeline(self, save_path=None, show_note=True, note_text=None):
        plt.figure(figsize=(20, 10))

        self._assign_colors()
        downshift = 0.2

        for _, row in self.df.iterrows():
            color_tuple = ast.literal_eval(row[self.color_column])
            plt.scatter(
                row["BirthYear"], downshift, marker="o", color=color_tuple, s=200
            )
            plt.text(
                row["BirthYear"] + 0.7,
                downshift - 0.02,
                row["DrugName"],
                fontsize=12,
                ha="left",
                rotation=0,
            )
            downshift -= 1.2

            # legend building
            color_palette = row["ColorPalette"]
            drug_class = row["DrugClass"]

        self._customize_timeline_appearance()

        # Add a legend for ColorPalette and DrugClass (remove duplicates)
        df_legend = self.df[["ColorPalette", "DrugClass"]].drop_duplicates()

        patches_list = []
        for _, row in df_legend.iterrows():
            color_palette = row["ColorPalette"]
            color = color_palette[:-1].lower()
            drug_class = row["DrugClass"]

            # Create a colored patch for the drug class
            patch = patches.Patch(color=color, label=drug_class)
            patches_list.append(patch)

        # Create the legend using the patches
        plt.legend(handles=patches_list, loc="upper right")

        if show_note and note_text:
            self._add_note(note_text)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


# example
# plotter = BirthYearTimelinePlotter(df=df_plotting, style=1)
# plotter.plot_birth_year_timeline(
#    save_path=None, show_note=True, note_text="This is a custom note."
# )


class BirthYearTimelinePlotter_Intersection:
    def __init__(self, df, style=1):
        self.style = style
        self.df = df

    def _assign_colors(self):
        """
        Assign colors to data points based on a 'color' column in the DataFrame if it exists.

        Args:
            df (DataFrame): The DataFrame containing data to be plotted.

        Returns:
            colors (list): List of colors for data points.
        """
        if "Color_EMA" not in self.df.columns:
            self.color_column = "Color"
            self.df[self.color_column] = ["#31a8ca"] * len(self.df)
        else:
            self.color_column = "Color"

    def _customize_timeline_appearance(self):
        plt.yticks([])
        plt.xlabel("Year")
        plt.title("Timeline of Birth Years and Drug Names")
        # limits
        min_year = min(
            self.df["BirthYear_EMA"].min(),
            self.df["BirthYear_FDA"].min(),
        )
        max_year = max(
            self.df["BirthYear_EMA"].max(),
            self.df["BirthYear_FDA"].max(),
        )

        plt.xlim(min_year - 2, max_year + 5)
        if self.style == 1:
            plt.xticks(range(min_year, max_year + 1), rotation=45)
            plt.xticks(
                range(max(min_year - 2, 1960), max_year + 2, 5),
                labels=range(max(min_year - 2, 1960), max_year + 2, 5),
                rotation=45,
            )
        elif self.style == 2:
            plt.xticks(
                self.df["BirthYear_EMA"].unique(),
                labels=self.df["BirthYear_EMA"].unique(),
                rotation=45,
            )
        plt.grid(which="major", axis="x", linestyle="-", linewidth=0.8, color="#CCCCCC")
        plt.grid(
            which="minor", axis="x", linestyle="--", linewidth=0.2, color="#CCCCCC"
        )
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))

    def _add_note(self, note_text):
        y_min = plt.gca().get_ylim()[0]
        plt.text(
            self.df["BirthYear"].min() - 2,
            y_min - 3,
            rf"$\mathit{{Note:}}$ {note_text}",
            fontsize=10,
            ha="left",
            va="top",
        )

    def plot_birth_year_timeline(
        self, figsize=(15, 8), save_path=None, show_note=True, note_text=None
    ):
        plt.figure(figsize=figsize)

        self._assign_colors()
        downshift = 0.2

        for index, row in self.df.iterrows():
            color_tuple = ast.literal_eval(row["Color_EMA"])
            plt.scatter(
                row["BirthYear_EMA"],
                downshift,
                marker="o",
                color=color_tuple,
                s=200,
                label=row["DrugName_EMA"],
            )

            plt.text(
                row["BirthYear_EMA"] + 0.7,
                downshift + 0.22,
                row["DrugName_EMA"],
                fontsize=12,
                ha="left",
                rotation=0,
            )

            sameyear = row["BirthYear_EMA"] == row["BirthYear_FDA"]

            plt.scatter(
                row["BirthYear_FDA"],
                [downshift if not sameyear else downshift - 0.5][0],
                marker="*",
                color=color_tuple,
                s=200,
                # label=row["DrugName"],
            )

            # thin line between the two
            plt.plot(
                [row["BirthYear_EMA"], row["BirthYear_FDA"]],
                [downshift, downshift],
                color=color_tuple,
                linewidth=0.4,
            )
            downshift -= 1.2

        self._customize_timeline_appearance()

        # Add a legend for ColorPalette and DrugClass (remove duplicates)
        df_legend = self.df[["ColorPalette_EMA", "DrugClass_EMA"]].drop_duplicates()

        patches_list = []
        for _, row in df_legend.iterrows():
            color_palette = row["ColorPalette_EMA"]
            color = color_palette[:-1].lower()
            drug_class = row["DrugClass_EMA"]

            # Create a colored patch for the drug class
            patch = patches.Patch(color=color, label=drug_class)
            patches_list.append(patch)

        # Create the legend using the patches
        # Now, add a marker to the legend
        marker_legend_ema = mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markersize=10,
            label="EMA",
        )
        marker_legend_fda = mlines.Line2D(
            [],
            [],
            color="black",
            marker="*",
            linestyle="None",
            markersize=10,
            label="FDA",
        )
        patches_list.append(marker_legend_ema)
        patches_list.append(marker_legend_fda)

        # Create the legend using the patches
        plt.legend(handles=patches_list, loc="upper right")

        if show_note and note_text:
            self._add_note(note_text)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
