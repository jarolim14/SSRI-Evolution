import ast

import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt


class BirthYearTimelinePlotter:
    def __init__(self, dataframe, style=1):
        self.dataframe = dataframe
        self.style = style
        self.color_column = "Color"
        self._assign_colors()

    def _assign_colors(self):
        """Assign colors to each drug based on its class."""
        self.dataframe[self.color_column] = self.dataframe.apply(
            lambda row: self._get_color_for_drug(row), axis=1
        )

    def _get_color_for_drug(self, row):
        """Get color for a drug based on its class."""
        color_palette = row["ColorPalette"]
        return f"({color_palette[:-1].lower()})"

    def _customize_timeline_appearance(self):
        plt.yticks([])
        plt.xlabel("Year")
        plt.title("Timeline of Birth Years and Drug Names")
        min_year, max_year = self.dataframe["BirthYear"].min(), self.dataframe["BirthYear"].max()
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
                self.dataframe["BirthYear"].unique(),
                labels=self.dataframe["BirthYear"].unique(),
                rotation=45,
            )
        plt.grid(which="major", axis="x", linestyle="-", linewidth=0.8, color="#CCCCCC")
        plt.grid(
            which="minor", axis="x", linestyle="--", linewidth=0.2, color="#CCCCCC"
        )
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))

    def _add_note(self, note_text):
        """Add a note to the plot."""
        y_min = plt.gca().get_ylim()[0]
        plt.text(
            self.dataframe["BirthYear"].min() - 2,
            y_min - 3,
            rf"$\mathit{{Note:}}$ {note_text}",
            fontsize=10,
            ha="left",
            va="top",
        )

    def plot_birth_year_timeline(self, save_path=None, show_note=True, note_text=None):
        """Plot the birth year timeline."""
        plt.figure(figsize=(20, 10))

        self._assign_colors()
        downshift = 0.2

        for _, row in self.dataframe.iterrows():
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

        self._customize_timeline_appearance()

        # Add a legend for ColorPalette and DrugClass (remove duplicates)
        legend_dataframe = self.dataframe[["ColorPalette", "DrugClass"]].drop_duplicates()

        patches_list = []
        for _, row in legend_dataframe.iterrows():
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


class BirthYearTimelineIntersectionPlotter:
    def __init__(self, dataframe, style=1):
        self.dataframe = dataframe
        self.style = style
        self._assign_colors()

    def _assign_colors(self):
        """Assign colors to each drug based on its class."""
        self.dataframe["Color_EMA"] = self.dataframe.apply(
            lambda row: self._get_color_for_drug(row, "EMA"), axis=1
        )
        self.dataframe["Color_FDA"] = self.dataframe.apply(
            lambda row: self._get_color_for_drug(row, "FDA"), axis=1
        )

    def _get_color_for_drug(self, row, agency):
        """Get color for a drug based on its class and agency."""
        color_palette = row[f"ColorPalette_{agency}"]
        return f"({color_palette[:-1].lower()})"

    def _customize_timeline_appearance(self):
        """Customize the appearance of the timeline plot."""
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xlabel("Year", fontsize=14)
        plt.ylabel("Drugs", fontsize=14)
        plt.title("Drug Birth Years by Agency", fontsize=16, pad=20)
        plt.xticks(rotation=45)
        plt.tight_layout()

    def _add_note(self, note_text):
        """Add a note to the plot."""
        y_min = plt.gca().get_ylim()[0]
        plt.text(
            self.dataframe["BirthYear"].min() - 2,
            y_min - 3,
            rf"$\mathit{{Note:}}$ {note_text}",
            fontsize=10,
            ha="left",
            va="top",
        )

    def plot_birth_year_timeline(
        self, figsize=(15, 8), save_path=None, show_note=True, note_text=None
    ):
        """Plot the birth year timeline with intersection data."""
        plt.figure(figsize=figsize)

        self._assign_colors()
        downshift = 0.2

        for index, row in self.dataframe.iterrows():
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

            same_year = row["BirthYear_EMA"] == row["BirthYear_FDA"]

            plt.scatter(
                row["BirthYear_FDA"],
                [downshift if not same_year else downshift - 0.5][0],
                marker="*",
                color=color_tuple,
                s=200,
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
        legend_dataframe = self.dataframe[["ColorPalette_EMA", "DrugClass_EMA"]].drop_duplicates()

        patches_list = []
        for _, row in legend_dataframe.iterrows():
            color_palette = row["ColorPalette_EMA"]
            color = color_palette[:-1].lower()
            drug_class = row["DrugClass_EMA"]

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
