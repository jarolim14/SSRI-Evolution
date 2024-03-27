import matplotlib.pyplot as plt


class ClusterPlotter:
    def __init__(self, df, cluster_column):
        self.df = df
        self.cluster_column = cluster_column

    def plot_rolling_cluster_distribution(self, cluster_number_label_dict, window=2):
        df_sub = self.df[
            self.df[self.cluster_column].isin(cluster_number_label_dict.keys())
        ]

        # Set the size of the figure
        plt.figure(figsize=(15, 10))

        # Perform data manipulation
        cluster_distribution = (
            df_sub.groupby("year")[self.cluster_column]
            .value_counts(normalize=True)
            .unstack()
        )
        rolling_cluster_distribution = cluster_distribution.rolling(window).mean()

        # Plot the data
        plt.plot(rolling_cluster_distribution)

        # Legend with both cluster number and label
        plt.legend(
            labels=[
                f"{i}: {cluster_number_label_dict[i]}"
                for i in rolling_cluster_distribution.columns
            ],
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        plt.title("Cluster Distribution Over Time (Rolling Mean)")
        plt.xlabel("Year")
        plt.ylabel("Normalized Cluster Count")

        # Display the plot
        plt.show()


"""
# Assuming df_0025 is your DataFrame and 'cluster' is the name of the cluster column
# Initialize the class with your DataFrame and the column name
cluster_analysis = ClusterListPlotter(df_0025, "cluster")

# Plot the rolling cluster distribution for specified cluster numbers
cluster_analysis.plot_rolling_cluster_distribution(
    cluster_numbers_list=[0, 1, 2, 3, 4, 5, 6], window=2
)
"""
