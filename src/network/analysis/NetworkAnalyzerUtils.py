import glob

import igraph as ig
import pandas as pd


class NetworkAnalyzerUtils:
    def __init__(self) -> None:
        pass

    def load_graph_files(self):
        """
        Load graph files and return a dictionary of graphs and a DataFrame.

        Returns:
            params_graph_dict (dict): A dictionary containing graph objects, where the keys are the graph names.
            df (DataFrame): A DataFrame containing embeddings.

        """
        graph_files = glob.glob(
            "../data/05-graphs/weighted-knn-citation-graph/*.graphml"
        )
        graph_files.sort()

        params_graph_dict = {
            f.split("/")[-1]
            .replace("weighted_", "")
            .replace("_knn_citation.graphml", ""): ig.Graph.Read_GraphML(f)
            for f in graph_files
        }

        df = pd.read_pickle("../data/04-embeddings/df_with_specter2_embeddings.pkl")

        return params_graph_dict, df

    def params_excel_saver_with_hyperlinks(
        self, filepath, overall_summary_df, explorer_sheets_dict
    ):
        # create summary sheet
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:  # type: ignore
            # Write the summary sheet
            overall_summary_df.to_excel(writer, sheet_name="summary", index=False)

            # Access the workbook through the writer.book attribute after writing
            workbook = writer.book

            sorted_sheet_names = sorted(explorer_sheets_dict.keys())

            for idx, sheet_name in enumerate(sorted_sheet_names):
                sheet_data = explorer_sheets_dict[sheet_name]

                row = overall_summary_df.iloc[idx]
                sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)

                current_sheet = workbook[sheet_name]

                # find max row and add 5
                lastrow = current_sheet.max_row

                # Add a hyperlink in A20 to go back to the summary sheet
                current_sheet.cell(row=lastrow + 7, column=1).hyperlink = "#summary!A1"
                current_sheet.cell(row=lastrow + 7, column=1).value = "Back to Summary"
                current_sheet.cell(row=lastrow + 7, column=1).style = "Hyperlink"

            # Update the summary sheet with hyperlinks to each cluster sheet
            summary_sheet = workbook["summary"]
            for row in range(2, summary_sheet.max_row + 1):
                sheetname = f"{summary_sheet.cell(row=row, column=1).value}_res{summary_sheet.cell(row=row, column=2).value}"
                summary_sheet.cell(row=row, column=1).hyperlink = f"#'{sheetname}'!A1"
                summary_sheet.cell(row=row, column=1).style = "Hyperlink"
            # print(f"File saved to {filepath}")

    def clusters_excel_saver_with_hyperlinks(
        self, filepath, df_summary, cluster_titles_sheets_dict
    ):
        """
        Saves summary DataFrame and combined sheets to an Excel file,
        with hyperlinks from the combined sheets back to the summary sheet,
        and from the summary sheet to each combined sheet.

        Parameters:
        - filepath: The path to the Excel file to save.
        - df_summary: The summary DataFrame to save to the 'summary' sheet.
        - cluster_titles_sheets_dict: A dict of DataFrames to save to individual sheets, with hyperlinks back to the summary.
        """
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Write the summary sheet
            df_summary.to_excel(writer, sheet_name="summary", index=False)

            # Access the workbook through the writer.book attribute after writing
            workbook = writer.book

            # Iterate through cluster_titles_sheets_dict and write each to a new sheet
            for cluster, sheet in sorted(cluster_titles_sheets_dict.items(), key=lambda x: int(x[0])):
                sheet_name = f"cluster_{cluster}"
                sheet.to_excel(writer, sheet_name=sheet_name, index=False)

                # Access the current sheet
                current_sheet = workbook[sheet_name]

                # find max row and add 5
                lastrow = current_sheet.max_row

                # Add a hyperlink in A20 to go back to the summary sheet
                current_sheet.cell(row=lastrow + 7, column=1).hyperlink = "#summary!A1"
                current_sheet.cell(row=lastrow + 7, column=1).value = "Back to Summary"
                current_sheet.cell(row=lastrow + 7, column=1).style = "Hyperlink"
                # Add nr pubs info
                current_sheet.cell(row=lastrow + 5, column=1).value = (
                    f"NrPubs {df_summary[df_summary['Cluster'] == cluster]['Nr of Pubs'].values[0]}"
                )
                # add time info
                current_sheet.cell(row=lastrow + 3, column=1).value = (
                    f"Year(Q1, Median, Q3) Q1={df_summary[df_summary['Cluster'] == cluster]['25th Percentile Year'].values[0]} Median={df_summary[df_summary['Cluster'] == cluster]['Median Year'].values[0]} Q3={df_summary[df_summary['Cluster'] == cluster]['75th Percentile Year'].values[0]}"
                )

            # Update the summary sheet with hyperlinks to each cluster sheet
            summary_sheet = workbook["summary"]
            for row in range(2, summary_sheet.max_row + 1):
                cluster_num = summary_sheet.cell(row=row, column=1).value
                summary_sheet.cell(row=row, column=1).hyperlink = (
                    f"#'cluster_{cluster_num}'!A1"
                )
                summary_sheet.cell(row=row, column=1).style = "Hyperlink"

    def cluster_coherence_analyzer_in_txt(
        self,
        analysis_data_dict,
        print_n_random_titles,
        word1,
        word2,
        print_word_string=True,
        save_to_file=True,
        file_name="output/tables/cluster-explorer/ParamsTest/cluster_analysis_output.txt",
        include_all_params_status=True,  # New argument to control the inclusion of all params status
    ):
        """
        Analyzes clusters for coherence based on the presence of specific words, prints details about matching clusters,
        highlights if multiple hits are found within the same parameters, and optionally saves the output to a text file.

        Args:
        - analysis_data_dict (dict): Dictionary containing analysis data with keys as parameters (alpha, k, res) and values
                                    as a list containing the summary DataFrame and a list of DataFrames for each cluster.
        - print_n_random_titles (int): Number of random titles to print from matching clusters.
        - word1 (str), word2 (str): Words to search for within the cluster's topic words.
        - print_word_string (bool): Whether to print the concatenated string of topic words.
        - save_to_file (bool): Whether to save the printed output to a text file.
        - file_name (str): Name of the file to save the output to, if save_to_file is True.
        """
        no_hit_params = []
        output = []  # List to capture the output
        output.append(f"Words: {word1.upper()} {word2.upper()}\n")

        for params, (summary, titles_df_list) in analysis_data_dict.items():
            hits_per_param = 0  # Counter for hits within the same parameters
            nr_of_clusters = len(titles_df_list)

            for i, row in summary.iterrows():
                Nr_of_Pubs = row["Nr of Pubs"]
                row_words_str = " ".join([str(value) for value in row[4:]])
                if word1 in row_words_str and word2 in row_words_str:
                    if hits_per_param == 0:
                        output.append("=" * 130)
                    hits_per_param += 1  # Increment the hit counter
                    output.append(
                        f"Found in {params};    Total Clusters: {nr_of_clusters};     Cluster: {row[0]};      Nr of Pubs in Cluster: {Nr_of_Pubs}"
                    )
                    if print_word_string:
                        output.append(row_words_str)
                    if print_n_random_titles != 0:
                        output.append("RANDOM TITLES")
                        titles = titles_df_list[i]["title_random"].head(
                            print_n_random_titles
                        )
                        output.extend([tit for tit in titles])
                    if hits_per_param > 0:
                        output.append("- " * 65)

            if (
                hits_per_param > 1
            ):  # Check if there were multiple hits for the parameters
                output.append(f"** Multiple hits found in parameters: {params} **\n")
            elif hits_per_param == 0 and include_all_params_status:
                no_hit_params.append(params)

        output.append(f"No hits found for Parameters: \n{set(no_hit_params)}\n")

        # Convert list to string
        output_str = "\n".join(output)

        # Print the output to console
        print(output_str)

        # Optionally save to file
        if save_to_file:
            with open(file_name, "w") as file:
                file.write(output_str)
