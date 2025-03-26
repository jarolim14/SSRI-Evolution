import re
from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass

import pandas as pd
from unidecode import unidecode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScopusConfig:
    """Configuration for ScopusDataCleaner."""
    columns_to_drop: List[str] = None
    column_rename_map: Dict[str, str] = None
    publication_types_to_keep: List[str] = None
    publication_subtypes_to_remove: List[str] = None
    column_order: List[str] = None

    def __post_init__(self):
        if self.columns_to_drop is None:
            self.columns_to_drop = [
                "@_fa", "link", "prism:issn", "prism:volume", "prism:issueIdentifier",
                "prism:pageRange", "affiliation", "prism:eIssn", "prism:isbn",
                "prism:coverDisplayDate", "pii", "source-id"
            ]
        
        if self.column_rename_map is None:
            self.column_rename_map = {
                "prism:url": "api_url",
                "dc:identifier": "scopus_id",
                "eid": "eid",
                "dc:title": "title",
                "dc:creator": "first_author",
                "prism:publicationName": "journal",
                "prism:coverDate": "date",
                "dc:description": "abstract",
                "citedby-count": "citedby_count",
                "prism:aggregationType": "publication_type",
                "subtype": "publication_subtype",
                "subtypeDescription": "publication_subtype_description",
                "author-count": "author_count",
                "author": "authors_json",
                "authkeywords": "authkeywords",
                "fund-no": "funding_no",
                "openaccess": "openaccess",
                "openaccessFlag": "openaccess_flag",
                "prism:doi": "doi",
                "pubmed-id": "pubmed_id",
                "freetoread": "freetoread",
                "freetoreadLabel": "freetoread_label",
                "fund-acr": "fund_acr",
                "fund-sponsor": "fund_sponsor",
                "article-number": "article_number",
            }
        
        if self.publication_types_to_keep is None:
            self.publication_types_to_keep = ["Journal"]
        
        if self.publication_subtypes_to_remove is None:
            self.publication_subtypes_to_remove = ["Conference Paper"]
        
        if self.column_order is None:
            self.column_order = [
                "eid", "title", "date", "first_author", "abstract", "doi",
                "year", "auth_year", "unique_auth_year", "pubmed_id", "api_url",
                "scopus_id", "journal", "citedby_count", "publication_type",
                "publication_subtype", "publication_subtype_description",
                "author_count", "authors_json", "authkeywords", "funding_no",
                "openaccess", "openaccess_flag", "freetoread", "freetoread_label",
                "fund_acr", "fund_sponsor", "article_number"
            ]

class ScopusDataCleaner:
    """
    A class for cleaning and preprocessing Scopus publication data.
    
    This class provides methods to clean, transform, and validate Scopus publication data.
    It handles tasks such as column renaming, duplicate removal, date formatting,
    and author name processing.
    
    Attributes:
        df (pd.DataFrame): The input DataFrame containing Scopus data
        removal_log (Dict): A log of data cleaning operations and their effects
        config (ScopusConfig): Configuration settings for the cleaner
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[ScopusConfig] = None) -> None:
        """
        Initialize the ScopusDataCleaner.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing Scopus data
            config (Optional[ScopusConfig]): Configuration settings for the cleaner
            
        Raises:
            ValueError: If the input DataFrame is empty or missing required columns
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")
            
        required_columns = ["prism:coverDate", "prism:aggregationType", "subtypeDescription"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        self.config = config or ScopusConfig()
        self.df = df.sort_values(by="prism:coverDate").reset_index(drop=True)
        self.removal_log = {"initial_count": len(df)}
        logger.info(f"Initialized with {self.df.shape[0]} rows and {self.df.shape[1]} columns")

    def drop_columns(self) -> None:
        """Remove specified columns from the DataFrame."""
        try:
            self.df = self.df.drop(self.config.columns_to_drop, axis=1)
            self.removal_log["columns_dropped"] = len(self.config.columns_to_drop)
            logger.info(f"Dropped {len(self.config.columns_to_drop)} columns")
        except KeyError as e:
            logger.error(f"Error dropping columns: {e}")
            raise

    def rename_columns(self) -> None:
        """Rename columns according to the configuration mapping."""
        try:
            self.df = self.df.rename(self.config.column_rename_map, axis=1)
            logger.info("Columns renamed successfully")
        except KeyError as e:
            logger.error(f"Error renaming columns: {e}")
            raise

    def subset_publication_type(self) -> None:
        """Filter publications by type according to configuration."""
        try:
            vc = self.df["publication_type"].value_counts()
            removed_types = {k: v for k, v in vc.items() if k not in self.config.publication_types_to_keep}
            
            for pub_type, count in removed_types.items():
                logger.info(f"Removing {count} {pub_type} publications")
                self.removal_log[pub_type] = count
            
            self.df = self.df[self.df["publication_type"].isin(self.config.publication_types_to_keep)]
            logger.info(f"Remaining publications: {self.df.shape[0]}")
        except KeyError as e:
            logger.error(f"Error subsetting publication types: {e}")
            raise

    def subset_publication_subtype(self) -> None:
        """Remove specified publication subtypes."""
        try:
            vc = self.df["publication_subtype_description"].value_counts()
            for subtype in self.config.publication_subtypes_to_remove:
                if subtype in vc:
                    count = vc[subtype]
                    logger.info(f"Removing {count} {subtype} publications")
                    self.removal_log[subtype] = count
            
            self.df = self.df[~self.df["publication_subtype_description"].isin(self.config.publication_subtypes_to_remove)]
            logger.info(f"Remaining publications: {self.df.shape[0]}")
        except KeyError as e:
            logger.error(f"Error subsetting publication subtypes: {e}")
            raise

    def remove_duplicates(self, column: str = "eid") -> None:
        """
        Remove duplicates based on a specified column while preserving NaN values.
        
        Args:
            column (str): Column name to check for duplicates
            
        Raises:
            KeyError: If the specified column doesn't exist
        """
        try:
            len_before = len(self.df)
            
            # Handle non-NaN and NaN values separately
            nona_df = self.df[self.df[column].notna()].drop_duplicates(subset=[column])
            nan_df = self.df[self.df[column].isna()]
            
            self.df = pd.concat([nona_df, nan_df]).reset_index(drop=True)
            removed_count = len_before - len(self.df)
            
            logger.info(f"Removed {removed_count} duplicates based on {column}")
            self.removal_log[f"duplicates removed on {column}"] = removed_count
        except KeyError as e:
            logger.error(f"Error removing duplicates: {e}")
            raise

    def date_formater(self) -> None:
        """Format date columns and extract year."""
        try:
            self.df["date"] = pd.to_datetime(self.df["date"])
            self.df["year"] = self.df["date"].dt.year
            logger.info("Date columns formatted successfully")
        except Exception as e:
            logger.error(f"Error formatting dates: {e}")
            raise

    def unique_auth_year_col(self) -> None:
        """Create unique author-year combinations with proper ordering."""
        try:
            # Sort by author and year
            self.df = self.df.sort_values(by=["first_author", "year"]).reset_index(drop=True)
            
            # Create auth_year column
            self.df["auth_year"] = [
                self._remove_initials(name) + "_" + str(year)
                for name, year in zip(self.df["first_author"], self.df["year"])
            ]
            
            # Create unique identifiers
            nameyear_counts = {}
            unique_nameyear = []
            
            for nameyear in self.df["auth_year"]:
                nameyear_counts[nameyear] = nameyear_counts.get(nameyear, 0) + 1
                count = nameyear_counts[nameyear]
                unique_nameyear.append(f"{nameyear}_{count}" if count > 1 else nameyear)
            
            self.df["unique_auth_year"] = unique_nameyear
            self.df = self.df.sort_values(by=["year"]).reset_index(drop=True)
            
            logger.info("Created unique_auth_year column")
        except Exception as e:
            logger.error(f"Error creating unique author-year column: {e}")
            raise

    @staticmethod
    def _remove_initials(name: str) -> str:
        """
        Remove initials from author names and standardize format.
        
        Args:
            name (str): Author name to process
            
        Returns:
            str: Processed name without initials
        """
        if not isinstance(name, str) or not name:
            return "NoAuth"
            
        # Transliterate to ASCII
        name = unidecode(name)
        
        # Remove initials and standardize format
        name = re.sub(r"([A-Z]\.)+", "", name)
        name = name.title()
        name = re.sub(r"\s+", "", name)
        name = re.sub(r"[^\w\s]", "", name)
        
        return name

    def remove_missing_abstracts(self) -> None:
        """
        Remove rows where the abstract is missing or empty.
        Updates the removal log with the count of removed rows.
        """
        try:
            len_before = len(self.df)
            self.df = self.df[self.df["abstract"].notna() & (self.df["abstract"].str.strip() != "")]
            removed_count = len_before - len(self.df)
            
            logger.info(f"Removed {removed_count} rows with missing abstracts")
            self.removal_log["missing_abstracts_removed"] = removed_count
        except KeyError as e:
            logger.error(f"Error removing missing abstracts: {e}")
            raise

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the cleaned DataFrame with specified column order.
        
        Returns:
            pd.DataFrame: Cleaned and ordered DataFrame
        """
        try:
            self.removal_log["final_count"] = len(self.df)
            self.df = self.df[self.config.column_order]
            return self.df
        except KeyError as e:
            logger.error(f"Error ordering columns: {e}")
            raise

    def get_removal_log(self) -> Dict[str, Union[int, str]]:
        """
        Get the log of data cleaning operations.
        
        Returns:
            Dict[str, Union[int, str]]: Dictionary containing cleaning operation logs
        """
        self.removal_log["final_count"] = len(self.df)
        return self.removal_log
