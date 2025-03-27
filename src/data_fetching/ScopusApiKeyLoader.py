import os
from typing import Dict, Any
from dotenv import load_dotenv


class ScopusApiKeyLoader:
    """
    Class to manage and load Scopus API keys.
    """

    @staticmethod
    def get_api_keys() -> Dict[str, Dict[str, Any]]:
        """
        Get Scopus API keys from environment variables.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of API keys with their names as keys and key info as values

        Raises:
            ValueError: If no valid API keys are found in environment variables
        """
        # Load environment variables from .env file
        load_dotenv()

        api_keys_config = {
            "api_key_A": {
                "key": os.getenv("SCOPUS_API_KEY_A"),
                "rate_limit": 40000,  # requests per week
                "description": "Primary API key with higher rate limit",
            },
            "api_key_B": {
                "key": os.getenv("SCOPUS_API_KEY_B"),
                "rate_limit": 10000,  # requests per week
                "description": "Secondary API key",
            },
            "api_key_deb": {
                "key": os.getenv("SCOPUS_API_KEY_DEB"),
                "rate_limit": 10000,  # requests per week
                "description": "Deb's API key",
            },
            "api_key_haoxin": {
                "key": os.getenv("SCOPUS_API_KEY_HAOXIN"),
                "rate_limit": 10000,  # requests per week
                "description": "Haoxin's API key",
            },
        }

        # Filter valid keys
        valid_api_keys = {}
        for key_name, info in api_keys_config.items():
            if info["key"] is not None:
                # Add key_name to the info dictionary
                info["key_name"] = key_name
                valid_api_keys[key_name] = info

        if not valid_api_keys:
            raise ValueError(
                "No valid API keys found in environment variables. Please check your .env file."
            )

        print("API Keys and Rate Limits:")
        for name, info in valid_api_keys.items():
            print(
                f"- {name}: {info['rate_limit']:,} requests per week ({info['description']})"
            )

        return valid_api_keys
