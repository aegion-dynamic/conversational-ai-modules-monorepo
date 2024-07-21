import json
import io
import csv
from typing import List, Dict, Any

class SampleDataManager:
    """Manages the sample data used for analysis."""

    def __init__(self, csv_data: str):
        """
        Initialize the SampleDataManager with CSV data.

        Args:
            csv_data (str): CSV-formatted string containing sample data.
        """
        self.sample_data = self._load_sample_data(csv_data)

    def _load_sample_data(self, csv_data: str) -> List[Dict[str, Any]]:
        """
        Load sample data from CSV string into a list of dictionaries.

        Args:
            csv_data (str): CSV-formatted string containing sample data.

        Returns:
            List[Dict[str, Any]]: List of dictionaries representing the sample data.
        """
        try:
            csv_file = io.StringIO(csv_data.strip())
            reader = csv.DictReader(csv_file)
            return [row for row in reader]
        except csv.Error as e:
            print(f"Error loading CSV data: {e}")
            return []

    def get_sample_data(self) -> str:
        """
        Get the sample data as a JSON-formatted string.

        Returns:
            str: JSON-formatted string of the sample data.
        """
        try:
            return json.dumps(self.sample_data, indent=2)
        except json.JSONEncodeError as e:
            print(f"Error encoding sample data to JSON: {e}")
            return "[]"