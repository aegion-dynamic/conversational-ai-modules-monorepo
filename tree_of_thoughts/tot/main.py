import logging
import json
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv


# Import the TreeOfThoughtsFramework class defined in tree_of_thoughts.py
from TreeOfThoughtsFramework import TreeOfThoughtsFramework

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)

class TreeOfThoughtsExecutor:
    """
    Executes the Tree of Thoughts approach for solving user queries.

    This class orchestrates the entire process of analyzing user input,
    generating thoughts, evaluating states, and producing a final output.

    Attributes:
        sample_csv_data (Optional[str]): CSV-formatted string containing sample data.
        api_key (str): OpenAI API key for accessing the language model.
        tot_framework (TreeOfThoughtsFramework): The Tree of Thoughts framework.
    """

    def __init__(self, api_key: str, sample_csv_data: Optional[str] = None):
        """
        Initialize the TreeOfThoughtsExecutor with an optional sample data and API key.

        Args:
            api_key (str): OpenAI API key for accessing the language model.
            sample_csv_data (Optional[str]): CSV-formatted string containing sample data. Default is None.

        Raises:
            ValueError: If api_key is invalid.
        """
        if not api_key:
            raise ValueError("api_key must not be empty")

        self.sample_csv_data = sample_csv_data
        self.api_key = api_key
        self.tot_framework = TreeOfThoughtsFramework(api_key=self.api_key)
        logging.info("TreeOfThoughtsExecutor initialized successfully.")

    def execute(self, user_query: str, num_thoughts: int = 3, max_steps: int = 3, best_states_count: int = 2) -> Dict[str, Any]:
        """
        Execute the Tree of Thoughts approach using the provided user query.

        This method orchestrates the entire process of analyzing the user query,
        generating thoughts, evaluating states, and producing a final output.

        Args:
            user_query (str): The user's input query.
            num_thoughts (int): Number of thoughts to generate at each step. Default is 3.
            max_steps (int): Maximum number of thinking steps. Default is 3.
            best_states_count (int): Number of best states to keep at each step. Default is 2.

        Returns:
            Dict[str, Any]: The final output as a dictionary containing analysis results.

        Raises:
            ValueError: If user_query is empty.
            TreeOfThoughtsError: If an error occurs during the execution process.
        """
        if not user_query:
            raise ValueError("user_query must not be empty")

        logging.info(f"Starting execution process for query: {user_query}")
        try:
            result = self.tot_framework.solve(problem=user_query, k=num_thoughts, T=max_steps, b=best_states_count)
            best_state, best_path = result
            parsed_solution = self.tot_framework.parse_solution(best_state, best_path)
            output = self.tot_framework.generate_output(parsed_solution)
            self.tot_framework.save_json(parsed_solution, filename="solution.json")
            logging.info("Completed the problem-solving process.")
            return parsed_solution
        except Exception as e:
            logging.error(f"Error during execution: {e}")
            raise TreeOfThoughtsError(f"An error occurred during execution: {e}")

class TreeOfThoughtsError(Exception):
    """Custom exception class for Tree of Thoughts errors."""
    pass

# # Example usage
# if __name__ == "__main__":
#     user_query = "explain me the complete lifecycle of mlops"
#     load_dotenv()
#     api_key = os.getenv("OPENAI_API_KEY") # Replace with your actual API key

#     try:
#         executor = TreeOfThoughtsExecutor(api_key=api_key)
#         output = executor.execute(user_query=user_query, num_thoughts=3, max_steps=3, best_states_count=2)
#         print(json.dumps(output, indent=2))
#     except TreeOfThoughtsError as e:
#         print(f"An error occurred: {e}")
#     except ValueError as e:
#         print(f"Invalid input: {e}")
