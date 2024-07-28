import logging
import json
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
# Import the classes defined in tree_of_thoughts.py
from tree_of_thoughts import SampleDataManager, IntentClassifier, ThoughtGenerator, StateEvaluator, TreeOfThoughts

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)

class TreeOfThoughtsExecutor:
    """
    Executes the Tree of Thoughts approach for solving user queries.

    This class orchestrates the entire process of analyzing user input,
    generating thoughts, evaluating states, and producing a final output.

    Attributes:
        sample_csv_data (str): CSV-formatted string containing sample data.
        api_key (str): OpenAI API key for accessing the language model.
        sample_data_manager (SampleDataManager): Manages the sample data.
        intent_classifier (IntentClassifier): Classifies user intent.
        thought_generator (ThoughtGenerator): Generates thoughts based on current state.
        state_evaluator (StateEvaluator): Evaluates the relevance of states.
        tree_of_thoughts (TreeOfThoughts): Implements the Tree of Thoughts algorithm.
    """

    def __init__(self, sample_csv_data: str, api_key: str):
        """
        Initialize the TreeOfThoughtsExecutor with sample data and API key.

        Args:
            sample_csv_data (str): CSV-formatted string containing sample data.
            api_key (str): OpenAI API key for accessing the language model.

        Raises:
            ValueError: If sample_csv_data is empty or api_key is invalid.
        """
        if not sample_csv_data or not api_key:
            raise ValueError("sample_csv_data and api_key must not be empty")

        self.sample_csv_data = sample_csv_data
        self.api_key = api_key
        self.sample_data_manager = SampleDataManager(csv_data=self.sample_csv_data)
        self.intent_classifier = IntentClassifier(api_key=self.api_key)
        self.thought_generator = ThoughtGenerator(api_key=self.api_key)
        self.state_evaluator = StateEvaluator(api_key=self.api_key)
        self.tree_of_thoughts = TreeOfThoughts(
            api_key=self.api_key,
            sample_data_manager=self.sample_data_manager,
            intent_classifier=self.intent_classifier,
            thought_generator=self.thought_generator,
            state_evaluator=self.state_evaluator
        )
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
            # Assuming an empty chat history for this example
            # In a real-world scenario, you might want to pass the actual chat history
            result = self.tree_of_thoughts.solve(
                user_input=user_query,
                chat_history=[],
                num_thoughts=num_thoughts,
                max_steps=max_steps,
                best_states_count=best_states_count
            )
            logging.info("Completed the problem-solving process.")
            return result
        except Exception as e:
            logging.error(f"Error during execution: {e}")




# # Example usage
# if __name__ == "__main__":

#     load_dotenv()
#     api_key = os.getenv("OPENAI_API_KEY") # Replace with your actual API key

#     try:
#         executor = TreeOfThoughtsExecutor(sample_csv_data=sample_csv_data, api_key=api_key)
#         output = executor.execute(user_query=user_query, num_thoughts=3, max_steps=3, best_states_count=2)
#         print(json.dumps(output, indent=2))
#     except TreeOfThoughtsError as e:
#         print(f"An error occurred: {e}")
#     except ValueError as e:
#         print(f"Invalid input: {e}")