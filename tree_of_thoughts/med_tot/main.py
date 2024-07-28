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
            raise TreeOfThoughtsError(f"An error occurred during execution: {e}")

class TreeOfThoughtsError(Exception):
    """Custom exception class for Tree of Thoughts errors."""
    pass

# Example usage
if __name__ == "__main__":
    sample_csv_data = """
    Location,Room,Product,Category,PackageID,Batch,CBD,THC,CBDA,CBG,CBN,THCA,CustomerRating,MedicalBenefitsReported,RepeatPurchaseFrequency,URL,Description
    Hennep,Sales Floor,Tangerine | 1:1:1 THC:CBD:CBG Gummies 20pk,Gummies,1A40A0300001771000037968,120423TNG100,1.29 mg/g,1.53 mg/g,0.0 mg/g,1.29 mg/g,0.07 mg/g,0.0 mg/g,7,Improved sleep,Often,http://example.com/products/gummies/tangerine-|-1:1:1-thc:cbd:cbg-gummies-20pk,"Introducing our top-rated Tangerine 1:1:1 THC:CBD:CBG Gummies, carefully crafted to deliver a harmonious blend of therapeutic benefits in every bite. With a perfect balance of THC, CBD, and CBG in each delicious gummy, these tantalizing treats are designed to elevate your wellness routine with a touch of citrusy bliss.

    Experience the soothing effects of these gummies on your journey to a restful night's sleep. Our customers rave about the results"
    Hennep,Sales Floor,S'mores | Milk Chocolate Bar 20pk,ChocolateBar,1A40A0300001771000037569,081623SMCB100,0.0 mg/g,2.12 mg/g,0.0 mg/g,0.1 mg/g,0.05 mg/g,0.0 mg/g,4,Anxiety reduction,Rarely,http://example.com/products/chocolatebar/s'mores-|-milk-chocolate-bar-20pk,"Indulge in the creamy goodness of our S'mores | Milk Chocolate Bar 20pk, the perfect treat for those craving a decadent experience. Made with premium quality milk chocolate, each bar is meticulously crafted to deliver a rich and satisfying flavor that will melt in your mouth with every bite.

    Not only does our S'mores | Milk Chocolate Bar offer a delicious taste sensation, but it also provides potential medical benefits by helping to reduce anxiety. So, whether you need a sweet pick-me-up"
    """
    user_query = "What category does 1906 Drops fall into?,"
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") # Replace with your actual API key

    try:
        executor = TreeOfThoughtsExecutor(sample_csv_data=sample_csv_data, api_key=api_key)
        output = executor.execute(user_query=user_query, num_thoughts=3, max_steps=3, best_states_count=2)
        print(json.dumps(output, indent=2))
    except TreeOfThoughtsError as e:
        print(f"An error occurred: {e}")
    except ValueError as e:
        print(f"Invalid input: {e}")