import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
from sample_data_manager import SampleDataManager
from intent_classifier import IntentClassifier
from thought_generator import ThoughtGenerator
from state_evaluator import StateEvaluator
from tree_of_thoughts import TreeOfThoughts

@dataclass
class DataInputs:
    """
    Data class to hold input parameters for TreeOfThoughtsExecutor.
    """
    api_key: Optional[str] = None
    user_query: Optional[str] = None
    sample_csv_data: Optional[str] = None
    num_thoughts: Optional[int] = None
    num_iterations: Optional[int] = None
    json_output_prompt: Optional[str] = None
    classification_prompt: Optional[str] = None
    thought_generation_prompt: Optional[str] = None
    evaluation_prompt: Optional[str] = None


class TreeOfThoughtsExecutor:
    """
    Executor class for the Tree of Thoughts process.
    
    This class initializes and executes the Tree of Thoughts algorithm
    using the provided input parameters.
    """

    def __init__(self, data_inputs: DataInputs):
        """
        Initialize the TreeOfThoughtsExecutor.

        Args:
            data_inputs (DataInputs): An instance of DataInputs containing all necessary parameters.

        Raises:
            ValueError: If required inputs are missing.
        """
        if not data_inputs:
            raise ValueError("DataInputs instance must be provided")
        if not data_inputs.json_output_prompt:
            raise ValueError("JSON output prompt must be provided")
        if not data_inputs.classification_prompt:
            raise ValueError("Classification prompt must be provided")

        self.data_inputs = data_inputs
        self.tree_of_thoughts = TreeOfThoughts(
            api_key=self.data_inputs.api_key,
            sample_data_manager=SampleDataManager(self.data_inputs.sample_csv_data),
            intent_classifier=IntentClassifier(
                api_key=self.data_inputs.api_key,
                classification_prompt=self.data_inputs.classification_prompt
            ),
            thought_generator=ThoughtGenerator(
                api_key=self.data_inputs.api_key,
                thought_generation_prompt=self.data_inputs.thought_generation_prompt
            ),
            state_evaluator=StateEvaluator(
                api_key=self.data_inputs.api_key,
                evaluation_prompt=self.data_inputs.evaluation_prompt
            ),
            json_output_prompt=self.data_inputs.json_output_prompt
        )
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the Tree of Thoughts process.

        Returns:
            Dict[str, Any]: The result of the Tree of Thoughts process.

        Raises:
            ValueError: If user_query is empty.
        """
        if not self.data_inputs.user_query:
            raise ValueError("user_query must not be empty")

        return self.tree_of_thoughts.solve(
            user_input=self.data_inputs.user_query,
            chat_history=[],  # Assuming no chat history for this example
            num_thoughts=self.data_inputs.num_thoughts,
            max_steps=self.data_inputs.num_iterations,
            best_states_count=2  # Adjust as needed
        )