import json
import io
import csv
import openai
import logging
from typing import List, Dict, Any, Tuple

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

class IntentClassifier:
    """Classifies the user's intent based on their input."""

    def __init__(self, api_key: str):
        """
        Initialize the IntentClassifier with an OpenAI API key.

        Args:
            api_key (str): OpenAI API key for accessing the language model.
        """
        self.api_key = api_key
        openai.api_key = self.api_key

    def classify_intent(self, user_input: str) -> str:
        """
        Classify the user's intent based on their input.

        Args:
            user_input (str): The user's input text.

        Returns:
            str: A string representing the classified intent.
        """
        prompt = f"""
        Classify the user's intent based on the following input:

        User Input: {user_input}

        Possible intents:
        1. Phatic communication (greetings, farewells, etc.)
        2. Profanity or vulgar input
        3. SQL injection attempt
        4. Information request
        5. Other (not related to available data)

        Respond with only the number corresponding to the intent.
        """

        messages = [
            {"role": "system", "content": "You are a helpful assistant classifying user intent."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1,
                n=1,
                temperature=0.3
            )

            intent = response.choices[0].message['content'].strip()
            return intent
        except Exception as e:
            print(f"Error in intent classification: {e}")
            return "4"  # Default to information request in case of error

class ThoughtGenerator:
    """Generates thoughts based on the current state."""

    def __init__(self, api_key: str):
        """
        Initialize the ThoughtGenerator with an OpenAI API key.

        Args:
            api_key (str): OpenAI API key for accessing the language model.
        """
        self.api_key = api_key
        openai.api_key = self.api_key

    def generate_thoughts(self, current_state: str, num_thoughts: int) -> List[str]:
        """
        Generate possible next thoughts based on the current state.

        Args:
            current_state (str): The current state of the problem-solving process.
            num_thoughts (int): The number of thoughts to generate.

        Returns:
            List[str]: A list of generated thoughts.
        """
        prompt = f"Given the current state of the problem:\n\n{current_state}\n\nGenerate {num_thoughts} possible next thoughts or considerations. Each thought should provide a new perspective or additional information that could be relevant to addressing the problem."
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant generating thoughts for problem-solving."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"Your response should be in the following format:\n1. [First thought]\n2. [Second thought]\n...\n{num_thoughts}. [Last thought]"}
        ]
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=100,
                n=1,
                temperature=0.7
            )

            thoughts_text = response.choices[0].message['content'].strip()
            thoughts = [thought.split('. ', 1)[1] for thought in thoughts_text.split('\n') if '. ' in thought]
            return thoughts[:num_thoughts]  # Ensure we return exactly num_thoughts thoughts
        except Exception as e:
            print(f"Error in thought generation: {e}")
            return [f"Error in thought generation: {e}"] * num_thoughts

class StateEvaluator:
    """Evaluates the relevance and usefulness of each state."""

    def __init__(self, api_key: str):
        """
        Initialize the StateEvaluator with an OpenAI API key.

        Args:
            api_key (str): OpenAI API key for accessing the language model.
        """
        self.api_key = api_key
        openai.api_key = self.api_key

    def evaluate_states(self, states: List[str]) -> List[float]:
        """
        Evaluate the relevance and usefulness of each state for the problem.

        Args:
            states (List[str]): A list of states to evaluate.

        Returns:
            List[float]: A list of numerical ratings for each state.
        """
        prompt = f"Evaluate the following states in terms of their relevance and usefulness for addressing the problem. Rate each state on a scale of 0 to 10, where 10 is the most relevant and useful."
        
        states_text = "\n".join(f"{i+1}. {state}" for i, state in enumerate(states))
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant evaluating problem-solving states."},
            {"role": "user", "content": f"{prompt}\n\n{states_text}"},
            {"role": "user", "content": "Provide only the numerical ratings, one per line:"}
        ]
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=100,
                n=1,
                temperature=0.3
            )

            ratings_text = response.choices[0].message['content'].strip()
            ratings = [float(rating) for rating in ratings_text.split('\n') if rating.replace('.', '').isdigit()]
            
            # Ensure we have a rating for each state
            if len(ratings) < len(states):
                ratings.extend([0.0] * (len(states) - len(ratings)))
            return ratings[:len(states)]
        except Exception as e:
            print(f"Error in state evaluation: {e}")
            return [0.0] * len(states)

class TreeOfThoughts:
    """Implements the Tree of Thoughts algorithm with improved qualitative data extraction and handling of user-requested columns."""

    def __init__(self, api_key: str, sample_data_manager: SampleDataManager, 
                 intent_classifier: IntentClassifier, thought_generator: ThoughtGenerator, 
                 state_evaluator: StateEvaluator):
        """
        Initialize the TreeOfThoughts solver.

        Args:
            api_key (str): OpenAI API key for accessing the language model.
            sample_data_manager (SampleDataManager): An instance of SampleDataManager.
            intent_classifier (IntentClassifier): An instance of IntentClassifier.
            thought_generator (ThoughtGenerator): An instance of ThoughtGenerator.
            state_evaluator (StateEvaluator): An instance of StateEvaluator.
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        self.sample_data_manager = sample_data_manager
        self.intent_classifier = intent_classifier
        self.thought_generator = thought_generator
        self.state_evaluator = state_evaluator
        self.logger = logging.getLogger(__name__)

    def solve(self, user_input: str, chat_history: List[str], num_thoughts: int = 3, max_steps: int = 3, best_states_count: int = 2) -> Dict[str, Any]:
        """
        Solve the problem using the Tree of Thoughts approach.

        Args:
            user_input (str): The user's input query.
            chat_history (List[str]): The chat history.
            num_thoughts (int): Number of thoughts to generate at each step.
            max_steps (int): Maximum number of thinking steps.
            best_states_count (int): Number of best states to keep at each step.

        Returns:
            Dict[str, Any]: The final output as a dictionary.
        """
        self.logger.info(f"Starting solve method with user input: {user_input}")
        intent = self.intent_classifier.classify_intent(user_input)
        self.logger.info(f"Classified intent: {intent}")

        if intent in ['1', '2', '3', '5']:
            error_messages = {
                '1': "The user has made a phatic communication.",
                '2': "The user has used profanity or vulgar language.",
                '3': "The user has attempted an SQL injection.",
                '5': "The user's query is not related to the available data."
            }
            self.logger.warning(f"Intent classified as {intent}. Returning error message.")
            return {
                "summary": error_messages[intent],
                "quantitative_data": {},
                "qualitative_data": {},
                "user_requested_columns": [],
                "intent": ["phatic_communication", "profanity", "sql_injection", "other"][int(intent) - 1],
            }

        initial_state = f"User Input: {user_input}\nChat History: {chat_history}"
        self.logger.info("Starting tree search")
        best_state, thought_path = self._tree_search(initial_state, num_thoughts, max_steps, best_states_count)
        self.logger.info("Tree search completed")
        return self._generate_json_output(best_state, thought_path)

    def _tree_search(self, initial_state: str, num_thoughts: int, max_steps: int, best_states_count: int) -> Tuple[str, List[str]]:
        """
        Perform tree search to explore the thought space.

        Args:
            initial_state (str): The initial state of the problem.
            num_thoughts (int): Number of thoughts to generate at each step.
            max_steps (int): Maximum number of thinking steps.
            best_states_count (int): Number of best states to keep at each step.

        Returns:
            Tuple[str, List[str]]: The best final state and the path of thoughts leading to it.
        """
        self.logger.info(f"Starting tree search with num_thoughts={num_thoughts}, max_steps={max_steps}, best_states_count={best_states_count}")
        states = [(initial_state, [])]  # (state, path)

        for step in range(max_steps):
            self.logger.info(f"Tree search step {step + 1}")
            new_states = []
            for state, path in states:
                new_thoughts = self.thought_generator.generate_thoughts(state, num_thoughts)
                self.logger.debug(f"Generated {len(new_thoughts)} new thoughts")
                new_states.extend([(f"{state}\nThought: {thought}", path + [thought]) for thought in new_thoughts])

            if not new_states:
                self.logger.warning("No new states generated. Breaking tree search.")
                break

            values = self.state_evaluator.evaluate_states([state for state, _ in new_states])
            states = sorted(zip(new_states, values), key=lambda x: x[1], reverse=True)[:best_states_count]
            states = [state for state, _ in states]
            self.logger.info(f"Selected {len(states)} best states")

        self.logger.info("Tree search completed")
        return states[0] if states else (initial_state, [])

    def _generate_json_output(self, final_state: str, thought_path: List[str]) -> Dict[str, Any]:
        """
        Generate the final JSON output based on the best state and thought path.

        Args:
            final_state (str): The best final state.
            thought_path (List[str]): The path of thoughts leading to the final state.

        Returns:
            Dict[str, Any]: The final output as a dictionary.
        """
        self.logger.info("Generating JSON output")
        sample_data = self.sample_data_manager.get_sample_data()
        
        # Extract user query from final state
        user_query = final_state.split('\n')[0].replace('User Input: ', '')

        prompt = f"""
        Based on the following user query, final state, thought path, and sample data:

        User Query: {user_query}
        Final State: {final_state}
        Thought Path: {thought_path}

        Sample Data:
        {sample_data}

        Generate a JSON object with the following structure:
        {{
            "summary": "A detailed summary of the user input and analysis, focusing on qualitative aspects",
            "quantitative_data": {{
                "column name": "relevant numerical data from the analysis",
                ...
            }},
            "qualitative_data": {{
                "column name": "detailed qualitative information related to the query",
                ...
            }},
            "user_requested_columns": ["List of columns the user explicitly or implicitly requested"],
            "intent": "information_request",
        }}

        Guidelines for generating the response:
        1. Focus primarily on extracting and presenting qualitative data that's most relevant to the user's query.
        2. Include detailed descriptions, categories, or other text-based information in the qualitative_data section.
        3. For the quantitative_data, only include numerical data that's directly relevant to the query.
        4. In the summary, provide a comprehensive analysis that ties together the qualitative and quantitative aspects.
        5. Ensure all data included is directly related to the user's query.
        6. If the query mentions specific criteria (e.g., a particular rating), make sure to filter the data accordingly.
        7. For user_requested_columns, only include columns that are explicitly or implicitly requested in the user's query. If no columns are requested, return an empty list [].

        The columns present in our database are: "Location,Room,Product,Category,PackageID,Batch,CBD,THC,CBDA,CBG,CBN,THCA,CustomerRating,MedicalBenefitsReported,RepeatPurchaseFrequency,URL,Description"
        """

      
        messages = [
            {"role": "system", "content": "You are a helpful assistant generating detailed JSON output based on analysis results. Focus on providing rich, relevant qualitative data along with supporting quantitative information. Be precise in identifying user-requested columns, returning an empty list if none are explicitly or implicitly requested."},
            {"role": "user", "content": prompt}
        ]

        try:
            self.logger.info("Sending request to OpenAI API")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1000,
                n=1,
                temperature=0.2
            )
            self.logger.info("Received response from OpenAI API")

            # Remove any potential markdown formatting
            json_string = response.choices[0].message['content'].strip()
            if json_string.startswith('```json'):
                json_string = json_string[7:]  # Remove ```json
            if json_string.endswith('```'):
                json_string = json_string[:-3]  # Remove ```
            json_string = json_string.strip()  # Remove any leading/trailing whitespace

            json_output = json.loads(json_string)
            
            # Ensure user_requested_columns is an empty list if no columns were requested
            if 'user_requested_columns' not in json_output or not json_output['user_requested_columns']:
                json_output['user_requested_columns'] = []

            self.logger.info("JSON output generated successfully")
            return json_output
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON: {e}")
            self.logger.error(f"Response text: {response.choices[0].message['content']}")
            return {"error": "Failed to generate valid JSON output"}
        except Exception as e:
            self.logger.error(f"Error in generating JSON output: {e}")
            return {"error": f"An error occurred: {str(e)}"}