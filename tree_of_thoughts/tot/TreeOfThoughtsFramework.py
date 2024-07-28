import openai
import logging
import json
from typing import List, Tuple, Dict
import numpy as np

class TreeOfThoughtsFramework:
    def __init__(self, api_key: str):
        """
        Initialize the TreeOfThoughtsFramework with the given API key.
        
        Parameters:
        api_key (str): The API key for OpenAI.
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        self.logger = logging.getLogger(__name__)

    def get_openai_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text using OpenAI's API.
        
        Parameters:
        text (str): The input text to generate an embedding for.
        
        Returns:
        np.ndarray: The embedding as a numpy array.
        """
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = response['data'][0]['embedding']
            return np.array(embedding)
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return np.zeros(1536)  # Return a zero vector if there's an error (size may vary based on the model)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate the cosine similarity between two vectors.
        
        Parameters:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.
        
        Returns:
        float: The cosine similarity between vec1 and vec2.
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    def generate_thoughts(self, state: str, k: int) -> List[str]:
        """
        Generate k possible next thoughts or considerations based on the given state.
        
        Parameters:
        state (str): The current state of the problem.
        k (int): The number of thoughts to generate.
        
        Returns:
        List[str]: A list of generated thoughts.
        """
        prompt = f"Given the current state of the problem:\n\n{state}\n\nGenerate {k} possible next thoughts or considerations. Each thought should provide a new perspective or additional information that could be relevant to addressing the problem."
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant generating thoughts for problem-solving."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"Your response should be in the following format:\n1. [First thought]\n2. [Second thought]\n...\n{k}. [Last thought]"}
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
            return thoughts[:k]
        except Exception as e:
            self.logger.error(f"Error in thought generation: {e}")
            return [f"Error in thought generation: {e}"] * k

    def evaluate_states(self, states: List[str], query: str) -> List[float]:
        """
        Evaluate the given states in terms of their relevance and usefulness for addressing the problem.
        
        Parameters:
        states (List[str]): The list of states to evaluate.
        query (str): The query representing the problem.
        
        Returns:
        List[float]: A list of combined scores for each state.
        """
        prompt = "Evaluate the following states in terms of their relevance and usefulness for addressing the problem. Rate each state on a scale of 0 to 10, where 10 is the most relevant and useful."
        
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
            
            if len(ratings) < len(states):
                ratings.extend([0.0] * (len(states) - len(ratings)))
            ratings = ratings[:len(states)]
            
            # Calculate OpenAI embeddings for query and states
            query_embedding = self.get_openai_embedding(query)
            states_embeddings = [self.get_openai_embedding(state) for state in states]
            similarities = [self.cosine_similarity(query_embedding, state_embedding) for state_embedding in states_embeddings]
            
            # Combine ratings and similarities
            combined_scores = [(rating + similarity) / 2 for rating, similarity in zip(ratings, similarities)]
            return combined_scores
        except Exception as e:
            self.logger.error(f"Error in state evaluation: {e}")
            return [0.0] * len(states)

    def solve(self, problem: str, k: int = 3, T: int = 3, b: int = 2) -> Tuple[str, List[str]]:
        """
        Solve the given problem using a breadth-first search (BFS) approach.
        
        Parameters:
        problem (str): The initial problem state.
        k (int): The number of thoughts to generate at each step.
        T (int): The maximum number of steps to perform.
        b (int): The number of best states to keep at each step.
        
        Returns:
        Tuple[str, List[str]]: The best state and the path leading to it.
        """
        self.logger.info(f"Starting to solve problem: {problem}")
        self.logger.info(f"Parameters: k={k}, T={T}, b={b}")
        return self._bfs(problem, k, T, b)

    def _bfs(self, problem: str, k: int, T: int, b: int) -> Tuple[str, List[str]]:
        """
        Perform a breadth-first search (BFS) to solve the problem.
        
        Parameters:
        problem (str): The initial problem state.
        k (int): The number of thoughts to generate at each step.
        T (int): The maximum number of steps to perform.
        b (int): The number of best states to keep at each step.
        
        Returns:
        Tuple[str, List[str]]: The best state and the path leading to it.
        """
        initial_state = problem
        states = [(initial_state, [])]
        self.logger.info(f"Starting BFS with initial state: {initial_state}")

        for step in range(T):
            self.logger.info(f"Step {step + 1}/{T}")
            new_states = []
            for state, path in states:
                self.logger.debug(f"Generating thoughts for state: {state}")
                new_thoughts = self.generate_thoughts(state, k)
                self.logger.debug(f"Generated thoughts: {new_thoughts}")
                new_states.extend([(f"{state} -> {thought}", path + [thought]) for thought in new_thoughts])

            if not new_states:
                self.logger.warning("No new states generated. Stopping early.")
                break

            self.logger.info(f"Evaluating {len(new_states)} new states")
            values = self.evaluate_states([state for state, _ in new_states], problem)
            
            if len(values) != len(new_states):
                self.logger.warning(f"Mismatch between number of states ({len(new_states)}) and evaluations ({len(values)})")
                values = values[:len(new_states)]
            
            states = sorted(zip(new_states, values), key=lambda x: x[1], reverse=True)[:b]
            states = [state for state, _ in states]
            self.logger.info(f"Selected top {b} states: {[state[0] for state in states]}")

        if not states:
            self.logger.warning("No states remaining after search.")
            return (problem, ["No additional thoughts generated"])
        
        best_state, best_path = states[0]
        self.logger.info(f"BFS completed. Best state: {best_state}")
        self.logger.debug(f"Path to best state: {best_path}")
        return best_state, best_path

    def parse_solution(self, best_state: str, path: List[str]) -> Dict:
        """
        Parse the solution to create a structured representation of the thought process.
        
        Parameters:
        best_state (str): The best state found.
        path (List[str]): The path of thoughts leading to the best state.
        
        Returns:
        Dict: A dictionary containing the problem, final solution, and thought process.
        """
        steps = [{"step": i + 1, "thought": thought} for i, thought in enumerate(path)]
        
        return {
            "problem": best_state.split(" -> ")[0] if " -> " in best_state else best_state,
            "final_solution": best_state.split(" -> ")[-1] if " -> " in best_state else best_state,
            "thought_process": steps
        }

    def generate_output(self, parsed_solution: Dict) -> str:
        """
        Generate a formatted output string from the parsed solution.
        
        Parameters:
        parsed_solution (Dict): The parsed solution dictionary.
        
        Returns:
        str: The formatted output string.
        """
        output = f"Problem: {parsed_solution['problem']}\n\n"
        output += "Thought Process:\n"
        for step in parsed_solution['thought_process']:
            output += f"{step['step']}. {step['thought']}\n"
        output += f"\nFinal Solution: {parsed_solution['final_solution']}\n"
        
        return output

    def save_json(self, parsed_solution: Dict, filename: str = "solution.json") -> None:
        """
        Save the parsed solution to a JSON file.
        
        Parameters:
        parsed_solution (Dict): The parsed solution dictionary.
        filename (str): The name of the JSON file to save the solution to.
        """
        with open(filename, 'w') as f:
            json.dump(parsed_solution, f, indent=2)

def setup_logging(log_file='tree_of_thoughts.log', level=logging.INFO):
    """
    Set up logging for the TreeOfThoughtsFramework.
    
    Parameters:
    log_file (str): The name of the log file.
    level (int): The logging level.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'
    )
