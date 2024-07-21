import logging
from typing import List, Tuple
from thought_generator import ThoughtGenerator
from state_evaluator import StateEvaluator

class TreeOfThoughts:
    """Implements the Tree of Thoughts algorithm."""

    def __init__(self, thought_generator: ThoughtGenerator, state_evaluator: StateEvaluator):
        """
        Initialize the TreeOfThoughts solver.

        Args:
            thought_generator (ThoughtGenerator): An instance of ThoughtGenerator.
            state_evaluator (StateEvaluator): An instance of StateEvaluator.
        """
        self.thought_generator = thought_generator
        self.state_evaluator = state_evaluator
        self.logger = logging.getLogger(__name__)

    def solve(self, problem: str, k: int = 3, T: int = 3, b: int = 2) -> Tuple[str, List[str]]:
        """
        Solve the problem using the Tree of Thoughts approach.
        
        Args:
            problem (str): The initial problem statement.
            k (int): Number of thoughts to generate at each step.
            T (int): Maximum number of thinking steps.
            b (int): Number of best states to keep at each step.
        
        Returns:
            Tuple[str, List[str]]: The best final state and the path of thoughts leading to it.
        """
        self.logger.info(f"Starting to solve problem: {problem}")
        self.logger.info(f"Parameters: k={k}, T={T}, b={b}")
        return self._bfs(problem, k, T, b)

    def _bfs(self, problem: str, k: int, T: int, b: int) -> Tuple[str, List[str]]:
        """
        Perform Breadth-First Search to explore the tree of thoughts.
        
        Args:
            problem (str): The initial problem statement.
            k (int): Number of thoughts to generate at each step.
            T (int): Maximum number of thinking steps.
            b (int): Number of best states to keep at each step.
        
        Returns:
            Tuple[str, List[str]]: The best final state and the path of thoughts leading to it.
        """
        initial_state = problem
        states = [(initial_state, [])]  # Initial state with empty path
        self.logger.info(f"Starting BFS with initial state: {initial_state}")

        for step in range(T):
            self.logger.info(f"Step {step + 1}/{T}")
            new_states = []
            for state, path in states:
                self.logger.debug(f"Generating thoughts for state: {state}")
                new_thoughts = self.thought_generator.generate(state, k)
                self.logger.debug(f"Generated thoughts: {new_thoughts}")
                new_states.extend([(f"{state} -> {thought}", path + [thought]) for thought in new_thoughts])

            if not new_states:
                self.logger.warning("No new states generated. Stopping early.")
                break

            self.logger.info(f"Evaluating {len(new_states)} new states")
            values = self.state_evaluator.evaluate([state for state, _ in new_states])
            
            if len(values) != len(new_states):
                self.logger.warning(f"Mismatch between number of states ({len(new_states)}) and evaluations ({len(values)})")
                values = values[:len(new_states)]  # Truncate values if necessary
            
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