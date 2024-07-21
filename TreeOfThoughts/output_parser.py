import json
from typing import Dict, List

class OutputParser:
    def parse_solution(self, best_state: str, path: List[str]) -> Dict:
        """
        Parse the solution into a structured format.
        
        Args:
            best_state (str): The best final state.
            path (List[str]): The path of thoughts leading to the best state.
        
        Returns:
            Dict: A structured representation of the solution.
        """
        steps = [{"step": i + 1, "thought": thought} for i, thought in enumerate(path)]
        
        return {
            "problem": best_state.split(" -> ")[0] if " -> " in best_state else best_state,
            "final_solution": best_state.split(" -> ")[-1] if " -> " in best_state else best_state,
            "thought_process": steps
        }

    def generate_output(self, parsed_solution: Dict) -> str:
        """
        Generate a formatted string output from the parsed solution.
        
        Args:
            parsed_solution (Dict): The parsed solution.
        
        Returns:
            str: A formatted string representation of the solution.
        """
        output = f"Problem: {parsed_solution['problem']}\n\n"
        output += "Thought Process:\n"
        for step in parsed_solution['thought_process']:
            output += f"{step['step']}. {step['thought']}\n"
        output += f"\nFinal Solution: {parsed_solution['final_solution']}\n"
        
        return output

    def save_json(self, parsed_solution: Dict, filename: str = "solution.json") -> None:
        """
        Save the parsed solution as a JSON file.
        
        Args:
            parsed_solution (Dict): The parsed solution.
            filename (str): The name of the file to save the JSON to.
        """
        with open(filename, 'w') as f:
            json.dump(parsed_solution, f, indent=2)