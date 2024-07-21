import os
import logging
from dotenv import load_dotenv
from thought_generator import ThoughtGenerator
from state_evaluator import StateEvaluator
from tree_of_thoughts import TreeOfThoughts
from output_parser import OutputParser

def setup_logging(log_file='tree_of_thoughts.log', level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'
    )

def solve_problem(problem, api_key):
    logger = logging.getLogger(__name__)

    thought_generator = ThoughtGenerator(api_key)
    state_evaluator = StateEvaluator(api_key)
    tot_solver = TreeOfThoughts(thought_generator, state_evaluator)
    output_parser = OutputParser()

    logger.info(f"Initial problem: {problem}")

    best_state, path = tot_solver.solve(problem)
    logger.info(f"Best state: {best_state}")
    logger.info(f"Path: {path}")

    # Parse the solution
    parsed_solution = output_parser.parse_solution(best_state, path)
    
    # Generate formatted output
    formatted_output = output_parser.generate_output(parsed_solution)
    
    # Print the formatted output
    print("\nFormatted Output:")
    print(formatted_output)
    
    # Save the solution as JSON
    output_parser.save_json(parsed_solution)
    logger.info("Solution saved as JSON.")

def main():
    setup_logging()

    # Load API key from environment variable
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables.")
        return

    # Get problem statement from user input
    problem = input("Enter the problem statement: ")
    
    solve_problem(problem, api_key)

if __name__ == "__main__":
    main()
