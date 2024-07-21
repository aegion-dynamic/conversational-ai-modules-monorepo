from typing import List
import openai

class ThoughtGenerator:
    """Generates thoughts based on the current state."""

    def __init__(self, api_key: str):
        """
        Initialize the ThoughtGenerator.

        Args:
            api_key (str): OpenAI API key for accessing the language model.
        """
        self.api_key = api_key
        openai.api_key = self.api_key

    def generate(self, state: str, k: int) -> List[str]:
        """
        Generate k possible next thoughts based on the current state.
        
        Args:
            state (str): The current state of the problem-solving process.
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
            return thoughts[:k]  # Ensure we return exactly k thoughts
        except Exception as e:
            print(f"Error in thought generation: {e}")
            return [f"Error in thought generation: {e}"] * k