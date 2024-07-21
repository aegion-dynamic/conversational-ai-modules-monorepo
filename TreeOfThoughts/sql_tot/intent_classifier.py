import openai
from typing import List

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