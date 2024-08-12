import openai

class IntentClassifier:
    """
    A class for classifying user intent using OpenAI's GPT model.
    """

    def __init__(self, api_key: str, classification_prompt: str = None):
        """
        Initialize the IntentClassifier.

        Args:
            api_key (str): OpenAI API key.
            classification_prompt (str, optional): Custom prompt for intent classification.
        """
        self.api_key = api_key
        openai.api_key = self.api_key
        
        self.classification_prompt = classification_prompt or self.default_classification_prompt()

    @staticmethod
    def default_classification_prompt():
        """
        Provide a default classification prompt if none is provided.

        Returns:
            str: Default classification prompt.
        """
        return """
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

    def classify_intent(self, user_input: str) -> str:
        """
        Classify the intent of the user input using OpenAI's GPT model.

        Args:
            user_input (str): The user's input to classify.

        Returns:
            str: The classified intent as a string (1-5).
        """
        # Format the prompt with the user input
        prompt = self.classification_prompt.format(user_input=user_input)

        messages = [
            {"role": "system", "content": "You are a helpful assistant classifying user intent."},
            {"role": "user", "content": prompt}
        ]

        try:
            # Call OpenAI API for intent classification
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