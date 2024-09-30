import openai
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
import json
from typing import List, Dict, Any
from enum import Enum, auto

class State(Enum):
    """Enum class to represent the different states of the conversation."""
    CASUAL_CONVERSATION = auto()
    MEDICAL_CONDITION_UNDERSTANDING = auto()
    SOLUTION_PROVISION = auto()
    END_CONVERSATION = auto()

class CannabisRecommendationSystem:
    """
    A system for recommending medical cannabis products based on user input and health conditions.
    """

    def __init__(self, api_key: str):
        """
        Initialize the CannabisRecommendationSystem.

        Args:
            api_key (str): The OpenAI API key for accessing GPT-4.
        """
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        self.state = State.CASUAL_CONVERSATION
        self.context = {}
        self.conversation_history = []
        self.knowledge_base = self.load_knowledge_base()
    def load_knowledge_base(self) -> Dict[str, Any]:
        """
        Load the knowledge base with information about cannabis products and effects.

        Returns:
            Dict[str, Any]: A dictionary containing various categories of cannabis-related information.
        """
        return {
            "product_types": ["Oil drops", "Vape pen", "Edible", "Pill", "Dried flower"],
            "effects": ["Calming", "Mood-lifting", "Balanced", "Sleep-promoting", "Energizing", "Relaxing"],
            "onset_times": ["Quick", "Fast", "Medium", "Slow"],
            "durations": ["Short", "Medium", "Long"],
            "use_times": ["Any time", "Daytime", "Nighttime"],
            "common_issues": ["Pain", "Worry", "Sleep problems", "Nausea", "Low mood", "Swelling", "Headache"],
            "strengths": ["Mild", "Moderate", "Strong"],
            "main_ingredients": ["THC", "CBD", "CBN", "CBG", "THCV"]
        }

    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and generate a response.

        Args:
            user_input (str): The user's input message.

        Returns:
            Dict[str, Any]: A dictionary containing the bot's response and other relevant information.
        """
        self.conversation_history.append(ChatCompletionUserMessageParam(content=user_input, role="user"))
        response = self.process_input(user_input)
        self.conversation_history.append(ChatCompletionSystemMessageParam(content=response['bot_response'], role="assistant"))
        return response

    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input based on the current conversation state.

        Args:
            user_input (str): The user's input message.

        Returns:
            Dict[str, Any]: A dictionary containing the bot's response and other relevant information.
        """
        if self.state == State.CASUAL_CONVERSATION:
            return self.handle_casual_conversation(user_input)
        elif self.state == State.MEDICAL_CONDITION_UNDERSTANDING:
            return self.handle_medical_condition_understanding(user_input)
        elif self.state == State.SOLUTION_PROVISION:
            return self.handle_solution_provision(user_input)
        elif self.state == State.END_CONVERSATION:
            return self.handle_end_conversation(user_input)

    def handle_casual_conversation(self, user_input: str) -> Dict[str, Any]:
        """
        Handle the casual conversation state, determining if the user is asking about a health issue.

        Args:
            user_input (str): The user's input message.

        Returns:
            Dict[str, Any]: A dictionary containing the bot's response and new state information.
        """
        prompt = f"""
        Analyze the following user input in the context of a medical cannabis recommendation system:
        User input: "{user_input}"
        
        Tasks:
        1. Determine if the user is asking for help with a health issue that might benefit from medical cannabis.
        2. If so, suggest transitioning to the MEDICAL_CONDITION_UNDERSTANDING state.
        3. If not, continue the casual conversation while gently steering towards health-related topics.

        Provide a response in the following JSON format:
        {{
            "bot_response": "Your response to the user",
            "new_state": "CASUAL_CONVERSATION or MEDICAL_CONDITION_UNDERSTANDING",
            "is_health_inquiry": true/false
        }}
        """
        response = self.get_openai_response(prompt)
        
        if response['is_health_inquiry']:
            self.state = State.MEDICAL_CONDITION_UNDERSTANDING
        else:
            self.state = State[response['new_state']]
        
        return response

    def handle_medical_condition_understanding(self, user_input: str) -> Dict[str, Any]:
        """
        Handle the medical condition understanding state, gathering information about the user's health issue.

        Args:
            user_input (str): The user's input message describing their health issue.

        Returns:
            Dict[str, Any]: A dictionary containing the bot's response, identified issues, and follow-up questions.
        """
        prompt = f"""
        Analyze the following user input to understand their health issue:
        User input: "{user_input}"
        Conversation history: {json.dumps(self.conversation_history)}
        
        Tasks:
        1. Identify the main health issue or symptom the user is seeking help for. If it's a common issue like a headache, try to determine if it might be a more specific condition (e.g., migraine, tension headache).
        2. Extract any additional symptoms or concerns mentioned.
        3. Provide a compassionate response acknowledging the user's condition.
        4. Generate 5-7 follow-up questions to gather more information about the user's condition and preferences. These questions should be similar to what a doctor might ask, tailored to the user's specific health issue, and focused on information that will help recommend an appropriate cannabis product. Consider the following aspects:
           - Severity and frequency of symptoms
           - How long they've been experiencing the issue
           - What makes the symptoms better or worse
           - Their experience with cannabis (if any)
           - Their preferences for taking medication (e.g., pills, liquids, etc.)
           - When they typically need relief (day/night)
           - How quickly they need the effects to start
           - How long they need the effects to last
        5. For each question, provide 3-5 simple answer options as examples, but encourage the user to answer in their own words.

        Provide a response in the following JSON format:
        {{
            "bot_response": "Your compassionate response to the user",
            "identified_issue": "The main health issue or symptom identified, including any potential specific conditions",
            "additional_symptoms": ["symptom1", "symptom2", ...],
            "follow_up_questions": [
                {{
                    "question": "Question text",
                    "example_options": ["Option 1", "Option 2", "Option 3"]
                }},
                ...
            ]
        }}
        """
        response = self.get_openai_response(prompt)
        
        self.context['identified_issue'] = response['identified_issue']
        self.context['additional_symptoms'] = response['additional_symptoms']
        self.context['follow_up_questions'] = response['follow_up_questions']
        
        return response

    def handle_solution_provision(self, user_input: str) -> Dict[str, Any]:
        """
        Handle the solution provision state, recommending cannabis products based on the user's health issue.

        Args:
            user_input (str): The user's input message (not used in this method, but kept for consistency).

        Returns:
            Dict[str, Any]: A dictionary containing the bot's response, recommended products, and usage guidelines.
        """
        prompt = f"""
        Based on the following context, provide a detailed cannabis product recommendation:
        Context: {json.dumps(self.context)}
        Knowledge base: {json.dumps(self.knowledge_base)}

        Tasks:
        1. Summarize the user's health issue, symptoms, and preferences using simple language.
        2. Recommend 2-3 specific cannabis products that might help with the user's health concerns.
        3. Explain why you're recommending these products, including how they might help.
        4. Provide simple guidelines on how to use the products, including how much to use and when.
        5. Mention any potential side effects or things to be careful about.

        Provide a response in the following JSON format:
        {{
            "bot_response": "Your detailed summary and product recommendation in simple terms",
            "summary": "A brief summary of the user's health issue and needs",
            "recommended_products": [
                {{
                    "name": "Product name",
                    "type": "Product type from knowledge base",
                    "main_ingredients": ["Main active ingredients"],
                    "effects": ["Expected effects"],
                    "usage": "Simple usage instructions"
                }},
                ...
            ],
            "usage_guidelines": "General usage and dosage information in simple terms",
            "precautions": "Any important precautions or potential side effects in simple terms"
        }}
        """
        response = self.get_openai_response(prompt)
        self.state = State.END_CONVERSATION
        return response

    def handle_end_conversation(self, user_input: str) -> Dict[str, Any]:
        """
        Handle the end conversation state, providing a closing message to the user.

        Args:
            user_input (str): The user's input message (not used in this method, but kept for consistency).

        Returns:
            Dict[str, Any]: A dictionary containing the bot's closing response.
        """
        return {
            "bot_response": "Thank you for using our medical cannabis recommendation system. Remember to talk to a healthcare professional before starting any new treatment. Is there anything else I can help you with?",
            "new_state": "END_CONVERSATION"
        }

    def get_openai_response(self, prompt: str) -> Dict[str, Any]:
        """
        Get a response from the OpenAI API using the GPT-4 model.

        Args:
            prompt (str): The prompt to send to the OpenAI API.

        Returns:
            Dict[str, Any]: A dictionary containing the API's response, parsed from JSON.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    ChatCompletionSystemMessageParam(content="You are a medical cannabis recommendation assistant. Only answer questions related to medical cannabis use. Use simple language that's easy for non-medical people to understand. If asked about non-medical topics, politely redirect the conversation to medical cannabis-related topics.", role="system"),
                    ChatCompletionUserMessageParam(content=prompt, role="user")
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            if response and response.choices:
                content = response.choices[0].message.content
                if content:
                    return json.loads(content)
                else:
                    return {"bot_response": "I'm sorry, I didn't understand that. Could you please ask your question again?", "new_state": "CASUAL_CONVERSATION"}
            else:
                return {"bot_response": "There was a problem with my response. Please try again later.", "new_state": "CASUAL_CONVERSATION"}
            
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            return {"bot_response": "I'm sorry, there was a problem understanding your request.", "new_state": "CASUAL_CONVERSATION"}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"bot_response": "An unexpected error occurred. Please try again later.", "new_state": "CASUAL_CONVERSATION"}

class CannabisRecommendationApp:
    """
    A class to run the Cannabis Recommendation System as an interactive application.
    """

    def __init__(self, api_key: str):
        """
        Initialize the CannabisRecommendationApp.

        Args:
            api_key (str): The OpenAI API key for accessing GPT-4.
        """
        self.system = CannabisRecommendationSystem(api_key)

    def run(self):
        """
        Run the Cannabis Recommendation Application, managing the conversation flow with the user.
        """
        print("Welcome to the Medical Cannabis Recommendation Assistant!")
        while True:
            print(f"\nCurrent State: {self.system.state}")
            
            if self.system.state == State.CASUAL_CONVERSATION:
                user_input = input("You: ")
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Thank you for using the Medical Cannabis Recommendation Assistant. Goodbye!")
                    break
                response = self.system.chat(user_input)
                print(f"Assistant: {response['bot_response']}")

            elif self.system.state == State.MEDICAL_CONDITION_UNDERSTANDING:
                user_input = input("Please describe your health issue or symptoms: ")
                response = self.system.chat(user_input)
                print(f"Assistant: {response['bot_response']}")
                
                for question in response['follow_up_questions']:
                    print(f"\nAssistant: {question['question']}")
                    print("For example, you could say something like:")
                    for option in question['example_options']:
                        print(f"- {option}")
                    print("But feel free to answer in your own words.")
                    
                    answer = input("Your answer: ")
                    self.system.context[question['question']] = answer
                    print("Assistant: Thank you for your response.")

                print("\nThank you for answering all the questions. Now I'll provide a recommendation.")
                self.system.state = State.SOLUTION_PROVISION

            elif self.system.state == State.SOLUTION_PROVISION:
                response = self.system.chat("")
                print(f"Assistant: {response['bot_response']}")
                print("\nRecommended Products:")
                for product in response.get('recommended_products', []):
                    print(f"- {product['name']} ({product['type']})")
                    print(f"  Main ingredients: {', '.join(product['main_ingredients'])}")
                    print(f"  Effects: {', '.join(product['effects'])}")
                    print(f"  Usage: {product['usage']}")
                    print()
                print(f"Usage Guidelines: {response.get('usage_guidelines', '')}")
                print(f"Precautions: {response.get('precautions', '')}")
                self.system.state = State.END_CONVERSATION

            elif self.system.state == State.END_CONVERSATION:
                user_input = input("Is there anything else I can help you with? (yes/no): ")
                if user_input.lower() == 'yes':
                    self.system.state = State.CASUAL_CONVERSATION
                else:
                    print("Thank you for using the Medical Cannabis Recommendation Assistant. Goodbye!")
                    self.print_session_summary()
                    break

    def print_session_summary(self):
        """
        Print a summary of the conversation session, including the context and conversation history.
        """
        print("\nSession Summary:")
        print(json.dumps(self.system.context, indent=2))
        print("\nConversation History:")
        for message in self.system.conversation_history:
            print(f"{message['role'].capitalize()}: {message['content']}")

def main():
    """
    The main function to run the Cannabis Recommendation Application.
    """
    app = CannabisRecommendationApp("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")
    app.run()

if __name__ == "__main__":
    main()