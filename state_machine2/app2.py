"""
Enhanced Medical Cannabis Recommendation System

This module implements an improved version of the conversational AI system for providing
medical cannabis recommendations. It uses the OpenAI API to generate responses and guide
users through a structured conversation about their medical needs and potential cannabis
treatments.

Classes:
    State: Enum representing different states of the conversation.
    CannabisRecommendationSystem: Main class handling the recommendation logic.
    CannabisRecommendationApp: Application class to run the recommendation system.

Note: This version includes more comprehensive medical assessment and recommendation stages.
"""

import openai
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
import json
from typing import List, Dict, Any
from enum import Enum, auto

class State(Enum):
    """Enum representing different states of the conversation."""
    INITIAL_INQUIRY = auto()
    MEDICAL_ASSESSMENT = auto()
    RECOMMENDATION = auto()
    CONCLUSION = auto()

class CannabisRecommendationSystem:
    """
    Enhanced class for handling the cannabis recommendation system logic.

    This class manages the conversation state, processes user input, and generates
    appropriate responses using the OpenAI API. It includes more comprehensive
    medical assessment and recommendation stages.

    Attributes:
        api_key (str): OpenAI API key.
        client (openai.OpenAI): OpenAI client instance.
        state (State): Current state of the conversation.
        context (dict): Stores information gathered during the conversation.
        conversation_history (list): List of conversation messages.
        knowledge_base (dict): Stores information about cannabis products and effects.
        state_requirements (dict): Defines requirements and goals for each state.
    """

    def __init__(self, api_key: str):
        """
        Initialize the CannabisRecommendationSystem.

        Args:
            api_key (str): OpenAI API key.
        """
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        self.state = State.INITIAL_INQUIRY
        self.context = {}
        self.conversation_history = []
        self.knowledge_base = self.load_knowledge_base()
        self.state_requirements = {
            State.INITIAL_INQUIRY: {
                "required_info": ["is_medical_query"],
                "goal": "Understand the user's initial inquiry and overall context regarding health and cannabis."
            },
            State.MEDICAL_ASSESSMENT: {
                "required_info": ["main_symptom", "additional_symptoms", "symptom_severity", "symptom_duration", "previous_treatments", "medical_history", "lifestyle_factors"],
                "goal": "Gather comprehensive information about the user's health condition, symptoms, and relevant factors."
            },
            State.RECOMMENDATION: {
                "required_info": ["suitable_products", "usage_guidelines", "precautions", "expected_effects", "user_preferences"],
                "goal": "Provide tailored cannabis product recommendations based on the user's health information and preferences."
            },
            State.CONCLUSION: {
                "required_info": ["user_satisfaction", "understood_recommendations", "remaining_concerns", "next_steps"],
                "goal": "Ensure the user's questions are answered, recommendations are understood, and provide clear next steps."}
        }
        print(f"Initial State: {self.state.name}")

    def load_knowledge_base(self) -> Dict[str, Any]:
        """
        Load the knowledge base with information about cannabis products and effects.

        Returns:
            Dict[str, Any]: A dictionary containing the knowledge base information.
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
            Dict[str, Any]: A dictionary containing the system's response and other relevant information.
        """
        self.conversation_history.append(ChatCompletionUserMessageParam(content=user_input, role="user"))
        response = self.process_input(user_input)
        self.conversation_history.append(ChatCompletionSystemMessageParam(content=response['bot_response'], role="assistant"))
        return response

    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process the user input based on the current state and generate a response.

        Args:
            user_input (str): The user's input message.

        Returns:
            Dict[str, Any]: A dictionary containing the system's response and other relevant information.
        """
        current_state_info = self.state_requirements[self.state]
        print(f"\nCurrent State: {self.state.name}")
        print(f"State Goal: {current_state_info['goal']}")
        
        response = self.get_llm_response(user_input, current_state_info)
        self.context.update(response.get('extracted_info', {}))

        print("\nCurrent Context:")
        print(json.dumps(self.context, indent=2))

        missing_info = [info for info in current_state_info['required_info'] if info not in self.context]
        print("\nMissing Information:")
        print(json.dumps(missing_info, indent=2))

        if not missing_info or response.get('state_complete', False):
            print(f"\nGoal Fulfilled: {current_state_info['goal']}")
            self.transition_to_next_state()

        if self.state == State.CONCLUSION:
            response['conversation_complete'] = True
        
        return response

    def get_llm_response(self, user_input: str, state_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response using the OpenAI API based on the user input and current state.

        Args:
            user_input (str): The user's input message.
            state_info (Dict[str, Any]): Information about the current state.

        Returns:
            Dict[str, Any]: A dictionary containing the generated response and other relevant information.
        """
        prompt = f"""
        Analyze the following user input in the context of a medical cannabis recommendation system:
        User input: "{user_input}"
        Current state: {self.state.name}
        State goal: {state_info['goal']}
        Required information: {json.dumps(state_info['required_info'])}
        Current context: {json.dumps(self.context)}
        
        Tasks:
        1. Extract all relevant information from the user input that aligns with the required information for this state.
        2. Generate a concise, friendly response that addresses the user's input and moves towards fulfilling the state's goal.
        3. If in the RECOMMENDATION state, provide a complete recommendation with usage guidelines and precautions.
        4. If in the CONCLUSION state, summarize the recommendation and provide next steps without asking follow-up questions.
        5. For other states, if necessary, ask a single, focused follow-up question to gather missing information.
        6. Provide 2-3 sample options for the user to choose from in their response, except in the CONCLUSION state.
        7. Determine if the current state's goal has been sufficiently met to transition to the next state.

        Provide a response in the following JSON format:
        {{
            "bot_response": "Your concise response to the user",
            "extracted_info": {{"info_key": "value"}},
            "follow_up_question": "A single follow-up question, if needed",
            "sample_options": ["Option 1", "Option 2", "Option 3"],
            "state_complete": true/false
        }}
        """
        return self.get_openai_response(prompt)

    def transition_to_next_state(self):
        """
        Transition to the next state in the conversation flow.
        """
        state_order = list(State)
        current_index = state_order.index(self.state)
        if current_index < len(state_order) - 1:
            self.state = state_order[current_index + 1]
            print(f"\nTransitioning to: {self.state.name}")
            print(f"New State Goal: {self.state_requirements[self.state]['goal']}")
        else:
            print("\nConversation complete.")

    def get_openai_response(self, prompt: str) -> Dict[str, Any]:
        """
        Send a request to the OpenAI API and get a response.

        Args:
            prompt (str): The prompt to send to the API.

        Returns:
            Dict[str, Any]: A dictionary containing the API's response.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    ChatCompletionSystemMessageParam(content="You are a friendly medical cannabis recommendation assistant. Use simple language and keep responses concise.", role="system"),
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
                    return {"bot_response": "I didn't understand that. Could you please rephrase?"}
            else:
                return {"bot_response": "There was a problem with my response. Please try again."}
            
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            return {"bot_response": "I'm having trouble understanding. Can you try again?"}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"bot_response": "An unexpected error occurred. Please try again."}

class CannabisRecommendationApp:
    """
    Application class to run the Enhanced Cannabis Recommendation System.

    This class initializes the CannabisRecommendationSystem and handles the main interaction loop.

    Attributes:
        system (CannabisRecommendationSystem): An instance of the CannabisRecommendationSystem.
    """

    def __init__(self, api_key: str):
        """
        Initialize the CannabisRecommendationApp.

        Args:
            api_key (str): OpenAI API key.
        """
        self.system = CannabisRecommendationSystem(api_key)

    def run(self):
        """
        Run the main interaction loop for the Cannabis Recommendation System.
        """
        print("Welcome to the Medical Cannabis Recommendation Assistant!")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Thank you for using the Medical Cannabis Recommendation Assistant. Goodbye!")
                break

            response = self.system.chat(user_input)
            print(f"\nAssistant: {response['bot_response']}")
            print(f"Current State: {self.system.state.name}")

            if 'follow_up_question' in response and self.system.state != State.CONCLUSION:
                print(f"{response['follow_up_question']}")
                if 'sample_options' in response:
                    print("Sample options:")
                    for i, option in enumerate(response['sample_options'], 1):
                        print(f"{i}. {option}")

            if response.get('conversation_complete', False):
                print("\nThank you for using the Medical Cannabis Recommendation Assistant. Take care!")
                break

def main():
    """
    Main function to run the Enhanced Cannabis Recommendation App.
    """
    api_key ="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"  # Replace with your actual OpenAI API key
    app = CannabisRecommendationApp(api_key)
    app.run()

if __name__ == "__main__":
    main()