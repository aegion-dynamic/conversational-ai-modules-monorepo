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
    """
    Enum representing the different states of the conversation.
    """
    INITIAL_INQUIRY = auto()
    MEDICAL_ASSESSMENT = auto()
    RECOMMENDATION = auto()
    CONCLUSION = auto()

class CannabisRecommendationSystem:
    """
    A system for providing medical cannabis recommendations based on user input and state management.
    """

    def __init__(self, api_key: str):
        """
        Initialize the CannabisRecommendationSystem.

        Args:
            api_key (str): The API key for OpenAI.
        """
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        self.state = State.INITIAL_INQUIRY
        self.context = {}
        self.conversation_history = []
        self.knowledge_base = self.load_knowledge_base()
        self.state_requirements = {
            State.INITIAL_INQUIRY: {
                "required_info": ["is_medical_query", "main_symptom"],
                "goal": "Understand the user's initial inquiry, overall context regarding health and cannabis, and identify the main symptom."
            },
            State.MEDICAL_ASSESSMENT: {
                "required_info": ["additional_symptoms", "symptom_severity", "symptom_duration", "previous_treatments", "medical_history", "lifestyle_factors"],
                "goal": "Gather comprehensive information about the user's health condition, symptoms, and relevant factors."
            },
            State.RECOMMENDATION: {
                "required_info": ["suitable_products", "usage_guidelines", "precautions", "expected_effects", "user_preferences"],
                "goal": "Provide tailored cannabis product recommendations based on the user's health information and preferences, using the knowledge base."
            },
            State.CONCLUSION: {
                "required_info": ["user_satisfaction", "understood_recommendations", "remaining_concerns", "next_steps"],
                "goal": "Ensure the user's questions are answered, recommendations are understood, and provide clear next steps."
            }
        }
        print(f"Initial State: {self.state.name}")

    def load_knowledge_base(self) -> Dict[str, Any]:
        """
        Load the knowledge base with cannabis-related information.

        Returns:
            Dict[str, Any]: A dictionary containing various categories of cannabis information.
        """
        return {
            "product_types": {
                "Oil drops": {"onset": "medium", "duration": "long"},
                "Vape pen": {"onset": "quick", "duration": "short"},
                "Edible": {"onset": "slow", "duration": "long"},
                "Pill": {"onset": "medium", "duration": "medium"},
                "Dried flower": {"onset": "quick", "duration": "medium"}
            },
            "effects": ["Calming", "Mood-lifting", "Balanced", "Sleep-promoting", "Energizing", "Relaxing"],
            "onset_times": {
                "Quick": "within minutes",
                "Fast": "15-30 minutes",
                "Medium": "30-60 minutes",
                "Slow": "1-2 hours"
            },
            "durations": {
                "Short": "1-2 hours",
                "Medium": "3-6 hours",
                "Long": "6-8 hours or more"
            },
            "use_times": ["Any time", "Daytime", "Nighttime"],
            "common_issues": ["Pain", "Anxiety", "Insomnia", "Nausea", "Depression", "Inflammation", "Headache"],
            "strengths": ["Mild", "Moderate", "Strong"],
            "main_ingredients": {
                "THC": {"effects": ["Euphoria", "Pain relief", "Appetite stimulation"]},
                "CBD": {"effects": ["Anti-inflammatory", "Anxiety reduction", "Non-psychoactive"]},
                "CBN": {"effects": ["Sedation", "Sleep aid"]},
                "CBG": {"effects": ["Anti-inflammatory", "Neuroprotective"]},
                "THCV": {"effects": ["Appetite suppression", "Energy boost"]}
            }
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
        Process the user input based on the current state and generate a response.

        Args:
            user_input (str): The user's input message.

        Returns:
            Dict[str, Any]: A dictionary containing the bot's response and other relevant information.
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
        Generate a response using the OpenAI language model.

        Args:
            user_input (str): The user's input message.
            state_info (Dict[str, Any]): Information about the current state.

        Returns:
            Dict[str, Any]: A dictionary containing the bot's response and other relevant information.
        """
        prompt = f"""
        Analyze the following user input in the context of a medical cannabis recommendation system:
        User input: "{user_input}"
        Current state: {self.state.name}
        State goal: {state_info['goal']}
        Required information: {json.dumps(state_info['required_info'])}
        Current context: {json.dumps(self.context)}
        Knowledge base: {json.dumps(self.knowledge_base)}
        
        Tasks:
        1. Extract all relevant information from the user input that aligns with the required information for this state.
        2. Generate a concise, friendly response that addresses the user's input and moves towards fulfilling the state's goal.
        3. If in the INITIAL_INQUIRY state, identify both the medical query and the main symptom.
        4. If in the RECOMMENDATION state:
           a. Generate 3 natural language queries based on the current context and user's symptoms. These queries should be formulated as questions that could be used to search a product database.
           b. Use the information in the current context to make the queries more specific and relevant.
           c. Avoid medical jargon and use simple, easy-to-understand language.
           d. Ensure questions are directly related to items in the knowledge base.
           e. Do not provide specific product recommendations or usage guidelines at this stage.
        5. For all states, generate 1-2 natural language queries that could be relevant based on the current context and state. These should be more general for earlier states and become more specific as the conversation progresses.
        6. If in the CONCLUSION state, summarize the conversation and provide next steps without asking follow-up questions.
        7. For states other than RECOMMENDATION and CONCLUSION, if necessary, ask a single, focused follow-up question to gather missing information.
        8. Determine if the current state's goal has been sufficiently met to transition to the next state.
        9. Ensure the response covers all aspects of the current state's goal.

        Provide a response in the following JSON format:
        {{
            "bot_response": "Your concise response to the user",
            "extracted_info": {{"info_key": "value"}},
            "follow_up_question": "A single follow-up question, if needed",
            "natural_language_queries": ["Query 1", "Query 2", "Query 3"],
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
        Get a response from the OpenAI API.

        Args:
            prompt (str): The prompt to send to the OpenAI API.

        Returns:
            Dict[str, Any]: A dictionary containing the bot's response and other relevant information.
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
    Application class for running the Cannabis Recommendation System.
    """

    def __init__(self, api_key: str):
        """
        Initialize the CannabisRecommendationApp.

        Args:
            api_key (str): The API key for OpenAI.
        """
        self.system = CannabisRecommendationSystem(api_key)

    def run(self):
        """
        Run the Cannabis Recommendation Application, handling user interactions.
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

            if 'natural_language_queries' in response:
                print("\nBased on our conversation, here are some queries we could use to find relevant information:")
                for i, query in enumerate(response['natural_language_queries'], 1):
                    print(f"{i}. {query}")

            if 'follow_up_question' in response and self.system.state != State.CONCLUSION:
                print(f"{response['follow_up_question']}")

            if response.get('conversation_complete', False):
                print("\nThank you for using the Medical Cannabis Recommendation Assistant. Take care!")
                break

from dotenv import load_dotenv
import os

def main():
    """
    Main function to run the Cannabis Recommendation Application.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    app = CannabisRecommendationApp(api_key)
    app.run()

if __name__ == "__main__":
    main()