import openai
import json
from typing import List, Dict, Any, Optional
from enum import Enum, auto

class State(Enum):
    CASUAL_CONVERSATION = auto()
    MEDICAL_INQUIRY = auto()
    UNDERSTANDING_PROBLEM = auto()
    SOLUTION_PROVISION = auto()

class ChainOfThoughtsModule:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.state = State.CASUAL_CONVERSATION
        self.context = {}
        self.conversation_history = []
        self.knowledge_base = self.load_knowledge_base()

    def load_knowledge_base(self) -> Dict[str, Any]:
        return {
            "categories": ["Tincture", "Vaporizer", "Edible", "Capsule", "Flower"],
            "effects": ["Non-euphoric", "Euphoric", "Balanced", "Sedating", "Energizing", "Relaxing"],
            "onsets": ["Immediate", "Fast", "Medium", "Slow"],
            "durations": ["Short", "Medium", "Long"],
            "timeOfUse": ["Any", "Day", "Night"],
            "medical_conditions": ["Pain", "Anxiety", "Insomnia", "Nausea", "Depression", "Inflammation", "Headache"]
        }

    def chat(self, user_input: str) -> Dict[str, Any]:
        self.conversation_history.append({"role": "user", "content": user_input})
        
        response = self.process_input(user_input)
        
        self.conversation_history.append({"role": "assistant", "content": response['bot_response']})
        return response

    def process_input(self, user_input: str) -> Dict[str, Any]:
        if self.state == State.CASUAL_CONVERSATION:
            return self.handle_casual_conversation(user_input)
        elif self.state == State.MEDICAL_INQUIRY:
            return self.handle_medical_inquiry(user_input)
        elif self.state == State.UNDERSTANDING_PROBLEM:
            return self.handle_understanding_problem(user_input)
        elif self.state == State.SOLUTION_PROVISION:
            return self.handle_solution_provision(user_input)

    def handle_casual_conversation(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze the following user input in the context of a medical cannabis recommendation system:
        User input: "{user_input}"
        
        Determine if the user is initiating a medical inquiry. If so, transition to the MEDICAL_INQUIRY state.
        Otherwise, maintain a casual conversation.

        Provide a response in the following JSON format:
        {{
            "bot_response": "Response to the user",
            "new_state": "CASUAL_CONVERSATION or MEDICAL_INQUIRY",
            "follow_up_questions": [
                {{
                    "question": "A follow-up question generated from the knowledge base",
                    "options": ["Option 1", "Option 2", "Option 3"]
                }}
            ]
        }}
        """
        response = self.get_openai_response(prompt)
        self.state = State[response['new_state']]
        return response

    def handle_medical_inquiry(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze the following user input in the context of a medical cannabis inquiry:
        User input: "{user_input}"
        Context: {json.dumps(self.context)}
        Knowledge base: {json.dumps(self.knowledge_base)}
        
        Determine if we have enough information to understand the problem. If so, transition to the UNDERSTANDING_PROBLEM state.
        Otherwise, gather more information about the medical condition.

        Generate follow-up questions based on the knowledge base. The questions should help understand the user's condition, preferences, and needs related to medical cannabis use.

        Provide a response in the following JSON format:
        {{
            "identified_condition": "Name of the medical condition if identified, or null",
            "severity": "Severity level if mentioned, or null",
            "mentioned_preferences": ["List", "of", "any", "mentioned", "preferences"],
            "bot_response": "Response to the user",
            "new_state": "MEDICAL_INQUIRY or UNDERSTANDING_PROBLEM",
            "follow_up_questions": [
                {{
                    "question": "A follow-up question generated from the knowledge base",
                    "options": ["Option 1", "Option 2", "Option 3"]
                }}
            ]
        }}
        """
        response = self.get_openai_response(prompt)
        self.update_context(response)
        self.state = State[response['new_state']]
        return response

    def handle_understanding_problem(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze the following user input and context to ensure we fully understand the medical problem:
        User input: "{user_input}"
        Context: {json.dumps(self.context)}
        Knowledge base: {json.dumps(self.knowledge_base)}
        
        Determine if we have sufficient information to provide a solution. If so, transition to the SOLUTION_PROVISION state.
        Otherwise, gather any remaining necessary information.

        Generate follow-up questions based on the knowledge base if more information is needed.

        Provide a response in the following JSON format:
        {{
            "bot_response": "Response to the user",
            "new_state": "UNDERSTANDING_PROBLEM or SOLUTION_PROVISION",
            "follow_up_questions": [
                {{
                    "question": "A follow-up question generated from the knowledge base",
                    "options": ["Option 1", "Option 2", "Option 3"]
                }}
            ]
        }}
        """
        response = self.get_openai_response(prompt)
        self.state = State[response['new_state']]
        return response

    def handle_solution_provision(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""
        Generate a medical cannabis product recommendation based on the following context:
        Context: {json.dumps(self.context)}
        User input: "{user_input}"
        Knowledge base: {json.dumps(self.knowledge_base)}

        Provide a response in the following JSON format:
        {{
            "bot_response": "Detailed product recommendation",
            "new_state": "SOLUTION_PROVISION or CASUAL_CONVERSATION",
            "follow_up_questions": [
                {{
                    "question": "Do you have any questions about this recommendation?",
                    "options": ["Yes", "No"]
                }}
            ]
        }}
        """
        response = self.get_openai_response(prompt)
        self.state = State[response['new_state']]
        return response

    def get_openai_response(self, prompt: str) -> Dict[str, Any]:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable medical cannabis recommendation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )
        return json.loads(response.choices[0].message['content'])

    def update_context(self, response: Dict[str, Any]):
        for key in ['identified_condition', 'severity', 'mentioned_preferences']:
            if key in response and response[key]:
                self.context[key] = response[key]