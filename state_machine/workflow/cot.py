import openai
import json
from typing import List, Dict, Any, Optional
from enum import Enum, auto

class State(Enum):
    GATHERING_INFO = auto()
    RECOMMENDING = auto()

class ChainOfThoughtsModule:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.state = State.GATHERING_INFO
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
        
        if self.state == State.GATHERING_INFO:
            response = self.gather_information(user_input)
        else:
            response = self.provide_recommendation(user_input)
        
        self.conversation_history.append({"role": "assistant", "content": response['bot_response']})
        return response

    def gather_information(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze the following user input in the context of a medical cannabis recommendation system:
        User input: "{user_input}"
        Context: {json.dumps(self.context)}
        Conversation history: {json.dumps(self.conversation_history)}
        Knowledge base: {json.dumps(self.knowledge_base)}

        Provide a response in the following JSON format:
        {{
            "identified_condition": "Name of the medical condition if identified, or null",
            "severity": "Severity level if mentioned, or null",
            "mentioned_preferences": ["List", "of", "any", "mentioned", "preferences"],
            "intent": "User's perceived intent",
            "bot_response": "Response to the user",
            "follow_up_questions": [
                {{
                    "question": "A follow-up question based on missing information",
                    "options": ["Option 1", "Option 2", "Option 3"]
                }}
            ]
        }}
        """
        response = self.get_openai_response(prompt)
        self.update_context(response)
        
        if self.is_ready_to_recommend():
            self.state = State.RECOMMENDING
        
        return response

    def provide_recommendation(self, user_input: str) -> Dict[str, Any]:
        prompt = f"""
        Generate a medical cannabis product recommendation based on the following context:
        Context: {json.dumps(self.context)}
        User input: "{user_input}"
        Knowledge base: {json.dumps(self.knowledge_base)}

        Provide a response in the following JSON format:
        {{
            "bot_response": "Detailed product recommendation",
            "follow_up_questions": [
                {{
                    "question": "Do you have any questions about this recommendation?",
                    "options": ["Yes", "No"]
                }}
            ]
        }}
        """
        return self.get_openai_response(prompt)

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
        for key in ['identified_condition', 'severity', 'mentioned_preferences', 'intent']:
            if key in response and response[key]:
                self.context[key] = response[key]

    def is_ready_to_recommend(self) -> bool:
        required_info = ['identified_condition', 'severity', 'mentioned_preferences']
        return all(info in self.context for info in required_info)