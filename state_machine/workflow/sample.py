import enum
import json
import openai
import os

class State(enum.Enum):
    START = "start"
    UNDERSTANDING = "understanding"
    CLARIFYING = "clarifying"
    SOLVING = "solving"
    FINISHED = "finished"

class OpenAIAPI:
    def __init__(self):
        openai.api_key = "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"
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

    def call(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a problem-solving assistant that provides responses in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return json.dumps({
                "action": "finish",
                "new_state": "finished",
                "response": "I encountered an error. Let's end our conversation here.",
                "update_context": {},
                "new_goal": "End conversation due to error"
            })

class DynamicStatefulLLMSolver:
    def __init__(self, llm_api):
        self.llm_api = llm_api
        self.state = State.START
        self.context = {
            "problem": "",
            "understanding": "",
            "clarifications": [],
            "solution": "",
            "goal": "Initiate problem-solving process",
            "conversation_history": []
        }

    def run(self):
        while self.state != State.FINISHED:
            self.step()

    def step(self):
        prompt = self.create_prompt()
        response = self.llm_api.call(prompt)
        self.process_response(response)

    def create_prompt(self):
        base_prompt = f"""
Current State: {self.state.value}
Current Goal: {self.context['goal']}
Context: {json.dumps(self.context, indent=2)}

Based on the current state, goal, and context, determine the next action. 
Your response should be in JSON format with the following structure:
{{
    "action": "string",
    "new_state": "string",
    "response": "string",
    "update_context": {{ }},
    "new_goal": "string"
}}

Possible actions:
- "casual_conversation": Engage in casual conversation (START state)
- "ask_problem": Ask the user to describe their problem (START state)
- "summarize_understanding": Summarize current understanding (UNDERSTANDING state)
- "ask_clarification": Ask for specific clarification (CLARIFYING state)
- "provide_solution": Offer a solution (SOLVING state)
- "finish": End the conversation (any state)

Ensure the "new_state" is one of: "start", "understanding", "clarifying", "solving", "finished".
The "update_context" should contain any new information to add to or update in the context.
The "new_goal" should reflect the main objective for the next state.
"""
        return base_prompt

    def process_response(self, response):
        try:
            action_data = json.loads(response)
            self.state = State(action_data["new_state"])
            
            # Update context with any new information
            if "update_context" in action_data:
                self.context.update(action_data["update_context"])
            
            self.context["goal"] = action_data["new_goal"]
            
            if action_data["action"] == "casual_conversation":
                print(action_data["response"])
                user_input = input("Your response: ")
                self.context["conversation_history"].append(user_input)
            elif action_data["action"] == "ask_problem":
                self.context["problem"] = input(action_data["response"] + " ")
            elif action_data["action"] == "ask_clarification":
                clarification = input(action_data["response"] + " ")
                self.context["clarifications"].append(clarification)
            elif action_data["action"] in ["summarize_understanding", "provide_solution"]:
                print(action_data["response"])
                confirmation = input("Is this correct? (yes/no): ")
                if confirmation.lower() != "yes":
                    self.state = State.CLARIFYING
            elif action_data["action"] == "finish":
                print(action_data["response"])
                self.state = State.FINISHED
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error processing LLM response: {e}")
            self.state = State.FINISHED

# Usage
if __name__ == "__main__":
    openai_api = OpenAIAPI()
    solver = DynamicStatefulLLMSolver(openai_api)
    solver.run()
