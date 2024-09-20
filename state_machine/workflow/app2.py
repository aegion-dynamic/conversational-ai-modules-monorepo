from cot import ChainOfThoughtsModule, State
import json

class CannabisRecommendationApp:
    def __init__(self, api_key: str):
        self.cot_module = ChainOfThoughtsModule(api_key)
        self.session_data = {}

    def run(self):
        print("Welcome to the Medical Cannabis Recommendation Assistant!")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Thank you for using the Medical Cannabis Recommendation Assistant. Goodbye!")
                break

            response = self.cot_module.chat(user_input)
            
            print(f"Assistant: {response['bot_response']}")

            # Update session data
            self.update_session_data(response)

            # Print session data
            print("\nCurrent Session Data:")
            print(json.dumps(self.session_data, indent=2))

            # Print JSON output
            print("\nCurrent JSON Output:")
            print(json.dumps(response, indent=2))

            if response['follow_up_questions']:
                for q in response['follow_up_questions']:
                    print(f"\n{q['question']}")
                    for i, option in enumerate(q['options'], 1):
                        print(f"{i}. {option}")
                    
                    while True:
                        choice = input("Your choice (enter the number): ")
                        if choice.isdigit() and 1 <= int(choice) <= len(q['options']):
                            selected_option = q['options'][int(choice) - 1]
                            print(f"You selected: {selected_option}")
                            follow_up_response = self.cot_module.chat(selected_option)
                            print(f"Assistant: {follow_up_response['bot_response']}")
                            
                            # Update session data for follow-up
                            self.update_session_data(follow_up_response)
                            
                            # Print updated session data
                            print("\nUpdated Session Data:")
                            print(json.dumps(self.session_data, indent=2))
                            
                            # Print updated JSON output
                            print("\nUpdated JSON Output:")
                            print(json.dumps(follow_up_response, indent=2))
                            break
                        else:
                            print("Invalid choice. Please try again.")

    def update_session_data(self, response):
        if 'identified_condition' in response:
            self.session_data['condition'] = response['identified_condition']
        if 'severity' in response:
            self.session_data['severity'] = response['severity']
        if 'mentioned_preferences' in response:
            self.session_data['preferences'] = response['mentioned_preferences']
        if 'intent' in response:
            self.session_data['intent'] = response['intent']

def main():
    app = CannabisRecommendationApp("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")
    app.run()

if __name__ == "__main__":
    main()