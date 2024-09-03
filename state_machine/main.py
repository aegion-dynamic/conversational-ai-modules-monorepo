import logging
from state_machine.bot.chat_bot import CannabisRecommendationBot

logging.basicConfig(
    filename='logs/cannabis_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    bot = CannabisRecommendationBot()
    print("Welcome to the Cannabis Recommendation Bot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = bot.process_user_input(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
