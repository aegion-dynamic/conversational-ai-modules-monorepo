import logging
from os import mkdir
from pathlib import Path

from py import log
from state_machine.bot.chat_bot import CannabisRecommendationBot


log_path = Path('./logs/cannabis_bot.log')
if not log_path.exists():
    # log_path.parent.mkdir(parents=True)
    log_path.touch(exist_ok=True)

logging.basicConfig(
    filename='logs/cannabis_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
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
