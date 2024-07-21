import logging
from sample_data_manager import SampleDataManager
from intent_classifier import IntentClassifier
from thought_generator import ThoughtGenerator
from state_evaluator import StateEvaluator
from tree_of_thoughts import TreeOfThoughts
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()

def setup_logging(log_file='application.log'):
    """Set up logging configuration to write to a file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'
    )
    return logging.getLogger(__name__)

def process_user_input(tot_solver: TreeOfThoughts, intent_classifier: IntentClassifier, user_input: str, chat_history: List[str]) -> Dict[str, Any]:
    """
    Process the user input using the Tree of Thoughts solver and intent classifier.

    Args:
        tot_solver (TreeOfThoughts): An instance of the TreeOfThoughts class.
        intent_classifier (IntentClassifier): An instance of the IntentClassifier class.
        user_input (str): The user's input query.
        chat_history (List[str]): The chat history.

    Returns:
        Dict[str, Any]: The processed output as a dictionary.
    """
    intent = intent_classifier.classify_intent(user_input)

    if intent in ['1', '2', '3', '5']:
        error_messages = {
            '1': "The user has made a phatic communication.",
            '2': "The user has used profanity or vulgar language.",
            '3': "The user has attempted an SQL injection.",
            '5': "The user's query is not related to the available data."
        }
        return {
            "summary": error_messages[intent],
            "quantitative_data": {},
            "qualitative_data": {},
            "user_requested_columns": [],
            "intent": ["phatic_communication", "profanity", "sql_injection", "other"][int(intent) - 1],
        }

    return tot_solver.solve(user_input, chat_history)

def load_api_key() -> str:
    """
    Load the OpenAI API key from environment variables.

    Returns:
        str: The OpenAI API key.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Error: OPENAI_API_KEY not found in environment variables.")
    return api_key

def initialize_components(api_key: str, sample_csv_data: str) -> Dict[str, Any]:
    """
    Initialize the necessary components for processing user input.

    Args:
        api_key (str): The OpenAI API key.
        sample_csv_data (str): The sample CSV data.

    Returns:
        Dict[str, Any]: A dictionary containing the initialized components.
    """
    sample_data_manager = SampleDataManager(sample_csv_data)
    intent_classifier = IntentClassifier(api_key)
    thought_generator = ThoughtGenerator(api_key)
    state_evaluator = StateEvaluator(api_key)
    tot_solver = TreeOfThoughts(api_key, sample_data_manager, intent_classifier, thought_generator, state_evaluator)

    return {
        "tot_solver": tot_solver,
        "intent_classifier": intent_classifier
    }

def main():
    logger = setup_logging()
    
    try:
        api_key = load_api_key()
    except ValueError as e:
        logger.error(e)
        return

    sample_csv_data = """
    Location,Room,Product,Category,PackageID,Batch,CBD,THC,CBDA,CBG,CBN,THCA,CustomerRating,MedicalBenefitsReported,RepeatPurchaseFrequency,URL,Description
    Hennep,Sales Floor,Tangerine | 1:1:1 THC:CBD:CBG Gummies 20pk,Gummies,1A40A0300001771000037968,120423TNG100,1.29 mg/g,1.53 mg/g,0.0 mg/g,1.29 mg/g,0.07 mg/g,0.0 mg/g,7,Improved sleep,Often,http://example.com/products/gummies/tangerine-|-1:1:1-thc:cbd:cbg-gummies-20pk,"Introducing our top-rated Tangerine 1:1:1 THC:CBD:CBG Gummies, carefully crafted to deliver a harmonious blend of therapeutic benefits in every bite. With a perfect balance of THC, CBD, and CBG in each delicious gummy, these tantalizing treats are designed to elevate your wellness routine with a touch of citrusy bliss.

    Experience the soothing effects of these gummies on your journey to a restful night's sleep. Our customers rave about the results"
    Hennep,Sales Floor,S'mores | Milk Chocolate Bar 20pk,ChocolateBar,1A40A0300001771000037569,081623SMCB100,0.0 mg/g,2.12 mg/g,0.0 mg/g,0.1 mg/g,0.05 mg/g,0.0 mg/g,4,Anxiety reduction,Rarely,http://example.com/products/chocolatebar/s'mores-|-milk-chocolate-bar-20pk,"Indulge in the creamy goodness of our S'mores | Milk Chocolate Bar 20pk, the perfect treat for those craving a decadent experience. Made with premium quality milk chocolate, each bar is meticulously crafted to deliver a rich and satisfying flavor that will melt in your mouth with every bite.

    Not only does our S'mores | Milk Chocolate Bar offer a delicious taste sensation, but it also provides potential medical benefits by helping to reduce anxiety. So, whether you need a sweet pick-me-up"
    Hennep,Sales Floor,Pax Era Life | Blaze |,Accessory,28,,,,,,,,8,Seizure control,Rarely,http://example.com/products/accessory/pax-era-life-|-blaze-|,"Introducing the revolutionary Pax Era Life | Blaze |, the ultimate accessory for modern individuals seeking top-notch seizure control in a sleek and convenient package. This cutting-edge product boasts a stellar Customer Rating of 8, reflecting the high satisfaction level among users who have experienced its life-changing benefits. 

    Crafted with precision and innovation, the Pax Era Life | Blaze | is designed for rare purchase frequencies, making it a coveted and esteemed addition to your wellness toolkit. With its advanced technology and medical-grade quality,"
    Hennep,Sales Floor,Pax Era Pro | Black |,Accessory,8.40E+11,,,,,,,,5,Nausea relief,Often,http://example.com/products/accessory/pax-era-pro-|-black-|,"Introducing the Pax Era Pro | Black |, the ultimate accessory that combines sleek design with practicality, perfect for those seeking relief from nausea. With a 5-star customer rating and a purchase frequency that is often, this product is a must-have for anyone looking for a stylish and effective solution.

    Crafted with precision and attention to detail, the Pax Era Pro in black offers a seamless vaping experience, making your relief from nausea quick and convenient. The cutting-edge technology of this accessory ensures optimal performance"
    Hennep,Sales Floor,Pax Era Pro | Gray |,Accessory,8.40E+11,,,,,,,,7,Pain relief,Never,http://example.com/products/accessory/pax-era-pro-|-gray-|,"Introducing the Pax Era Pro in sleek Gray - the ultimate accessory for enhanced pain relief. Elevate your vaping experience with cutting-edge technology and unparalleled convenience. 

    Crafted for the modern connoisseur, the Pax Era Pro is designed to deliver relief precisely when you need it. With a focus on pain relief, this accessory is a game-changer for those seeking holistic wellness solutions. 

    Featuring a glowing Customer Rating of 7, the Pax Era Pro has garnered praise for its sleek design,"
    Hennep,Sales Floor,Pax Era Pro | Sapphire |,Accessory,34,,,,,,,,10,Appetite stimulation,Never,http://example.com/products/accessory/pax-era-pro-|-sapphire-|,"Introducing the Pax Era Pro in enchanting Sapphire - the perfect companion for your vaping experience. Elevate your sessions with this sleek and advanced accessory designed for supreme functionality and style.

    Crafted with precision, the Pax Era Pro in Sapphire boasts a superb Customer Rating of 10, ensuring top-notch quality and user satisfaction. Seamlessly integrated technology allows for precise temperature control, ensuring a personalized vaping experience tailored to your preferences. The sleek, portable design makes it the perfect on-the-go companion for all your"
    Hennep,Sales Floor,Pax 3 Smart Complete | Sage,Accessory,41,,,,,,,,3,Improved sleep,Rarely,http://example.com/products/accessory/pax-3-smart-complete-|-sage,"Introducing the Pax 3 Smart Complete in the sophisticated Sage color, the must-have accessory for those seeking improved sleep and relaxation. This sleek and innovative product offers a unique blend of style and functionality, making it a standout choice in the world of wellness accessories.

    Designed with your well-being in mind, the Pax 3 Smart Complete is crafted to enhance your sleep quality and help you unwind after a long day. Whether you struggle with restlessness or simply want to create a more tranquil bedtime routine, this"
    Hennep,Sales Floor,Daily Drops | Sunrise Punch (50mL),Tincture,1A40A0300004A9D000014711,DD 9.6.23,0.00%,1.01%,0.0 mg/g,0.19%,0.06 mg/g,0.15 mg/g,7,Anxiety reduction,Rarely,http://example.com/products/tincture/daily-drops-|-sunrise-punch-(50ml),"Introducing Daily Drops | Sunrise Punch (50mL) - a soothing tincture specially crafted to brighten your day and ease your mind. This delightful blend offers a refreshing touch of Sunrise Punch flavor while providing effective relief from anxiety.

    Formulated with care and precision, this tincture is designed to help you find calm and balance in the midst of life's challenges. The harmonious combination of natural ingredients works seamlessly to promote relaxation and reduce feelings of stress and tension.

    With a customer rating"
    Hennep,Sales Floor,Dream Drops | Bedtime Bliss (60mL),Tincture,1A40A0300004A9D000014920,ZD 9.12.23,0.00%,0.54%,0.0 mg/g,0.01%,1.72 mg/g,0.07 mg/g,8,Improved sleep,Never,http://example.com/products/tincture/dream-drops-|-bedtime-bliss-(60ml),"Introducing Dream Drops | Bedtime Bliss, the ultimate solution for a restful night's sleep. This 60mL tincture is carefully crafted to promote improved sleep, so you can wake up feeling refreshed and rejuvenated every morning.

    Formulated with premium ingredients, Dream Drops | Bedtime Bliss offers a natural and effective way to enhance your bedtime routine. Simply take a few drops before bed and let the soothing blend work its magic as you drift off into a peaceful slumber.

    With a"
    """

    components = initialize_components(api_key, sample_csv_data)
    tot_solver = components["tot_solver"]
    intent_classifier = components["intent_classifier"]

    chat_history = []
    while True:
        user_input = input("Enter your query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        result = process_user_input(tot_solver, intent_classifier, user_input, chat_history)
        print(json.dumps(result, indent=2))

        chat_history.append(user_input)

if __name__ == "__main__":
    main()