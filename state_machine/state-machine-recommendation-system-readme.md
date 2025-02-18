# State Machine Recommendation System

## Overview

This project implements a flexible, state-based conversation system that guides users through a structured dialogue to provide personalized recommendations. Built around the concept of a state machine, it systematically collects necessary information before making tailored suggestions.

## System Architecture

![State Machine Recommendation System Architecture](path/to/your/architecture_diagram.png)

*The diagram above illustrates the core components and flow of the state machine recommendation system. It shows how user input flows through the state machine, how context is accumulated, and how recommendations are generated.*

## Core Concept: State-Driven Conversations

At its heart, this system uses a finite state machine to:

1. Guide conversations through predefined stages
2. Collect specific information at each stage
3. Use AI to understand user needs naturally
4. Generate personalized recommendations
5. Ensure user satisfaction through proper closure

Think of it as a smart conversation navigator that knows exactly what questions to ask and when to move to the next topic to deliver the best recommendations.

## How It Works

### State Machine Architecture

The system progresses through a series of states (e.g., "Initial Inquiry" → "Assessment" → "Recommendation" → "Conclusion"). Each state has:

- **Required Information**: Specific data points to collect
- **Goal**: The purpose of this conversation stage
- **Completion Criteria**: Conditions to move to the next state
- **Query Generation**: Optional logic to search for recommendations

### Key Components

1. **SystemConfig**: Defines states, knowledge base, and system behavior
2. **RecommendationSystem**: Manages state transitions and conversation flow
3. **Response**: Standardized structure for system outputs
4. **LLM Integration**: Leverages AI models for natural language understanding

### Flow Diagram

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Initial   │      │  Detailed   │      │ Personalized│      │ Confirmation│
│   Inquiry   │─────▶│ Assessment  │─────▶│Recommendation│─────▶│& Next Steps │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
```

## State Transition Visualization

![State Transition Flow](path/to/your/state_transition_diagram.png)

*This diagram shows how the system transitions between states based on information collection and user inputs. It illustrates the decision points that trigger state changes.*

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key or Azure OpenAI credentials

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/state-machine-recommendation.git

# Install dependencies
pip install openai pydantic python-dotenv
```

### Basic Configuration

```python
from state_machine2.app2 import SystemConfig, RecommendationSystem

# Create a configuration for your domain
my_config = SystemConfig(
    name="My Recommendation System",
    description="Helps users find the perfect [products/services]",
    initial_message="Hi! How can I help you today?",
    
    # Define your conversation states
    states={
        "FIRST_STATE": {
            "required_info": ["key_data_point1", "key_data_point2"],
            "goal": "Understand initial needs",
            "generate_queries": False
        },
        # Add more states as needed...
    },
    
    # Define relevant knowledge
    knowledge_base={
        "category1": ["option1", "option2", "option3"],
        "category2": ["choice1", "choice2", "choice3"]
    }
)

# Initialize system
system = RecommendationSystem(my_config)
```

## Example Implementation: Cannabis Recommendation System

The included implementation demonstrates how this architecture can be applied to medical cannabis recommendations:

### States Definition

1. **Initial Inquiry**:
   - Captures: `is_medical_query`, `main_symptom`
   - Goal: Understand why the user is seeking help

2. **Medical Assessment**:
   - Captures: `symptom_severity`, `symptom_duration`, `medical_history`, etc.
   - Goal: Build comprehensive understanding of medical needs

3. **Recommendation**:
   - Uses collected context to suggest products
   - Generates queries that could be used with a knowledge graph
   - Provides usage guidelines and precautions

4. **Conclusion**:
   - Ensures recommendations are understood
   - Addresses remaining concerns
   - Provides clear next steps

### Sample Usage

```python
from main import main

# Run the cannabis recommendation system
main()
```

### Example Conversation

```
System: Hi! I'm here to help recommend cannabis products for your needs. What brings you here today?

User: I've been having trouble sleeping lately.

System: I'm sorry to hear about your sleep troubles. That can be really frustrating. To help you better, could you tell me a bit more about your sleep issues? For example, do you have trouble falling asleep, staying asleep, or both?

[Conversation continues through assessment and recommendation states...]
```

### Conversation Flow Example

![Sample Conversation Flow](D:\conversational-ai-modules-monorepo\state_machine\Cannabis_workflow.png)

*This image shows a real example of a conversation flowing through different states, demonstrating how the system collects information and generates recommendations.*

## Making Your Own Recommendation System

To adapt this framework for your own domain:

1. Define your states and required information
2. Customize your knowledge base
3. Set appropriate goals for each state
4. Implement domain-specific query generation if needed

Example configuration:

```python
restaurant_config = SystemConfig(
    name="Restaurant Recommendation System",
    description="Helps users find restaurants matching their preferences",
    initial_message="Hello! Looking for a restaurant recommendation?",
    
    states={
        "PREFERENCE_COLLECTION": {
            "required_info": ["cuisine_type", "price_range", "location"],
            "goal": "Understand dining preferences",
            "generate_queries": False
        },
        "DETAILED_PREFERENCES": {
            "required_info": ["dietary_restrictions", "atmosphere", "occasion"],
            "goal": "Gather specific requirements",
            "generate_queries": False
        },
        "RECOMMENDATION": {
            "required_info": ["suitable_restaurants", "reservation_info"],
            "goal": "Provide tailored restaurant suggestions",
            "generate_queries": True
        },
        "CONCLUSION": {
            "required_info": ["satisfaction", "reservation_assistance"],
            "goal": "Confirm satisfaction and offer next steps",
            "generate_queries": False
        }
    },
    
    knowledge_base={
        "cuisine_types": ["Italian", "Japanese", "Mexican", "Indian", "American"],
        "price_ranges": ["Budget", "Mid-range", "Upscale", "Luxury"],
        "atmospheres": ["Casual", "Romantic", "Family-friendly", "Business", "Trendy"]
    }
)
```

## Future Improvements

### Tree of Thoughts Integration

We plan to enhance the system with a "tree of thoughts" approach that will:

- Explore multiple reasoning paths simultaneously
- Consider alternative interpretations of user input
- Generate more diverse recommendation options
- Dynamically select the most promising conversation paths

![Tree of Thoughts Concept](D:\conversational-ai-modules-monorepo\state_machine\tree_of_thoughts.jpeg)

*This illustration shows how the Tree of Thoughts approach allows the system to explore multiple reasoning paths for more robust recommendations.*

### Knowledge Graph Navigation

Future versions will incorporate knowledge graph capabilities to:

- Map relationships between symptoms, preferences, and recommendations
- Leverage graph structures for more intelligent questioning
- Enable more precise matching of user needs to recommendations
- Support complex reasoning about related concepts

![Knowledge Graph Integration](D:\conversational-ai-modules-monorepo\state_machine\ToG.png)

*The above diagram shows how a knowledge graph can be integrated with the state machine to make context-aware recommendations.*

### Adaptive Question Sequencing

We're developing adaptive algorithms that will:

- Optimize question ordering based on user responses
- Skip irrelevant questions automatically
- Focus on highest-value information gathering
- Personalize the conversation flow for each user

## Contributing

We welcome contributions! Please feel free to submit pull requests or open issues for:

- New domain configurations
- Enhanced reasoning capabilities
- User interface improvements
- Documentation and examples

## License

This project is licensed under the MIT License - see the LICENSE file for details.
