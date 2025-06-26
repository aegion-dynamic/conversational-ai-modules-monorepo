from langchain.prompts import PromptTemplate

simple_prompt = PromptTemplate.from_template(
    """
You are a helpful assistant that generates GraphQL queries.

### SCHEMA ###
{schema}

### USER QUESTION ###
{user_query}

### GRAPHQL QUERY ###
"""
)
