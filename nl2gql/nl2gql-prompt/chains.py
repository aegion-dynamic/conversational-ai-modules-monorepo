import os
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from .prompts import simple_prompt

def create_graphql_chain():    
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    output_parser = StrOutputParser()
    chain = simple_prompt | llm | output_parser
    return chain

def run_chain(chain, schema: str, user_query: str):
    return chain.invoke({"schema": schema, "user_query": user_query})
