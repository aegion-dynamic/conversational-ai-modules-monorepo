from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr
from langchain_openai import AzureChatOpenAI

from utils.parameters import OPENAI_API_KEY, AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT


def get_default_llm():
    # llm = ChatOpenAI(model="gpt-4", api_key=SecretStr(OPENAI_API_KEY), temperature=0.3)  # Adjust for precision
    # Old Expert Setting was using
    # ChatOpenAI(
    #         api_key=SecretStr(OPENAI_API_KEY),
    #         temperature=0.1,
    #         model="gpt-4",
    #         verbose=True,
    #         max_tokens=1500,
    #     )

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=SecretStr(AZURE_OPENAI_KEY),
        api_version="2024-08-01-preview",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    return llm
