from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .schema import LLM


def get_llm(llm: LLM) -> BaseChatModel:
    """
    Get an LLM instance based on the LLM enum.
    """

    match llm:
        case LLM.OPENAI_GPT_41:
            return ChatOpenAI(model="gpt-4.1")
        case _:
            raise ValueError(f"Unknown LLM: {llm}")
### This is just extremely out of date, and the latest OpenAI model is the 5 series. Chat completions is non-functional on it, and the chat model does not support reasoning. So, this is just kind of a bad file. Ignore this. Essentially, write your own better version. 