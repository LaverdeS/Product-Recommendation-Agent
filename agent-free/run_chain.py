import pandas as pd

from fire import Fire
from pathlib import Path
from dotenv import load_dotenv
from operator import itemgetter
from rich.console import Console
from rich.markdown import Markdown
from prompts import SYS_PROMPT, FILTER_PROMPT, RECOMMEND_PROMPT

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import chain
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder



# Load secrets
load_dotenv()

# Read the data
data_path = Path(__file__).resolve().parent.parent / 'data' / 'Ecommerce_Product_List.csv'
df = pd.read_csv(data_path)

# Load chat model
chatgpt = ChatOpenAI(model='gpt-4o-mini', temperature=0)


@chain
def pandas_code_tool_executor(query):
    print(query)
    result_df = eval(query)
    if result_df.empty:
        return df.to_markdown()
    else:
        return result_df.to_markdown()


def get_session_history_db(session_id):
    """Retrieves conversation history from database based on a specific user or session ID"""
    return SQLChatMessageHistory(
        session_id,
        connection="sqlite:///memory.db"
    )


def memory_buffer_window(messages, k=10):
    """
    Creates a memory buffer window function to return the last K conversations
    # 10 here means retrieve only the last 2*10 user-AI conversations
    """
    return messages[-(2 * k):]


def product_recommender_demo(user_query:str):
    # prompt to load in history and current input from the user
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", SYS_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", """Current User Query:
                         {human_input}
                      """),
        ]
    )



    # create a basic LLM Chain which only sends the last K conversations per user
    rephrase_query_chain = (
            RunnablePassthrough.assign(history=lambda x: memory_buffer_window(x["history"]))
            |
            prompt_template
            |
            chatgpt
            |
            StrOutputParser()
    )

    filter_prompt_template = ChatPromptTemplate.from_template(FILTER_PROMPT)

    data_filter_chain = (
             filter_prompt_template
               |
             chatgpt
               |
             StrOutputParser()
               |
             pandas_code_tool_executor
    )

    recommend_prompt_template = ChatPromptTemplate.from_template(RECOMMEND_PROMPT)

    recommend_chain = (
             recommend_prompt_template
               |
             chatgpt
               |
             StrOutputParser()
    )

    combined_chain = (
            {
                'human_input': itemgetter('human_input'),
                'history': itemgetter('history')
            }
            |
            {
                'user_query': rephrase_query_chain
            }
            |
            RunnablePassthrough.assign(product_table=data_filter_chain)
            |
            recommend_chain
    )

    # create a conversation chain which can load memory based on specific user or session id
    conv_chain = RunnableWithMessageHistory(
        combined_chain,
        get_session_history_db,
        input_messages_key="human_input",
        history_messages_key="history",
    )

    def chat_with_llm(prompt: str, session_id: str):
        """
        Utility function to take in current user input prompt and their session ID.
        Streams result live back to the user from the LLM.
        """
        response = conv_chain.invoke({"human_input": prompt},
                                     {'configurable': {'session_id': session_id}})
        console = Console()
        console.print(Markdown(response))

    # DEMO
    # Conversation chain for user 1
    user_id = 'jim001'
    prompt = "looking for a tablet"
    chat_with_llm(prompt, user_id)

    prompt = "want one which has display larger than 10 inches"
    chat_with_llm(prompt, user_id)

    prompt = "need at least 128GB disk space"
    chat_with_llm(prompt, user_id)

    # Conversation chain for user 2
    user_id = 'bond007'
    prompt = "I want a laptop with a high rating"
    chat_with_llm(prompt, user_id)

    prompt = "I want at least 16GB memory"
    chat_with_llm(prompt, user_id)


if __name__ == "__main__":
    Fire(product_recommender_demo)