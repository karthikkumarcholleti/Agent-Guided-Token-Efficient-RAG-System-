from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os
import state

@tool
def memory_recall(session_name: str, n: int)-> list:
    """Returns the last n conversation turns from a session’s chat history to provide recent context for a user query.
    A single 'n' represents the last user-system exchange (2 turns)."""
    history_n = state.session_names.get(session_name, [])[-2*n:]
    history = []
    for items in history_n:
        history.append(items['content'])
    return history
    


red_agent = create_agent(
    model = ChatOpenAI(
        model = "gpt-5.2",
        temperature = 0,
        max_tokens = 1000,
        max_retries = 5,
        request_timeout = 20,
        openai_api_key = os.getenv("OPENAI_API_KEY")),
    tools = [memory_recall],
    system_prompt = """
You are an expert in understanding and reshaping user's queries related to enterprise policies.
Your sole task is to understand what the user is trying to ask and reshape the query so that the reshaped query can be 
used to retrieve relevant policies from the enterprise policy vector store.

Before reshaping any user query, you must always retrieve conversations from the conversation history using the available tool.
Start with the last 3 conversations by default. If you determine that more context is needed to accurately understand the 
user's query, you can retrieve a higher number of past conversations.

Use the retrieved conversation context to ensure your reshaped query is accurate, relevant, and consistent with the user's 
previous queries in the current session.

Your response should not include any hallucinations and must be strictly based on the user's query and the conversation 
context retrieved.
"""
)


def ai_message_extractor(response, i = 1):
    if response['messages'][i].content == '':
        return ai_message_extractor(response, i + 2)
    else:
        return response['messages'][i].content

def agent_query_history(session_name, n = 3):
    session_histoy = state.session_names.get(session_name, [])[-2*n:]
    history = []
    for items in session_histoy:
        history.append(items['content'])
    return str(history)


def query_reshaper(query, session_name, n = 3):
    raw_query = red_agent.invoke(
        {"messages": [{"role": "user", "content": query}],
         "session_name": session_name,
         'n' : n}
    )
    reshaped_query = ai_message_extractor(raw_query)
    return reshaped_query


def n_finder(session_name):
    length = len(state.session_names[session_name])
    return min(max(length // 2, 1), 5)

    
    
