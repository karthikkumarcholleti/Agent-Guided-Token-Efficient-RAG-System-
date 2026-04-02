from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os
from datetime import datetime
import state

def initite_session():
    print("""Welcome to the Enterprise Policy QA System.
          Please enter a name for your session.""")
    session_name = input()
    if len(session_name.strip()) < 3:
      print("Session name is too short. Please enter a name with at least 3 characters.")
      return initite_session()
    print(f"Session '{session_name}' initialized.")
    return session_name


def prompt_input():
    prompt = input("User:")
    if len(prompt.strip()) < 3:
        print("Prompt too short. Please enter a valid prompt.")
        return prompt_input()
    chat_log = {
        "timestamp": datetime.now().isoformat(),
        "user_prompt": prompt
    }
    return chat_log



# def load_chat_to_vector_store(chat_log, system_response, session_name):
#     vector_store = Chroma(
#         collection_name = session_name,
#         embedding_function = OpenAIEmbeddings(
#             api_key = os.getenv("OPENAI_API_KEY"),
#             model = "text-embedding-3-small",
#             max_retries = 3,
#             request_timeout = 10
#         ),
#         persist_directory = "./Chat_history"
#     )
#     document = Document(
#         page_content = f"User: {chat_log['user_prompt']}\nSystem: {system_response}",
#         metadata = {"timestamp": chat_log['timestamp']},
#         id = f"{session_name}_{chat_log['timestamp']}"
#     )
#     vector_store.add_documents(documents = [document])


def save_to_chat_history(chat_log, system_response, session_name):
    if session_name not in state.session_names.keys():
        temp1 = {"role": "user", "content": chat_log['user_prompt'], "timestamp": chat_log['timestamp']}
        temp2 = {"role": "system", "content": system_response['system_response'], "timestamp": system_response['timestamp']}
        state.session_names[session_name] = [temp1, temp2]
    else:
        temp1 = {"role": "user", "content": chat_log['user_prompt'], "timestamp": chat_log['timestamp']}
        temp2 = {"role": "system", "content": system_response['system_response'], "timestamp": system_response['timestamp']}
        state.session_names[session_name].append(temp1)
        state.session_names[session_name].append(temp2)


def list_sessions():
    return list(state.session_names.keys())


def clear_session_data(session_name):
    if session_name in state.session_names:
        del state.session_names[session_name]
        print(f"Session data for '{session_name}' has been cleared.")
    else:
        print(f"No session data found for '{session_name}'.")

