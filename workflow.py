from input_utils import initite_session, prompt_input, save_to_chat_history, list_sessions, clear_session_data
from prompt_utils import red_agent, memory_recall, ai_message_extractor, query_reshaper, n_finder
from rag_utils import connect_to_vector_store, filter_documens_content, retrieve_documents, get_sources
from model_utils import prompt_designer, initialize_model, query_response
from classification_utils import predict_use_memory
import state

def model_backend(chat_log, session_name, k = 5):
    user_prompt = chat_log['user_prompt']
    if predict_use_memory(user_prompt) == "Yes":
        if (session_name in state.session_names) and (len(state.session_names[session_name]) >= 1):
            n = n_finder(session_name)
            rag_query = query_reshaper(user_prompt, session_name, n = n)
            documents = retrieve_documents(rag_query, k = k)
            context_list = filter_documens_content(documents) 
            sources_list = get_sources(documents)
            prompt = prompt_designer(rag_query, context_list, sources_list)
            ans = query_response(prompt)
            return ans
        else:
            documents = retrieve_documents(user_prompt, k = k)
            context_list = filter_documens_content(documents) 
            sources_list = get_sources(documents)
            prompt = prompt_designer(user_prompt, context_list, sources_list)
            ans = query_response(prompt)
            return ans
    else:
        documents = retrieve_documents(user_prompt, k = k)
        context_list = filter_documens_content(documents) 
        sources_list = get_sources(documents)
        prompt = prompt_designer(user_prompt, context_list, sources_list)
        ans = query_response(prompt)
        return ans
    


def complete_workflow():
    sessions = list_sessions()
    print("Existing Sessions:", sessions)
    session_name = initite_session()
    while True:
        chat_log = prompt_input()
        if chat_log['user_prompt'].strip().lower() == 'exit':
            print("Exiting session.")
            break
        elif chat_log['user_prompt'].strip().lower() == 'clear session':
            clear_session_data(session_name)
            print(f"Session '{session_name}' cleared.")
            continue
        elif chat_log['user_prompt'].strip().lower() == 'list sessions':
            sessions = list_sessions()
            print("Existing Sessions:", sessions)
            continue
        elif chat_log['user_prompt'].strip().lower() == 'switch session':
            print("Existing Sessions:", sessions)
            session_name = initite_session()
            continue
        else:
            system_log = model_backend(chat_log, session_name)
            print(system_log['system_response'])
            save_to_chat_history(chat_log, system_log, session_name)