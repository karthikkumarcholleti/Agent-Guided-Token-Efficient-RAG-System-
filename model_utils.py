from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
from datetime import datetime


def prompt_designer(query, context_list, sources_list):
    n = len(sources_list)
    dynamic_str = ""
    for i in range(n):
        temp = f"""context: {str(context_list[i])}
                source: {str(sources_list[i])}"""
        dynamic_str += temp
    messages = [
        ("system",
        """You are an expert in enterprise policies. Use the context and sources provided to answer the user's question accurately.
         Always cite sources if relevant.
         If the given context does not help or is not sufficient, respond saying not enough information provided.
         Do not make up any thing that is not in the context."""),
        ("human", """Query: {query}.
         {dynamic_str}
         """)]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    prompt = prompt_template.invoke({"query": query, "dynamic_str": dynamic_str})
    return prompt



def initialize_model():
    model = ChatOpenAI(
        model = "gpt-5.2",
        temperature = 0,
        max_tokens = 1000,
        max_retries = 5,
        request_timeout = 20,
        openai_api_key = os.getenv("OPENAI_API_KEY"))
    return model


def query_response(prompt):
    model = initialize_model()
    response = model.invoke(prompt)
    answer = response.content
    system_log = {
        "timestamp": datetime.now().isoformat(),
        "system_response": answer
    }
    return system_log