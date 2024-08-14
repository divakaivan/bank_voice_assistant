
import os
import re
from typing import Dict, List

from openai import OpenAI as gpt_client
import llama_index
import phoenix as px
import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS
from llama_index import SQLDatabase, VectorStoreIndex, load_index_from_storage
from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.llms import ChatResponse, OpenAI
from llama_index.objects import (ObjectIndex, SQLTableNodeMapping,
                                 SQLTableSchema)
from llama_index.prompts import PromptTemplate
from llama_index.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.query_pipeline import FnComponent, InputComponent
from llama_index.query_pipeline import QueryPipeline as QP
from llama_index.retrievers import SQLRetriever
from llama_index.service_context import ServiceContext
from llama_index.storage import StorageContext
from sqlalchemy import create_engine
from streamlit_mic_recorder import mic_recorder
from transformers import pipeline

load_dotenv()

# px.launch_app()
# llama_index.set_global_handler("arize_phoenix")
# below helps to avoid launching tracing on every refresh
if "app_launched" not in st.session_state:
    st.session_state["app_launched"] = False
if not st.session_state["app_launched"]:
    px.launch_app()
    llama_index.set_global_handler("arize_phoenix")
    st.session_state["app_launched"] = True

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

engine = create_engine("sqlite:///app/transactions.db")
sql_database = SQLDatabase(engine)

def get_table_context_str(table_schema_objs: List[SQLTableSchema]):
    """
    Given a list of table schema objects, return a string with the context of each table.
    """
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(table_schema_obj.table_name)
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context
        context_strs.append(table_info)
    return "\n\n".join(context_strs)

table_parser_component = FnComponent(fn=get_table_context_str)

class TableInfo(BaseModel):
    table_name: str = Field(..., description="table name (must be underscores and NO spaces)")
    table_summary: str = Field(..., description="short, concise summary/caption of the table")

table_info = TableInfo.parse_file("app/table_info_directory/0_Ivanov_Transactions.json")

table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [SQLTableSchema(table_name=table_info.table_name, context_str=table_info.table_summary)]

obj_index = ObjectIndex.from_objects(table_schema_objs, table_node_mapping, VectorStoreIndex)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)

def parse_response_to_sql(response: ChatResponse) -> str:
    """
    Given a response from the LLM, parse out the SQL query.
    """
    response = response.message.content
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    return response.strip().strip("```").strip()

sql_parser_component = FnComponent(fn=parse_response_to_sql)

response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    formatted_history = ""
    for message in chat_history:
        role = "User" if message["role"] == "Human" else "Assistant"
        formatted_history += f"{role}: {message['content']}\n"
    return formatted_history.strip()

# Initialize Query Pipeline
qp = QP(verbose=False)

service_context = ServiceContext.from_defaults(callback_manager=qp.callback_manager)

def index_all_tables(sql_database: SQLDatabase, table_index_dir: str = "table_index_dir") -> Dict[str, VectorStoreIndex]:
    """
    Index all tables in the SQL database.
    """
    vector_index_dict = {}
    for table_name in sql_database.get_usable_table_names():
        storage_context = StorageContext.from_defaults(persist_dir=f"app/{table_index_dir}/{table_name}")
        index = load_index_from_storage(storage_context, index_id="vector_index", service_context=service_context)
        vector_index_dict[table_name] = index
    return vector_index_dict

vector_index_dict = index_all_tables(sql_database)

# Initialize SQL Retriever
sql_retriever = SQLRetriever(sql_database)

def get_table_context_and_rows_str(query_str: str, table_schema_objs: List[SQLTableSchema]):
    """
    Given a query string and a list of table schema objects, return a string with the context of each table and some example rows
    """
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(table_schema_obj.table_name)
        if table_schema_obj.context_str:
            table_opt_context = " The table description is: "
            table_opt_context += table_schema_obj.context_str
            table_info += table_opt_context

        vector_retriever = vector_index_dict[table_schema_obj.table_name].as_retriever(similarity_top_k=2)
        relevant_nodes = vector_retriever.retrieve(query_str)
        if len(relevant_nodes) > 0:
            table_row_context = "\nHere are some relevant example rows (values in the same order as columns above)\n"
            for node in relevant_nodes:
                table_row_context += str(node.get_content()) + "\n"
            table_info += table_row_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)

table_parser_component = FnComponent(fn=get_table_context_and_rows_str)

llm = OpenAI(model="gpt-3.5-turbo")

def run_with_chat_history(query: str, chat_history: List[Dict[str, str]]):
    text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(dialect=engine.dialect.name)

    # add conversation history to the text2sql prompt
    pattern = r"(Question:)"
    insert_text = "Today's Day is 14th October 2023 - keep that in mind if the user asks date related questions.\n\nTake into account the conversation history: {chat_history}\n\n"
    text2sql_prompt.template = re.sub(pattern, insert_text + r"\1", text2sql_prompt.template)
    # remove latest q from chat history
    pattern = r"(.*)(User:.*?)(\n\s*\n)"
    text2sql_prompt.template = re.sub(pattern, r"\1", text2sql_prompt.template, count=1, flags=re.DOTALL).strip()

    text2sql_prompt = text2sql_prompt.partial_format(chat_history=format_chat_history(chat_history))

    qp.add_modules({
        "input": InputComponent(),
        "table_retriever": obj_retriever,
        "table_output_parser": table_parser_component,
        "text2sql_prompt": text2sql_prompt,
        "text2sql_llm": llm,
        "sql_output_parser": sql_parser_component,
        "sql_retriever": sql_retriever,
        "response_synthesis_prompt": response_synthesis_prompt,
        "response_synthesis_llm": llm,
    })
    qp.add_link("input", "table_retriever")
    qp.add_link("input", "table_output_parser", dest_key="query_str")
    qp.add_link("table_retriever", "table_output_parser", dest_key="table_schema_objs")
    qp.add_link("input", "text2sql_prompt", dest_key="query_str")
    qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
    qp.add_chain(["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"])
    qp.add_link("sql_output_parser", "response_synthesis_prompt", dest_key="sql_query")
    qp.add_link("sql_retriever", "response_synthesis_prompt", dest_key="context_str")
    qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
    qp.add_link("response_synthesis_prompt", "response_synthesis_llm")
    response = qp.run(query=query)

    return response

################## Streamlit UI ##################

st.set_page_config(page_title="Your Finance Assistant")

st.markdown("""
    <style>
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .assistant {
        background-color: #f1f1f1;
        text-align: left;
    }
    .user {
        background-color: #a1c3d1;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("💰 Your Finance Assistant")

with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Ask anything about your transaction history.
    2. You can speak or type your queries.
    3. Read/Listen to the AI responses.
    """)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "AI", "content": "Hello! Ask me anything about your transaction history"},
    ]

def render_message(message: Dict[str, str]) -> None:
    """
    Render audio chat message
    """
    role = "assistant" if message["role"] == "AI" else "user"
    with st.chat_message(role):
        st.markdown(message["content"], unsafe_allow_html=True)
        # prevent audio for non-text messages
        if re.search("[a-zA-Z]", message["content"]):
            msg_tts = gTTS(message["content"], lang="en")
            audio_file = f"{role}_msg.mp3"
            msg_tts.save(audio_file)
            st.audio(audio_file)

for message in st.session_state.chat_history:
    render_message(message)

pipe = pipeline(model="openai/whisper-small")

def check_valid_prompt(user_text_query: str) -> str:
    """
    Check if the user query is valid and related to finance transactions.
    """
    check_prompt = f"Does the below user query look correct and related to finance transactions? Answer with 'Yes' or 'No':\n\n{user_text_query}"
    gpt4omini = gpt_client()
    response = gpt4omini.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": check_prompt}]
    )
    llm_check_response = response.choices[0].message.content
    return llm_check_response

INVALID_MSG_STR = "Hi, I am your Personal Finance Assistant. Please ask me questions related to your transactions."

############### Audio input ###############
user_audio_query = mic_recorder(start_prompt="Record audio ⏺️", stop_prompt="Stop audio ⏹️", key="recorder")
if user_audio_query is not None:
    with st.spinner("Processing your query..."):
        user_query = pipe(user_audio_query["bytes"])["text"]

    st.session_state.chat_history.append({"role": "Human", "content": user_query})
    render_message({"role": "Human", "content": user_query})
    if user_query and user_query.strip():
        llm_check_response = check_valid_prompt(user_query)
        if 'yes' in llm_check_response.lower():
            with st.spinner("Looking for answers..."):
                response = run_with_chat_history(user_query, st.session_state.chat_history)
                response_content = response.message.content if isinstance(response, ChatResponse) else str(response)

            st.session_state.chat_history.append({"role": "AI", "content": response_content})
            render_message({"role": "AI", "content": response_content})
        else:
            st.session_state.chat_history.append({"role": "AI", "content": INVALID_MSG_STR})
            render_message({"role": "AI", "content": INVALID_MSG_STR})

############### Text input ###############
user_text_query = st.chat_input("Or type your question here:")
if user_text_query:
    st.session_state.chat_history.append({"role": "Human", "content": user_text_query})
    render_message({"role": "Human", "content": user_text_query})

    llm_check_response = check_valid_prompt(user_text_query)
    if "yes" in llm_check_response.lower():
        with st.spinner("Looking for answers..."):
            response = run_with_chat_history(user_text_query, st.session_state.chat_history)
            response_content = response.message.content if isinstance(response, ChatResponse) else str(response)

        st.session_state.chat_history.append({"role": "AI", "content": response_content})
        render_message({"role": "AI", "content": response_content})
    else:
        st.session_state.chat_history.append({"role": "AI", "content": INVALID_MSG_STR})
        render_message({"role": "AI", "content": INVALID_MSG_STR})
