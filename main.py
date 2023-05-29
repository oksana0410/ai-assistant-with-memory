import os
import sys
import chromadb
import streamlit as st
from dotenv import load_dotenv
from uuid import uuid4
from chromadb.config import Settings
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate
)
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

project_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
persist_directory = os.path.join(project_directory, "chromadb/")

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory
))
embeddings = OpenAIEmbeddings()

st.title("🦜🔗 Your AI Assistant")

# Initialize collection
collection = client.get_or_create_collection("history")

system_message = """
This is a friendly conversation between a human and an AI.
The AI is designed to be talkative and provide detailed responses based on its context.
"""
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
chain = LLMChain(llm=chat, prompt=prompt)


# Session state functions
def init_session_state():
    return {
        "history": [],
        "conversation": [],
        "pagination_index": 0,
        "something": ""
    }


def get_session_state():
    return st.session_state.setdefault("session_state", init_session_state())


session_state = get_session_state()

# Conversation input form
with st.form("input_form"):
    input_text = st.text_input("Write your prompt:", key="input_text")
    submit_button = st.form_submit_button(label="Submit")

if submit_button and input_text:
    response = chain.run(input=input_text, history=session_state["history"])
    human_message = HumanMessage(content=input_text)
    ai_message = AIMessage(content=response)

    session_state["history"].extend([human_message, ai_message])
    session_state["conversation"].insert(0, f"💬  {response}\n")
    session_state["conversation"].insert(0, f"❓  {input_text}\n")

    full_conversation = "\n".join(session_state["conversation"])
    embeddings_full_conversation = embeddings.embed_documents([full_conversation])[0]

    document_id = str(uuid4())
    metadata = {"type": "conversation"}
    document = full_conversation

    collection.add(
        ids=[document_id],
        embeddings=[embeddings_full_conversation],
        metadatas=[metadata],
        documents=[document]
    )

if st.button("Reset"):
    session_state["conversation"] = []

# Pagination and conversation display
num_messages = len(session_state["conversation"])
messages_per_page = 6
num_pages = (num_messages + messages_per_page - 1) // messages_per_page

st.header("Conversation History")

start_index = session_state["pagination_index"] * messages_per_page
end_index = min(start_index + messages_per_page, num_messages)

for message in session_state["conversation"][start_index:end_index]:
    st.write(message)

if num_pages > 1:
    col1, col2, col3 = st.columns(3)
    if col2.button("Previous", key="previous_button") and session_state["pagination_index"] > 0:
        session_state["pagination_index"] -= 1
    col2.write(f"Page {session_state['pagination_index'] + 1} of {num_pages}")
    if col2.button("Next", key="next_button") and session_state["pagination_index"] < num_pages - 1:
        session_state["pagination_index"] += 1
