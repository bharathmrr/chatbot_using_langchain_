import streamlit as st
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
import ollama
from langgraph.checkpoint.memory import MemorySaver

# ==== State ====
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Ollama Client
llm = ollama.Client()
model = "gemma3:1b-it-qat"

# ==== Chat Node with streaming ====
def chatnode(state: ChatState) -> ChatState:
    messages = state["messages"]

    # Format messages for Ollama
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted.append({"role": "assistant", "content": msg.content})

    # Add system prompt
    system_prompt = {
        "role": "system",
        "content": (
            "You're a helpful, friendly AI assistant. "
            "Keep responses conversational, human-like, and relevant to the context. "
            "Use emojis if appropriate!"
        )
    }
    full_prompt = [system_prompt] + formatted

    # Stream response from Ollama
    ai_reply = ""
    for chunk in llm.chat(model=model, messages=full_prompt, stream=True):
        token = chunk["message"]["content"]
        ai_reply += token
        st.session_state["chat_history"][-1]["content"] += token
        chat_placeholder.markdown(st.session_state["chat_history"][-1]["content"])

    messages.append(AIMessage(content=ai_reply))
    return {"messages": messages}

# ==== Memory ====
checkpointer = MemorySaver()

# ==== Graph setup ====
graph = StateGraph(ChatState)
graph.add_node("chat_node", chatnode)
graph.set_entry_point("chat_node")
graph.set_finish_point("chat_node")
chatbot = graph.compile(checkpointer=checkpointer)
config = {'configurable': {'thread_id': '1'}}

# ==== Streamlit UI ====
st.set_page_config(page_title="Ollama Chatbot", page_icon="💬", layout="centered")
st.title("💬 Ollama LangGraph Chatbot")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history
for chat in st.session_state["chat_history"]:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# User input box
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add empty placeholder for AI response
    st.session_state["chat_history"].append({"role": "assistant", "content": ""})
    with st.chat_message("assistant"):
        chat_placeholder = st.empty()

    # Call LangGraph chatbot
    chatbot.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
