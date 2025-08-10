import sqlite3
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
import ollama
import os

DB_FILE = "chat_history.db"

# ==== Create DB if not exists ====
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT,
                    role TEXT,
                    content TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

# ==== DB Helpers ====
def save_message(thread_id, role, content):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)", 
              (thread_id, role, content))
    conn.commit()
    conn.close()

def load_messages(thread_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE thread_id = ? ORDER BY id ASC", (thread_id,))
    rows = c.fetchall()
    conn.close()
    messages = []
    for role, content in rows:
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages

# ==== State ====
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Ollama Client
llm = ollama.Client()
model = "gemma3:1b-it-qat"

# ==== Chat Node with streaming ====
def chatnode(state: ChatState, stream_callback=None, thread_id="1") -> ChatState:
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
        if stream_callback:
            stream_callback(token)

    # Save AI message to DB
    save_message(thread_id, "assistant", ai_reply)

    messages.append(AIMessage(content=ai_reply))
    return {"messages": messages}

# ==== LangGraph Setup ====
graph = StateGraph(ChatState)
graph.add_node("chat_node", lambda state: chatnode(state))
graph.set_entry_point("chat_node")
graph.set_finish_point("chat_node")
chatbot = graph.compile()

# ==== Public API ====
def get_response(user_prompt: str, thread_id="1", stream_callback=None):
    # Save user message
    save_message(thread_id, "user", user_prompt)

    # Load history
    history = load_messages(thread_id)

    return chatbot.invoke({"messages": history}, config={"configurable": {"thread_id": thread_id}})
