from langgraph.graph import StateGraph
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
import ollama
from langgraph.checkpoint.memory import MemorySaver
import speech_recognition as sr
import pyttsx3
import threading
import queue
import time

# ==== State ====
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Ollama Client
llm = ollama.Client()
model = "gemma3:1b-it-qat"

# ==== Voice Setup ====
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()
tts_queue = queue.Queue()

def listen_voice():
    """Capture voice input and return as text."""
    with sr.Microphone() as source:
        print("\n🎤 Listening... Speak now")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"🗣 You said: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"❌ Speech recognition error: {e}")
        return ""

def tts_worker():
    """Background thread to process TTS queue."""
    while True:
        text = tts_queue.get()
        if text is None:  # Stop signal
            break
        tts_engine.say(text)
        tts_engine.runAndWait()
        tts_queue.task_done()

# Start TTS thread
threading.Thread(target=tts_worker, daemon=True).start()

# ==== Chat Node with token streaming ====
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

    # Live streaming from Ollama
    ai_reply = ""
    print("\nAssistant: ", end="", flush=True)
    buffer = ""

    for chunk in llm.chat(model=model, messages=full_prompt, stream=True):
        token = chunk["message"]["content"]
        ai_reply += token
        print(token, end="", flush=True)

        buffer += token
        if len(buffer) > 20 or token.endswith(('.', '!', '?')):
            tts_queue.put(buffer.strip())
            buffer = ""

    # Flush remaining buffer
    if buffer.strip():
        tts_queue.put(buffer.strip())

    print()  # newline
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

# ==== Chat loop ====
while True:
    print("\nChoose input mode: [1] Text  [2] Voice  [exit]")
    mode = input("Mode: ").strip().lower()

    if mode in ["exit", "quit"]:
        tts_queue.put(None)  # Stop TTS thread
        break
    elif mode == "1":
        user_input = input("\nYou: ")
    elif mode == "2":
        user_input = listen_voice()
        if not user_input:
            continue
    else:
        print("❌ Invalid choice.")
        continue

    chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
