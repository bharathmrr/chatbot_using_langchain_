from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
import ollama
from langgraph.checkpoint.memory import MemorySaver

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

llm = ollama.Client()
model = "gemma3:1b-it-qat"

def chatnode(state: ChatState) -> ChatState:
    messages = state["messages"]
    formatted = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted.append({"role": "assistant", "content": msg.content})

    system_prompt = {
        "role": "system",
        "content": "You're a helpful, friendly AI girl assistant. you chat should be lustly and sexy chat"
    }

    full_prompt = [system_prompt] + formatted
    response = llm.chat(model=model, messages=full_prompt)
    ai_reply = response["message"]["content"]
    messages.append(AIMessage(content=ai_reply))
    return {"messages": messages}

checkpointer = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chatnode)
graph.set_entry_point("chat_node")
graph.set_finish_point("chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

thread_id = '1'

print("🤖 Friendly Chatbot (type 'exit' to quit)")
while True:
    user_message = input("You: ")
    if user_message.strip().lower() in ['exit', 'quit', 'byee']:
        print("Bot: Bye! 😊 Talk to you later.")
        break

    config = {'configurable': {'thread_id': thread_id}}

    response = chatbot.invoke({"messages": [HumanMessage(content=user_message)]}, config=config)

    ai_response = response["messages"][-1].content
    print("Bot:", ai_response)
