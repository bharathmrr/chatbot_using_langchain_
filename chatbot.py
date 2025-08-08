from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
import ollama

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
    response = llm.chat(model=model, messages=formatted)
    ai_reply = response["message"]["content"]
    messages.append(AIMessage(content=ai_reply))
    return {"messages": messages}

graph = StateGraph(ChatState)
graph.add_node("chat_node", chatnode)
graph.set_entry_point("chat_node")
graph.set_finish_point("chat_node")
chatbot = graph.compile()
while True:
    user_message=input("TYpe here: ")
    print('user: '+str(user_message))
    if user_message.strip().lower() in ['exit','quit','byee']:
        break
    resp=chatbot.invoke({'messages':[HumanMessage(content=user_message)]})

    print('AI',resp['messages'][-1].content)
    
