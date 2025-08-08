# chatbot_using_langchain_
Developed an intelligent chatbot powered by LangChain, integrating Large Language Models (LLMs) with custom tools and memory for dynamic, context-aware conversations. The chatbot is capable of answering user queries, invoking external tools
# 🤖 LangChain Chatbot

An intelligent, modular chatbot built using [LangChain](https://www.langchain.com/) and LLMs (OpenAI or Ollama) that supports multi-turn conversations, tool usage, and dynamic memory.

## 🔍 Overview

This chatbot is designed to simulate natural, interactive conversations powered by Large Language Models (LLMs). It leverages LangChain's core components to integrate:

- Contextual memory (e.g., ConversationBufferMemory)
- Custom tools (e.g., calculator, Python executor, search API)
- LangChain Agents for decision-making and tool invocation
- Streamlit frontend for real-time interaction

## 🚀 Features

- 💬 **Conversational AI**: Handles multi-turn dialogue with context retention
- 🧠 **Memory Support**: Maintains conversation history with LangChain Memory
- 🧩 **Tool Integration**: Supports calling custom tools like:
  - Python REPL
  - Math calculator
  - Web search API
- 🛠️ **Agent Support**: Uses LangChain Agents to reason and decide which tool to call
- 🖥️ **Streamlit UI**: Simple and interactive frontend interface
- 🧱 **Local & Cloud Models**: Supports OpenAI (cloud) and Ollama (local) LLMs

## 🧰 Tech Stack

- **LangChain**
- **Python**
- **LLMs**: OpenAI GPT / Ollama models (e.g., Mistral, Gemma)
- **Frontend**: Streamlit (or Flask)
- **Optional**: FAISS / ChromaDB (for vector memory)

## 📸 Screenshots

> _Include screenshots or a demo GIF here for better engagement_

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/bharathmrr/chatbot_using_langchain_.git
cd chatbot_using_langchain_


# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
