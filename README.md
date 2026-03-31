# 🤖 Agent‑Guided, Token‑Efficient RAG System  
_Intelligent Memory Routing Using a DistilBERT Classifier_

This project implements an **agent‑aware, cost‑optimized Retrieval‑Augmented Generation (RAG) pipeline** that dynamically decides whether conversational memory should be used **before** invoking a Large Language Model (LLM).

Traditional RAG systems blindly inject full chat history into every query. This architecture fixes that inefficiency by introducing a **lightweight DistilBERT classifier** that predicts whether the incoming query actually requires conversational context.

The result is a **production‑ready, token‑efficient, and scalable RAG system** that uses agents only when necessary — reducing cost while improving answer quality for context‑dependent queries.

---

## 🧠 Core Idea

**Not every query needs an agent or memory.**

Most real‑world queries are standalone. Injecting full conversation history into every LLM call wastes tokens and increases latency.

This system performs **early intent classification**:

- ✅ **YES → Memory Required**  
  Route through an **Agent + Buffer Memory**
- ❌ **NO → Memory Not Required**  
  Skip the agent and directly use **RAG**

This decision is made *before* any LLM invocation, ensuring only the minimum required context is sent downstream.

---

## 🏗️ Architecture Overview

### 1️⃣ Intent Classification (DistilBERT)

A fine‑tuned **DistilBERT classifier** analyzes the raw user query and predicts whether conversational memory is needed.

**Examples:**

- “What does the policy say about bribery?” → **No memory needed**  
- “How does it compare to the previous one?” → **Memory needed**

---

### 2️⃣ Contextual Branch — Memory Required 🧠

If the classifier predicts that context is required:

- A **LangChain‑based Agent (Red Agent)** is invoked  
- The agent retrieves the last **N** turns of conversation from **Buffer Memory**  
- Ambiguous or referential queries are rewritten into fully qualified queries  

**Example:**

> “Compare it to the first one”  
> ➡️ “Compare the Anti‑Bribery Policy with the Code of Ethics”

The refined query is then passed to the RAG pipeline.

---

### 3️⃣ Direct Branch — No Memory Needed 🚀

If the query is standalone:

- The agent is **skipped entirely**  
- No chat history is injected  
- The query is sent **directly to the vector store**

This path minimizes token usage and reduces inference latency.

---

### 4️⃣ Retrieval & Generation

For both branches:

- Relevant policy documents are retrieved from a **ChromaDB Vector Store**  
- A **GPT‑5.2‑based LLM** synthesizes the final response  
- Only the **necessary context** is passed to the model  

---

## 📂 Project Structure

```bash
.
├── workflow.py               # Main orchestration: classifier → agent → RAG
├── classification_utils.py   # DistilBERT classifier loading + prediction
├── prompt_utils.py           # Agent definition, memory tools, query rewriting
├── model_utils.py            # LLM initialization + system prompts
├── rag_utils.py              # ChromaDB vector store + similarity search
├── input_utils.py            # Session handling + chat logging
└── VectorStore.ipynb         # Ingestion: PDFs → chunks → embeddings → persistence
