# 🤖 Agent‑Guided Token‑Efficient RAG System  
_Intelligent Memory Routing Using a DistilBERT Classifier_

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![DistilBERT](https://img.shields.io/badge/Model-DistilBERT-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)
![RAG](https://img.shields.io/badge/Architecture-RAG-red)

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
  Skip the agent and route the query directly to the **RAG pipeline**.
  
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
- A **generative LLM** synthesizes the final response 
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


```


## ⚙️ Tech Stack

- Python
- DistilBERT
- LangChain
- ChromaDB
- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Jupyter Notebook

---

## 🌟 Use Cases

- Enterprise document assistants
- Policy and compliance Q&A systems
- Multi-turn conversational retrieval systems
- Cost-optimized production RAG pipelines
- Context-aware internal knowledge assistants
- Intelligent memory-aware chatbot applications

---
## 📊 Benchmark & Performance Analysis

All evaluation details are available in:

- `validation.ipynb`  
- `validation2.ipynb`  

### 📈 Performance Comparison

| Metric              | With Classification | Without Classification | Improvement    |
|---------------------|---------------------|------------------------|----------------|
| Precision           | 0.7896              | 0.6984                 | **+13.1%**     |
| Recall              | 0.8798              | 0.7931                 | **+10.9%**     |
| F1 Score            | 0.8320              | 0.7425                 | **+12.1%**     |
| Sentence Similarity | 0.6815              | 0.5908                 | **+15.3%**     |
| Token Usage         | ~1,694              | ~2,853                 | **💰 ~40% Saved** |

---

## 📉 Key Improvements

- 🔻 **Efficiency:** ~40% reduction in token usage  
- 🎯 **Quality:** Higher precision, recall, and semantic similarity  
- 🧠 **Smarter Routing:** Classifier‑guided agent utilization  
- ⚡ **Latency:** Faster responses due to reduced context size  

---

## 🤔 Why Agent‑Guided Routing Matters

### ❌ Traditional RAG Pipelines

- Always include full chat history  
- Token usage grows with conversation length  
- Higher latency and cost  
- Often injects irrelevant context into the LLM  

### ✅ Agent‑Guided Architecture

- Invokes agents **only when required**  
- Dynamically controls memory injection  
- Optimizes token usage per request  
- More scalable and **production‑ready** for real‑world workloads  

---

## 📝 Summary

This project demonstrates how **early‑stage intent classification + selective agent invocation** dramatically improves RAG efficiency.

It shows that agents are powerful — but expensive — and should be used **strategically**, not by default.  
The architecture balances **accuracy, cost, and scalability**, making it suitable for enterprise‑grade RAG deployments.

