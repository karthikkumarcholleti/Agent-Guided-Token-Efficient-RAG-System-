# Agent-Guided-Token-Efficient-RAG-System-

🤖 Agent-Guided Token-Efficient RAG System
This project implements an agent-aware, adaptive Retrieval-Augmented Generation (RAG) pipeline that dynamically determines whether conversational memory should be used before invoking a Large Language Model (LLM).

Unlike traditional RAG systems that blindly include chat history for every query, this architecture introduces a lightweight DistilBERT-based query classifier to intelligently decide if an agent with conversational memory is actually required.

By conditionally routing queries through an agent only when necessary, the system significantly reduces token usage, inference latency, and overall cost, while simultaneously improving answer quality for follow-up and context-dependent queries.

The result is a production-ready, cost-aware, and scalable RAG architecture that balances accuracy with efficiency.

🧠 Core Idea
Not every query needs an agent or conversational memory.

Most real-world user queries are standalone, yet traditional RAG pipelines unnecessarily inject full conversation history into every LLM call. This project solves that inefficiency using early intent classification.

A DistilBERT-based classifier analyzes the incoming user query and decides:

✅ YES → Route the query through an Agent + Memory
❌ NO → Skip the agent and directly use RAG
This decision is made before any LLM invocation, ensuring that only the minimum required context is sent downstream.

🏗️ Architecture Overview
The system follows a dual-path execution flow based on query intent:

1️⃣ Intent Classification
A fine-tuned DistilBERT classifier processes the raw user prompt to determine whether conversational context is required.

Examples:

"What does the policy say about bribery?" → No memory needed
"How does it compare to the previous one?" → Memory needed
2️⃣ Contextual Branch (Memory Required 🧠)
If the classifier predicts that context is required:

A LangChain-based Agent (Red Agent) is invoked
The agent retrieves the last N turns of conversation from Buffer Memory
The agent rewrites ambiguous or referential queries into fully qualified queries
Example:

"Compare it to the first one"
➡️ "Compare the Anti-Bribery Policy with the Code of Ethics"

This refined query is then passed to the RAG pipeline.

3️⃣ Direct Branch (No Memory 🚀)
If the query is standalone:

The agent is skipped entirely
No chat history is injected
The query is sent directly to the vector store
This path minimizes token usage and reduces inference latency.

4️⃣ Retrieval & Generation
For both branches:

Relevant policy documents are retrieved from a Chroma Vector Store
A generative LLM (GPT-5.2) synthesizes the final response
Only the necessary context is passed to the model
📂 Project Structure
.
├── workflow.py
│   └── Main orchestration loop: routes queries through classifier, agent, and RAG
│
├── classification_utils.py
│   └── Loads the DistilBERT classifier and predicts memory usage (Yes / No)
│
├── prompt_utils.py
│   └── Defines the Red Agent, memory recall tools, and query rewriting logic
│
├── model_utils.py
│   └── LLM initialization, system prompt design, and response generation
│
├── rag_utils.py
│   └── Vector store connection and similarity search (ChromaDB)
│
├── input_utils.py
│   └── Session handling, chat logging, and state management
│
├── VectorStore.ipynb
│   └── Ingestion pipeline: PDF loading, chunking, embeddings, and persistence

📊 Benchmark & Performance Analysis
The following metrics compare our Agent-Guided Architecture (using the DistilBERT classifier) against a Traditional RAG approach.

The files validation.ipynb and validation2.ipynb contains all the test related data.

📈 Performance Comparison
Metric	With Classification	Without Classification	Improvement
Precision	0.7896	0.6984	+13.1%
Recall	0.8798	0.7931	+10.9%
F1 Score	0.8320	0.7425	+12.1%
Sentence Similarity	0.6815	0.5908	+15.3%
Token Usage	~1,694	~2,853	💰 40% Saved
📉 Key Improvements
🔻 Efficiency: ~40% reduction in token usage.
🎯 Quality: Higher precision, recall, and semantic similarity.
🧠 Logic: Smarter agent utilization and lower LLM cost per request.
🤔 Why Agent-Guided Routing Matters?
❌ Traditional RAG Pipelines
Traditional systems always include full chat history, which scales poorly with long conversations and wastes tokens on irrelevant context, increasing both latency and cost.

✅ Agent-Guided Architecture
Our architecture invokes agents only when required. By dynamically controlling memory injection, we optimize token usage per request, making the system truly production-ready and cost-efficient.

Summary: This project demonstrates how early-stage intent classification combined with selective agent invocation dramatically improves RAG efficiency. It proves that agents are powerful—but expensive—and should be used strategically, not by default.

