# medichatbot
# Medical Chatbot using RAG + Local LLM (Gemma via Ollama)

An AI-powered medical assistant chatbot that answers user queries by retrieving context from uploaded medical documents (PDFs) and responding using a local large language model (LLM) — **Gemma** via **Ollama** — with a RAG (Retrieval-Augmented Generation) pipeline.


## Features

-  Upload and parse medical PDFs using LangChain
-  Semantic search using HuggingFace Embeddings
-  Vector storage and similarity search using **Pinecone**
-  Local LLM responses using **Gemma (2B)** with **Ollama**
-  Natural language Q&A using context-aware prompts
-  Fully offline LLM setup (no API usage for generation)


##  Tech Stack

| Technology        | Role                          |
|-------------------|-------------------------------|
| Python            | Core Programming Language     |
| LangChain         | RAG pipeline & document chaining |
| HuggingFace       | Embeddings (sentence-transformers) |
| Pinecone          | Vector DB for similarity search |
| Ollama            | Runs local LLMs like Gemma    |
| Gemma 2B          | Local LLM model               |
| Streamlit (optional) | Frontend UI               |


##  Project Structure

medical-chatbot/
│
├── Data/ # Medical PDFs for ingestion
├── src/
│ ├── helper.py # PDF loader, text splitter, embeddings
│ └── init.py # RAG setup and pipeline
│
├── .env # API keys and environment config
├── store_index.py # Script to ingest and store vector data
├── chatbot.py # Main chatbot logic using RAG
├── README.md # Project documentation



## ⚙ Setup Instructions

### 1. Clone the Repository

git clone https://github.com/your-username/medical-chatbot.git
cd medical-chatbot
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Setup Environment Variables
Create a .env file with your Pinecone API key and environment:

ini
Copy
Edit
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env
4. Install Ollama & Pull Gemma
Install Ollama and pull the Gemma model:

bash
Copy
Edit
ollama pull gemma:2b
5. Run the Indexing Script
bash
Copy
Edit
python store_index.py
6. Run the Chatbot
bash
Copy
Edit
python chatbot.py
Example Prompt
text
Copy
Edit
Q: What is Acne and how is it treated?
The model will retrieve related context from your medical PDFs and generate an answer using the local LLM.

