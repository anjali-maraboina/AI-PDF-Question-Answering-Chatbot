# AI PDF Question Answering Chatbot

## Project Overview
This project allows users to upload a PDF document and ask questions about its content. The chatbot retrieves relevant information from the PDF and generates answers using a local AI model.

## Features
- Upload PDF documents
- Extract text from PDF files
- Split text into chunks
- Generate text embeddings
- Retrieve relevant information using FAISS
- Answer questions using a local language model
- Display source pages

## Technologies Used
- Python
- Streamlit
- LangChain
- Hugging Face Sentence Transformers
- FAISS
- Ollama
- Llama 3.2 1B
- PyPDF

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Install Ollama and download the model:

```bash
ollama pull llama3.2:1b
```

## Run the Application

```bash
streamlit run app.py
```

Open the local URL shown in your terminal, upload a PDF, and ask questions.

## How It Works
1. The user uploads a PDF.
2. The application extracts the text.
3. The text is split into smaller chunks.
4. Hugging Face generates embeddings.
5. FAISS retrieves relevant chunks for a question.
6. Ollama uses a local language model to generate an answer.

## Note
Answer quality depends on PDF text extraction and information retrieval. Ollama and the required model must be installed locally.