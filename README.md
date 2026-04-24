# PDF Q&A System

## Overview
This project is a Document Question Answering System that allows users to ask questions based on a PDF document.

## Features
- Extracts text from PDF
- Splits text into chunks
- Uses embeddings for similarity search
- Retrieves relevant answers from document
- Works completely offline (no APIs)

## Tech Stack
- Python
- Sentence Transformers
- FAISS
- PyPDF2

## How to Run
pip install PyPDF2 sentence-transformers faiss-cpu numpy  
python app.py

## Example
Question: What is the objective of the task?  
Answer: Build a system where a user can upload a document and ask questions about it.

## Notes
- No external APIs used  
- Runs locally  
- Focused on retrieval-based answering
