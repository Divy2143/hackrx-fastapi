from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import requests
from PyPDF2 import PdfReader
import os
from sentence_transformers import SentenceTransformer, util

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to specific domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the sentence transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class HackrxRequest(BaseModel):
    documents: str
    questions: List[str]

def download_and_extract_text(document_url):
    response = requests.get(document_url)
    if response.status_code != 200:
        raise ValueError("Failed to download the document.")
    
    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    text = ""
    reader = PdfReader("temp.pdf")
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    os.remove("temp.pdf")
    return text

def process_document_and_answer_questions(document_url, questions):
    text = download_and_extract_text(document_url)
    
    # Naive chunking
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)

    answers = []
    for question in questions:
        question_embedding = embedding_model.encode(question, convert_to_tensor=True)
        scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
        best_idx = scores.argmax()
        best_chunk = chunks[best_idx]
        answers.append(f"Relevant section: {best_chunk[:300]}...")

    return answers

@app.post("/hackrx/run")
async def hackrx_run(data: HackrxRequest):
    try:
        document_url = data.documents
        questions = data.questions

        if not isinstance(document_url, str):
            raise HTTPException(status_code=400, detail="Invalid document URL.")
        if not isinstance(questions, list):
            raise HTTPException(status_code=400, detail="Questions must be a list.")

        answers = process_document_and_answer_questions(document_url, questions)
        return JSONResponse(content={"answers": answers})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
