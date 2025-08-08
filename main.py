from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import requests
from PyPDF2 import PdfReader
import os
from sentence_transformers import SentenceTransformer, util
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the sentence transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class HackrxRequest(BaseModel):
    documents: str
    questions: List[str]

def download_and_extract_text(document_url):
    """
    Downloads the PDF from the URL and extracts its text content.
    """
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
    """
    Full pipeline: download document, extract text, chunk it, embed, retrieve, and return answers.
    """
    text = download_and_extract_text(document_url)
    
    # Naive chunking (can be improved)
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Embed all chunks
    chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)

    answers = []
    for question in questions:
        question_embedding = embedding_model.encode(question, convert_to_tensor=True)
        scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
        best_idx = scores.argmax()
        best_chunk = chunks[best_idx]
        answer = f"Relevant section: {best_chunk[:300]}..."
        answers.append(answer)

    return answers

@app.post("/hackrx/run")
async def hackrx_run(data: HackrxRequest):
    """
    POST endpoint to receive document URL and questions, then return answers.
    """
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
