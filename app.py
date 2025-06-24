from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
import torch
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware

# Load Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load Gemini model
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Load Q&A data
with open("data.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

questions = [item["question"] for item in qa_pairs]
answers = [item["answer"] for item in qa_pairs]

# Sentence embedding model (multilingual)
embedder = SentenceTransformer("distiluse-base-multilingual-cased-v1")
question_embeddings = embedder.encode(questions, convert_to_tensor=True)

# ✅ THIS is what uvicorn needs to find!
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body schema
class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    user_question = query.text
    user_embedding = embedder.encode(user_question, convert_to_tensor=True)

    # Find closest match
    similarity_scores = util.cos_sim(user_embedding, question_embeddings)
    top_idx = torch.argmax(similarity_scores).item()
    context = answers[top_idx]

    # Prompt Gemini
    prompt = f"""
    மாணவர் கேட்ட கேள்வி: "{user_question}"
    கீழ்காணும் விளக்கத்தின் அடிப்படையில் பதில் அளி:
    "{context}"
    """

    try:
        response = model.generate_content(prompt)
        return {"answer": response.text}
    except Exception as e:
        return {"error": str(e)}
