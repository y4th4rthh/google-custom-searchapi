from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel
from typing import Optional
import uuid
import os
import tempfile
import datetime
import io
import httpx
import time
import re
import random
from system_prompts import SYSTEM_PROMPT

load_dotenv()
app = FastAPI()

MONGO_URI = os.getenv("MONGO_URI")
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client["neuraai"]
chats_collection = db["chats"]
users_collection = db["users"]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://neura-ai.netlify.app", "http://localhost:3000", "http://localhost:5173",
                   "https://neura-share.netlify.app", "https://admin-neura.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str
    model: str = "neura.essence1.o"
    user_id: Optional[str] = None
    sessionId: Optional[str] = None
    incognito: bool


MODEL_CONFIG = {
    "neura.essence1.o": {
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "tts_speed": False,
        "max_tokens": 1000,
        "temperature": 0.7
    }
}

# Configure Gemini
configure(api_key=os.getenv("GEMINI_API_KEY"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")


async def classify_query(query: str) -> str:
    """
    Ask Gemini if this query is related to current events or general knowledge.
    Returns either 'CURRENT' or 'GENERAL'.
    """
    prompt = f"""
Classify the following user query:

Query: "{query}"

Answer with only one word: 'CURRENT' if it refers to something happening now or after 2021
(e.g. political events, celebrity news, recent technology, ongoing wars, live matches, trending topics).

Or 'GENERAL' if it's general knowledge, concepts, history, math, science, facts before 2021.
    """
    try:
        model = GenerativeModel(model_name="gemini-2.5-flash",
                        system_instruction="You are a helpful assistant that classifies user queries as either CURRENT or GENERAL based on their content.")
        response = model.generate_content(prompt)
        answer = response.text.strip().upper()
        return "CURRENT" if "CURRENT" in answer else "GENERAL"
    except Exception as e:
        print(f"❌ Classification error: {e}")
        # fallback: default to GENERAL to avoid wasting Google API
        return "GENERAL"

async def google_search(query: str, num_results: int = 5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        data = resp.json()

    results = []
    for item in data.get("items", []):
        results.append({
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "link": item.get("link")
        })
    return results


async def summarize_google_results(query, search_data):
    if not search_data:
        return "No search results found."

    text_block = "\n\n".join(
        f"Title: {d['title']}\nSnippet: {d['snippet']}\n"
        for d in search_data
    )

    prompt = f"""
Summarize the following Google search results for the query "{query}".
Focus on useful, factual, up-to-date information. Keep it short and clear.

Search Results:
{text_block}
    """

    try:
        model = GenerativeModel(model_name="gemini-2.5-flash",
                        system_instruction="You are a helpful assistant that summarizes Google search results.")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Gemini summarization error: {e}")
        return "Summarization failed. Please try again."


    
async def google_ai_answer(query):
    print(f"🔍 Searching Google for: {query}")
    search_data = await google_search(query)
    summary = await summarize_google_results(query, search_data)

    formatted_urls = "\n".join([d["link"] for d in search_data]) if search_data else "No links found."
    return f"📝 **Summary:**\n\n{summary}\n\n🔗 **Sources:**\n{formatted_urls}"


@app.post("/search")
async def chat(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided")

    if req.model not in MODEL_CONFIG:
        raise HTTPException(
            status_code=400, detail=f"Invalid model: {req.model}")

    try:
        session_id = req.sessionId or str(uuid.uuid4())
        userId = req.user_id
        config = MODEL_CONFIG[req.model]
        chatvalue=""
        
        
        category = await classify_query(req.text)
        print(f"🧠 Query classified as: {category}")

        if category == "CURRENT":
           ai_response = await google_ai_answer(query = req.text)
        else:
           if req.sessionId:
             chatdatas = chats_collection.find({"session_id":req.sessionId})
             async for chatval in chatdatas:
                chatvalue+=chatval["user_text"]+chatval["ai_response"]
             sysPrompt = SYSTEM_PROMPT + f"\n\nOnly refer to the following previous chat history if the user's current input is clearly related to it. If it's a new or unrelated query, you may ignore this context.\nPrevious chat history:\n{chatvalue}\n"
           else:
             sysPrompt = SYSTEM_PROMPT

           model = GenerativeModel(model_name="gemini-2.5-flash",
                        system_instruction=sysPrompt)
           response = model.generate_content(req.text)
           ai_response = response.text

       
        print(req.incognito)
        if req.incognito == False:
            chat_doc = {
                "session_id": session_id,
                "timestamp": datetime.datetime.utcnow(),
                "user_text": req.text,
                "user_id": userId,
                "model": req.model,
                "ai_response": ai_response
            }
            await chats_collection.insert_one(chat_doc)

        response_data = {"text": ai_response, "session_id": session_id}

        return response_data

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Chat processing error: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
