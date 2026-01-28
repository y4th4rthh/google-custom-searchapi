from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from google.generativeai import configure, GenerativeModel
from typing import Optional
from gtts import gTTS
import speech_recognition as sr
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
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client["neuraai"]
chats_collection = db["chats"]
users_collection = db["users"]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://neura-ai.netlify.app", "http://localhost:3000", "http://localhost:5173", "https://neura-explore-ai.netlify.app/","https://neura-explore-ai.netlify.app",
                   "https://neura-share.netlify.app","https://dev-neura-ai.netlify.app" ,"https://admin-neura.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/audio_responses", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


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
    },
    "neura.swift1.o": {
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "tts_speed": False,
        "max_tokens": 1000,
        "temperature": 0.7
    },
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
        chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies user queries as either CURRENT or GENERAL based on their content."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                max_tokens=2000,
                temperature=0.9
            )
        response = chat_completion.choices[0].message.content
        print(response)
    
        answer = "CURRENT"

        return "CURRENT" if "CURRENT" in answer else "GENERAL"
    except Exception as e:
        print(f"❌ Classification error: {e}")
        # fallback: default to GENERAL to avoid wasting Google API
        return "CURRENT"

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
    print(results)
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
Focus on useful, factual, up-to-date information. Keep it short and clear (maximum 1–2 paragraphs).
Use emojis naturally where they help emphasize a point, make instructions clearer.

Output format (STRICT):
Paragraph 1
\u200b
Paragraph 2
\u200b

Search Results:
{text_block}
    """

    try:
        chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes Google search results."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                max_tokens=2000,
                temperature=0.9
            )
        response = chat_completion.choices[0].message.content
        return response.strip()
    except Exception as e:
        print(f"❌ Gemini summarization error: {e}")
        return "Summarization Failed! Your daily quota has expired. Please switch to another model or try later :("


    
async def google_ai_answer(query):
    print(f"🔍 Searching Google for: {query}")
    search_data = await google_search(query)
    summary = await summarize_google_results(query, search_data)
    print(summary)

    formatted_urls = "\n\n".join([f"- {d['link']}" for d in search_data]) if search_data else "No links found."
    # return f"📝 **Summary:**\n\n {" "} \n\n{summary}\n\n {" "} \n\n🔗 **Sources:**\n\n {" "} \n\n{formatted_urls}"
    return (
    "📝 **Summary:**\n\n"
    "\u200b\n\n"
    f"{summary}\n\n"
    "\u200b\n\n"
    "🔗 **Sources:**\n\n"
    "\u200b\n\n"
    f"{formatted_urls}"
    )


def create_tts_with_retry(text, filepath, max_retries=3):
    """Create TTS with retry logic and exponential backoff"""
    for attempt in range(max_retries):
        try:
            # Add jitter to prevent thundering herd
            if attempt > 0:
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)

            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filepath)
            return True

        except Exception as e:
            print(f"TTS attempt {attempt + 1} failed: {e}")
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt == max_retries - 1:
                    print("Max retries reached for TTS")
                    return False
                continue
            else:
                # For non-rate-limit errors, don't retry
                print(f"Non-rate-limit TTS error: {e}")
                return False

    return False



EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F" 
    "\U0001F300-\U0001F5FF"  
    "\U0001F680-\U0001F6FF"  
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

def sanitize_text_for_tts(text: str) -> str:
    """
    Remove Markdown symbols and code snippets from the text before TTS.
    """
    # Remove code blocks (```...```)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Remove inline code (`...`)
    text = re.sub(r"`[^`]+`", "", text)

    # Remove markdown links but keep the text part [text](url)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove bold/italic markers (**text**, __text__, *text*, _text_)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)

    # Remove markdown headers (# Header)
    text = re.sub(r"#+\s*", "", text)

    # Remove remaining symbols like > or -
    text = re.sub(r"[>`\-]+", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    text = EMOJI_PATTERN.sub(" ", text)

    return text.strip()



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

        # Generate unique filename
        filename = f"{uuid.uuid4()}.mp3"
        filepath = f"static/audio_responses/{filename}"

        # Create audio response with retry logic
        audio_url = None
        if req.model == "neura.swift1.o":
            print("AI RES", ai_response)
            sanitized_text = sanitize_text_for_tts(ai_response)
            tts_success = create_tts_with_retry(sanitized_text, filepath)

            if tts_success:
                audio_url = f"/static/audio_responses/{filename}"
            else:
                print("TTS failed after retries - returning text only")

        
        if audio_url:
            response_data = {"text": ai_response, "session_id": session_id, "audio_url": audio_url}
        else:
            response_data = {"text": ai_response, "session_id": session_id}

        return response_data

    except Exception as e:
        response_data = {"text": "Your daily quota has expired. Please switch to another model or try later :(", "session_id": session_id}
        return response_data

@app.get("/ping")
async def ping():
    """Keep-alive / health check endpoint"""
    return {"status": "ok"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
