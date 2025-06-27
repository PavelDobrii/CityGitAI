from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import os
import wikipedia
from googletrans import Translator
import openai
import json
import requests

app = FastAPI()

OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

wikipedia.set_lang("en")
translator = Translator()
openai.api_key = os.getenv("OPENAI_API_KEY")  # можно не использовать

class StoryRequest(BaseModel):
    topic: str
    style: str = "neutral"
    translate_to_ru: bool = True
    use_gpt: bool = False

@app.post("/generate")
async def generate_story(req: StoryRequest):
    print(f"[INFO] Generating story for topic: {req.topic}, style: {req.style}, translate: {req.translate_to_ru}, use_gpt: {req.use_gpt}")

    try:
        wiki_summary = wikipedia.summary(req.topic, sentences=5)
        print("[INFO] Wikipedia summary fetched successfully.")
    except Exception as e:
        print(f"[ERROR] Wikipedia fetch failed: {e}")
        return {"error": "Wikipedia fetch failed", "details": str(e)}

    prompt = f"""
Use the following facts to write a short story in Markdown (300-400 words) about: \"{req.topic}\"

Facts:
{wiki_summary}

Style: {req.style}
Include a short fun fact and two source links at the end.
"""
    print("[DEBUG] Prompt:")
    print(prompt)

    story_id = str(uuid.uuid4())
    story_path = os.path.join(OUTPUT_DIR, f"{story_id}.md")
    audio_path = os.path.join(OUTPUT_DIR, f"{story_id}.mp3")

    if req.use_gpt:
        try:
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a creative travel storyteller."},
                    {"role": "user", "content": prompt}
                ]
            )
            output = gpt_response.choices[0].message.content.strip()
            print("[INFO] GPT generation complete.")
        except Exception as e:
            print(f"[ERROR] GPT call failed: {e}")
            return {"error": "GPT call failed", "details": str(e)}
    else:
        try:
            ollama_payload = {
                "model": "llama3:8b",
                "prompt": prompt
            }
            response = requests.post("http://ollama:11434/api/generate", json=ollama_payload, timeout=120)
            output = response.text.strip()
            print("[INFO] Ollama (Llama3:8b) generation complete.")
        except Exception as e:
            print(f"[ERROR] Ollama call failed: {e}")
            return {"error": "Ollama call failed", "details": str(e)}

    if req.translate_to_ru:
        try:
            translated = translator.translate(output, src='en', dest='ru')
            output += "\n\n---\n\n" + translated.text
            print("[INFO] Translation to Russian complete.")
        except Exception as e:
            print(f"[ERROR] Translation failed: {e}")
            return {"error": "Translation failed", "details": str(e)}

    try:
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(output)
            print(f"[INFO] Markdown saved: {story_path}")
    except Exception as e:
        print(f"[ERROR] Saving markdown failed: {e}")
        return {"error": "Saving markdown failed", "details": str(e)}

    try:
        lang = "ru" if req.translate_to_ru else "en"
        tts_response = requests.post(
            "http://tts:5002/api/tts",
            json={"text": output[:1000], "speaker_id": None, "language_id": lang},
            timeout=60
        )
        if tts_response.status_code == 200:
            with open(audio_path, "wb") as f:
                f.write(tts_response.content)
            print(f"[INFO] Audio saved: {audio_path}")
        else:
            raise Exception(f"TTS server returned {tts_response.status_code}")
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")
        return {"error": "TTS failed", "details": str(e)}

    return {
        "story_id": story_id,
        "markdown_path": story_path,
        "audio_path": audio_path
    }
