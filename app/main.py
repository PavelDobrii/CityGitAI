from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import os
import wikipedia
import subprocess
import requests
from bs4 import BeautifulSoup
from langdetect import detect

# Если используешь Ollama:
import requests as req
# Если используешь OpenAI, раскомментируй:
# import openai

# Если нужен перевод:
from googletrans import Translator
translator = Translator()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/output")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "en")
DEFAULT_VOICE_EN = os.getenv("DEFAULT_VOICE_EN", "en/en_US/amy/medium/en_US-amy-medium.onnx")
DEFAULT_VOICE_RU = os.getenv("DEFAULT_VOICE_RU", "ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class StoryRequest(BaseModel):
    topic: str
    style: str = "neutral"
    lang: str | None = None  # "en" или "ru", auto-detected если не указано
    voice: str | None = None  # путь к модели голоса Piper

def get_wikipedia_summary(topic, lang='en'):
    wikipedia.set_lang(lang)
    try:
        return wikipedia.summary(topic, sentences=5)
    except Exception:
        return ""

def get_wikivoyage_intro(topic, lang='en'):
    url = f"https://{lang}.wikivoyage.org/wiki/{topic.replace(' ', '_')}"
    resp = requests.get(url)
    if resp.status_code == 200:
        soup = BeautifulSoup(resp.text, "html.parser")
        paras = soup.select('div.mw-parser-output > p')
        for p in paras:
            text = p.get_text(strip=True)
            if len(text) > 100:
                return text
    return ""

def get_osm_description(topic, lang='en'):
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": topic, "format": "json", "limit": 1, "accept-language": lang},
            headers={"User-Agent": "CityGitAI"}
        )
        if resp.status_code == 200:
            data = resp.json()
            if data:
                place = data[0]
                return f"{topic} is a {place.get('type','place')} located at {place.get('display_name','')}"
    except Exception:
        pass
    return ""

def ollama_generate(prompt, model="llama3:8b"):
    response = req.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt},
        stream=True  # чтобы читать построчно
    )
    result = ""
    for line in response.iter_lines():
        if line:
            try:
                json_line = line.decode("utf-8")
                data = __import__("json").loads(json_line)
                if "response" in data:
                    result += data["response"]
            except Exception as e:
                continue  # пропусти плохие строки
    return result


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate_story(req: StoryRequest):
    # 0. Определяем язык, если не указан
    if not req.lang:
        try:
            req.lang = detect(req.topic)
        except Exception:
            req.lang = DEFAULT_LANG
        if req.lang not in {"ru", "en"}:
            req.lang = DEFAULT_LANG

    # 1. Сбор фактов
    facts = []
    facts.append(get_wikipedia_summary(req.topic, req.lang))
    facts.append(get_wikivoyage_intro(req.topic, req.lang))
    facts.append(get_osm_description(req.topic, req.lang))
    facts_text = "\n\n".join([f for f in facts if f])

    # 2. Генерируем план истории через LLM
    plan_prompt = (
        f"As a talented travel storyteller, create a detailed, intriguing story plan for an article about {req.topic}. "
        "Include an introduction, key moments, plot twists, dialogues, little-known facts, and a strong ending."
        f"\nFacts:\n{facts_text}\nStyle: {req.style}"
    )
    plan = ollama_generate(plan_prompt)

    # 3. Генерируем саму историю по плану (больше слов, эмоций, диалогов)
    story_prompt = (
        f"Now write the full article (at least 700 words) based on this plan and the facts below. "
        "Make it vivid, emotional, immersive, with dialogues and urban legends, humor and unusual perspectives. "
        f"\nPlan:\n{plan}\nFacts:\n{facts_text}\nStyle: {req.style}"
    )
    english_text = ollama_generate(story_prompt)

    # 4. Переводим на русский, если нужно
    if req.lang == "ru":
        russian_text = translator.translate(english_text, src='en', dest='ru').text
        final_text = russian_text
    else:
        final_text = english_text

    # 5. Сохраняем историю как Markdown
    story_id = str(uuid.uuid4())
    story_path = os.path.join(OUTPUT_DIR, f"{story_id}.md")
    with open(story_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    # 6. Озвучка Piper
    data_dir = "/piper/models/piper-voices"
    if req.voice:
        model = req.voice
        if not os.path.isabs(model):
            model = os.path.join(data_dir, model)
    else:
        if req.lang == "ru":
            model = os.path.join(data_dir, DEFAULT_VOICE_RU)
        else:
            model = os.path.join(data_dir, DEFAULT_VOICE_EN)
    audio_path = os.path.join(OUTPUT_DIR, f"{story_id}.wav")

    try:
        result = subprocess.run(
            [
                "piper",
                "--model", model,
                "--data-dir", data_dir,
                "--output_file", audio_path
            ],
            input=final_text,
            encoding="utf-8",
            check=True,
            capture_output=True
        )
    except Exception as e:
        return {
            "error": "TTS failed",
            "details": str(e),
            "stderr": getattr(e, 'stderr', ''),
            "stdout": getattr(e, 'stdout', '')
        }

    return {
        "story_id": story_id,
        "markdown_path": story_path,
        "audio_path": audio_path,
        "plan": plan[:800],  # Можно вернуть план для интереса
        "fact_excerpt": facts_text[:800]
    }
