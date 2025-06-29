from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import os
import wikipedia
import subprocess
import requests
from bs4 import BeautifulSoup

# Если используешь Ollama:
import requests as req
# Если используешь OpenAI, раскомментируй:
# import openai

# Если нужен перевод:
from googletrans import Translator
translator = Translator()

app = FastAPI()
OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class StoryRequest(BaseModel):
    topic: str
    style: str = "neutral"
    lang: str = "en"  # "en" или "ru"

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

def ollama_generate(prompt, model="llama3:8b"):
    response = req.post(
        "http://ollama:11434/api/generate",
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


@app.post("/generate")
async def generate_story(req: StoryRequest):
    # 1. Сбор фактов
    facts = []
    facts.append(get_wikipedia_summary(req.topic, req.lang))
    facts.append(get_wikivoyage_intro(req.topic, req.lang))
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

    # 6. Озвучка Piper (укажи путь к модели и data-dir по своей структуре!)
    if req.lang == "ru":
        model = "/piper/models/piper-voices/ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium.onnx"
    else:
        model = "/piper/models/piper-voices/en/en_US/amy/medium/en_US-amy-medium.onnx"
    data_dir = "/piper/models/piper-voices"
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
