import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENROUTER_API_KEY")

resp = requests.get("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {key}"})
if resp.status_code == 200:
    models = resp.json()['data']
    free_vision = [m['id'] for m in models if 'free' in m['id'] and ('vision' in m['id'] or 'gemini' in m['id'] or 'llama-3.2' in m['id'])]
    print("Free Vision Models:", free_vision)
else:
    print("Error:", resp.text)
