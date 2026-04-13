#!/usr/bin/env python3
"""List available Gemini models."""
import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Try to read from .env
    from pathlib import Path
    env_file = Path.home() / ".openclaw/workspace/gemini-assistant/.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break

client = genai.Client(api_key=api_key)

print("Available models:")
for model in client.models.list():
    print(f"  - {model.name}")
    if hasattr(model, 'supported_generation_methods'):
        print(f"    Methods: {model.supported_generation_methods}")
