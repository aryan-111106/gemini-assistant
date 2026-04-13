
import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("No GEMINI_API_KEY")
    exit(1)

client = genai.Client(api_key=api_key)

print("Listing models...")
try:
    for m in client.models.list():
        if "generate" in m.name or "imagen" in m.name:
            print(f"Found gen model: {m.name}")
        if "flash" in m.name:
            print(f"Found flash model: {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")

print("\nChecking image gen capability...")
try:
    # Try to find the method for image generation
    if hasattr(client.models, 'generate_image'):
        print("client.models.generate_image exists")
    elif hasattr(client.models, 'generate_images'):
        print("client.models.generate_images exists")
    else:
        print("No obvious generate_image method on client.models")
        print(dir(client.models))
except Exception as e:
    print(f"Error checking image gen: {e}")
