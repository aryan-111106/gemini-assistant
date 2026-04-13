from src.assistant import GeminiAssistant
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

try:
    print("Initializing assistant...")
    # Initialize with a dummy key if env var is set, otherwise relies on env
    assistant = GeminiAssistant()
    
    print("Sending 'hello'...")
    # This calls send_message -> logic we want to test
    for chunk in assistant.send_message("hello", stream=True):
        print(chunk, end="", flush=True)
    print("\nDone.")
    
except Exception as e:
    print(f"\nCRASHED: {e}")
    import traceback
    traceback.print_exc()
