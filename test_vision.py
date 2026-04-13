#!/usr/bin/env python3
"""Test vision capability."""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.assistant import AIAssistant

# Check what provider we have
print("Checking available API keys...")
for var in ['TOGETHER_API_KEY', 'OPENROUTER_API_KEY', 'GROQ_API_KEY', 'OPENAI_API_KEY']:
    val = os.getenv(var)
    print(f"  {var}: {'SET' if val else 'not set'}")

try:
    assistant = AIAssistant()
    print(f"\nUsing provider: {assistant.provider}")
    print(f"Vision model: {assistant.vision_model}")
    
    # Test with the existing test image
    if os.path.exists('test_image.png'):
        print("\nAnalyzing test_image.png...")
        result = assistant.analyze_image('test_image.png')
        print(f"Result: {result[:500]}...")
    else:
        print("\nNo test_image.png found")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
