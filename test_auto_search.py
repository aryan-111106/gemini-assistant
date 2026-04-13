"""Test script to debug auto web search functionality."""

import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, "src")


def test_auto_search_flow():
    """Test the complete auto search flow."""
    print("Testing auto web search flow...")
    print("=" * 60)

    # Setup mock responses
    # First response: AI wants to search
    mock_response_search = Mock()
    mock_response_search.choices = [Mock()]
    mock_response_search.choices[0].delta.content = "SEARCH_FOR: weather in Tokyo today"
    mock_response_search.choices[
        0
    ].message.content = "SEARCH_FOR: weather in Tokyo today"

    # Second response: AI with search results
    mock_response_answer = Mock()
    mock_response_answer.choices = [Mock()]
    mock_response_answer.choices[
        0
    ].delta.content = "The weather in Tokyo today is sunny with a high of 25°C."
    mock_response_answer.choices[
        0
    ].message.content = "The weather in Tokyo today is sunny with a high of 25°C."

    # Create mock client
    mock_client = Mock()

    # Track what happens
    call_count = [0]

    def mock_create(*args, **kwargs):
        call_count[0] += 1
        print(f"\n[DEBUG] API call #{call_count[0]}")
        print(f"[DEBUG] Stream mode: {kwargs.get('stream', False)}")
        if "messages" in kwargs:
            last_msg = kwargs["messages"][-1]["content"][:100]
            print(f"[DEBUG] Last message: {last_msg}...")

        if call_count[0] == 1:
            if kwargs.get("stream"):
                return [mock_response_search]
            else:
                return mock_response_search
        else:
            if kwargs.get("stream"):
                return [mock_response_answer]
            else:
                return mock_response_answer

    mock_client.chat.completions.create = mock_create

    with patch("openai.OpenAI", return_value=mock_client):
        from src.assistant import AIAssistant

        assistant = AIAssistant(
            api_key="test", provider="openai", base_url="http://localhost:11434/v1"
        )

        print("\n[TEST] Sending message: 'What is the weather in Tokyo?'")

        # Collect response
        responses = []
        try:
            for chunk in assistant.send_message(
                "What is the weather in Tokyo?", stream=True
            ):
                responses.append(chunk)
                print(f"[CHUNK] {chunk[:50]}...")
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback

            traceback.print_exc()

        full_response = "".join(responses)
        print("\n" + "=" * 60)
        print("Full response:")
        print(full_response)
        print("=" * 60)

        # Check if web search was triggered
        from src.utils import web_search

        print("\n[TEST] Testing web_search function directly...")
        results = web_search("test query", num_results=2)
        print(f"Direct web_search returned {len(results)} results")

        if results:
            print(f"First result: {results[0]['title'][:50]}")


if __name__ == "__main__":
    test_auto_search_flow()
