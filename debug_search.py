"""Debug script to trace auto web search issues."""

import os
import sys

sys.path.insert(0, "src")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test")


def test_real_search():
    """Test with real API to see what's happening."""
    from src.assistant import AIAssistant
    from src.utils import web_search

    print("=" * 70)
    print("AUTO WEB SEARCH DEBUG TEST")
    print("=" * 70)

    # Test 1: Does web_search work?
    print("\n[TEST 1] Testing web_search function directly...")
    try:
        results = web_search("Python programming language", num_results=3)
        print(f"OK - web_search returned {len(results)} results")
        if results:
            print(f"  First result: {results[0]['title'][:50]}")
            print(f"  URL: {results[0]['url'][:50]}...")
    except Exception as e:
        print(f"FAIL web_search failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test 2: Test query extraction
    print("\n[TEST 2] Testing _extract_search_query...")
    assistant = AIAssistant(api_key="test", provider="openai")

    test_cases = [
        "SEARCH_FOR: weather in Tokyo",
        "Let me search. SEARCH_FOR: current news",
        "The answer is 42",
        "I need to check. SEARCH_FOR: latest iPhone price\nLet me find that.",
    ]

    for test in test_cases:
        result = assistant._extract_search_query(test)
        print(f"  Input: {test[:40]}... -> Extracted: {result}")

    # Test 3: Check if SEARCH_FOR is in the system prompt
    print("\n[TEST 3] Checking system prompt...")
    if "SEARCH_FOR:" in assistant.SYSTEM_PROMPT:
        print("OK SEARCH_FOR: is in SYSTEM_PROMPT")
    else:
        print("FAIL SEARCH_FOR: NOT in SYSTEM_PROMPT!")

    # Test 4: Check provider setup
    print("\n[TEST 4] Checking assistant configuration...")
    info = assistant.info()
    print(f"  Provider: {info['provider']}")
    print(f"  Chat Model: {info['chat_model']}")
    print(f"  Base URL: {info['base_url']}")

    # Test 5: Full integration test
    print("\n[TEST 5] Full integration test...")
    print("  (This will require a real API key)")

    api_key = (
        os.getenv("TOGETHER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )
    if not api_key or api_key == "test":
        print("  WARN No valid API key found in environment")
        print("  Set one of: TOGETHER_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY")
        return

    print("  Sending test query about current information...")
    print("  Query: 'What is the current price of Bitcoin?'")

    try:
        responses = []
        for chunk in assistant.send_message(
            "What is the current price of Bitcoin?", stream=True
        ):
            responses.append(chunk)
            # Print partial output
            if len(responses) % 5 == 0:
                print(f"  ... received {len(responses)} chunks", end="\r")

        full_response = "".join(responses)
        print(f"\n  OK Got response ({len(full_response)} chars)")
        print(f"\n  Response preview:\n{full_response[:300]}...")

        # Check if it looks like it searched
        if any(
            term in full_response.lower()
            for term in ["bitcoin", "btc", "$", "usd", "price"]
        ):
            print(
                "\n  OK Response contains Bitcoin-related terms (search likely worked)"
            )
        else:
            print("\n  WARN Response may not have searched (no Bitcoin terms found)")

    except Exception as e:
        print(f"\n  FAIL Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_real_search()
