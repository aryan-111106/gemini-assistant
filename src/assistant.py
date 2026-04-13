"""Core AI Assistant - OpenAI-compatible API for chat + vision."""

import os
import base64
import httpx
from pathlib import Path
from typing import Optional, Generator
import tempfile

from openai import OpenAI

from .memory import MemoryManager
from .voice import VoiceManager
from .utils import (
    read_file_content,
    get_file_type,
    validate_file,
    load_image,
    image_to_base64,
)


def get_wsl_host_ip() -> Optional[str]:
    """Get the Windows host IP from WSL."""
    try:
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                if line.startswith("nameserver"):
                    return line.split()[1]
    except:
        pass
    return None


class AIAssistant:
    """
    A multimodal AI assistant using OpenAI-compatible APIs.

    - Text chat: Primary provider (Together, Groq, OpenRouter, etc.)
    - Vision: OpenRouter or Ollama (local/Windows host)
    - Image generation: Together.ai FLUX or OpenAI DALL-E
    """

    SYSTEM_PROMPT = """You are JARVIS, a highly advanced AI assistant with web search capabilities.

CRITICAL INSTRUCTION - WEB SEARCH:
You have real-time web access. When the user asks about:
- Current events, news, or recent developments
- Weather, stock prices, sports scores, or time-sensitive information
- Facts that may have changed since your training data
- "What is the latest...", "Current...", "Today...", "Now..."
- Any query where you're uncertain about recency or accuracy

YOU MUST use web search by responding EXACTLY like this:
SEARCH_FOR: <specific search query>

For example:
User: "What's the weather in London?"
You: SEARCH_FOR: London weather today

User: "Who won the World Cup?"
You: SEARCH_FOR: World Cup winner 2024

The system will then fetch search results and you will receive them to formulate your answer.

Your other capabilities:
- Natural conversations with context awareness
- Image analysis and description
- Document reading (PDFs, code, text files)
- Coding, writing, research, and problem-solving

Tone: Polished, slightly British—precise, helpful, concise. Professional with occasional wit.
Style: Direct. Skip unnecessary pleasantries. Provide value."""

    # Model presets for different providers
    MODEL_PRESETS = {
        "together": {
            "base_url": "https://api.together.xyz/v1",
            "chat_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "image_model": "black-forest-labs/FLUX.1.1-pro",
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "chat_model": "meta-llama/llama-3.3-70b-instruct",
            "image_model": None,
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "chat_model": "llama-3.3-70b-versatile",
            "image_model": None,
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "chat_model": "gpt-4o",
            "image_model": "dall-e-3",
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "chat_model": "llama3.2",
            "image_model": None,
        },
    }

    # Vision models
    OPENROUTER_VISION_MODEL = "openrouter/free"
    OLLAMA_VISION_MODELS = [
        "qwen3-vl",
        "llava",
        "llama3.2-vision",
        "bakllava",
        "minicpm-v",
    ]

    def __init__(
        self,
        api_key: str = None,
        provider: str = None,
        base_url: str = None,
        chat_model: str = None,
        image_model: str = None,
        vision_model: str = None,
        openrouter_key: str = None,
        mic_device_name: str = None,
        mic_device: int = None,
    ):
        """
        Initialize the Assistant.

        Auto-detects provider from environment if not specified.
        Vision priority: OpenRouter > Ollama (Windows host or local)
        """
        # Auto-detect provider and API key
        self.provider, self.api_key = self._detect_provider(api_key, provider)

        if not self.api_key:
            raise ValueError(
                "No API key found. Set one of:\n"
                "  TOGETHER_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY\n"
                "Or run Ollama locally: ollama serve"
            )

        # Get preset config
        preset = self.MODEL_PRESETS.get(self.provider, {})

        # Configure OpenAI-compatible client for chat
        self.base_url = base_url or preset.get(
            "base_url", "https://api.together.xyz/v1"
        )
        self.chat_model = chat_model or preset.get(
            "chat_model", "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        )
        self.image_model = image_model or preset.get("image_model")

        # Main client for chat
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Ollama URL (check Windows host first, then localhost)
        self.ollama_url = self._find_ollama_url()

        # Vision client setup (priority: OpenRouter > Ollama)
        self.vision_client = None
        self.vision_model = vision_model
        self.vision_provider = None

        # Try OpenRouter first
        openrouter_key = openrouter_key or os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            self.vision_client = OpenAI(
                api_key=openrouter_key, base_url="https://openrouter.ai/api/v1"
            )
            self.vision_model = vision_model or self.OPENROUTER_VISION_MODEL
            self.vision_provider = "OpenRouter"

        # Fall back to Ollama if no OpenRouter
        if not self.vision_client and self.ollama_url:
            ollama_vision = self._get_ollama_vision_model()
            if ollama_vision:
                self.vision_client = OpenAI(
                    api_key="ollama", base_url=f"{self.ollama_url}/v1"
                )
                self.vision_model = vision_model or ollama_vision
                self.vision_provider = "Ollama"

        # Initialize managers
        self.memory = MemoryManager()

        # Get mic settings from env if not provided
        if mic_device is None:
            try:
                env_index = os.getenv("AI_MIC_INDEX")
                if env_index is not None:
                    mic_device = int(env_index)
            except (ValueError, TypeError):
                pass

        self.voice = VoiceManager(
            mic_device=mic_device, mic_device_name=mic_device_name, verbose=False
        )
        self.memory.new_conversation()
        self._chat_history: list[dict] = []

    def _find_ollama_url(self) -> Optional[str]:
        """Find Ollama URL - check Windows host (WSL) first, then localhost."""
        urls_to_try = []

        # In WSL, try Windows host first
        wsl_host = get_wsl_host_ip()
        if wsl_host:
            urls_to_try.append(f"http://{wsl_host}:11434")

        # Also try common alternatives
        urls_to_try.extend(
            [
                "http://host.docker.internal:11434",
                "http://localhost:11434",
                "http://127.0.0.1:11434",
            ]
        )

        for url in urls_to_try:
            try:
                resp = httpx.get(f"{url}/api/tags", timeout=2)
                if resp.status_code == 200:
                    return url
            except:
                continue

        return None

    def _detect_provider(
        self, api_key: str = None, provider: str = None
    ) -> tuple[str, str]:
        """Detect provider from environment or explicit config."""
        if provider and api_key:
            return provider, api_key

        providers = [
            ("together", "TOGETHER_API_KEY"),
            ("openrouter", "OPENROUTER_API_KEY"),
            ("groq", "GROQ_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
        ]

        for prov, env_var in providers:
            key = os.getenv(env_var)
            if key:
                return prov, key

        # Check for Ollama
        if self._find_ollama_url():
            return "ollama", "ollama"

        return provider or "together", api_key

    def _get_ollama_vision_model(self) -> Optional[str]:
        """Get an available vision model from Ollama."""
        if not self.ollama_url:
            return None

        try:
            resp = httpx.get(f"{self.ollama_url}/api/tags", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                available = [m["name"] for m in data.get("models", [])]

                # Check for known vision models (including cloud models)
                for model_name in available:
                    base = model_name.split(":")[0]
                    for vm in self.OLLAMA_VISION_MODELS:
                        if vm in base.lower():
                            return model_name
        except:
            pass
        return None

    def _analyze_with_vision(self, image_b64: str, prompt: str) -> str:
        """Use vision model for image analysis."""
        if not self.vision_client:
            return "❌ Image analysis not available. Set OPENROUTER_API_KEY or run Ollama with a vision model."

        try:
            response = self.vision_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Vision error: {str(e)}"

    def _extract_search_query(self, text: str) -> Optional[str]:
        """Extract search query from text if it contains SEARCH_FOR:."""
        if "SEARCH_FOR:" in text:
            parts = text.split("SEARCH_FOR:")
            if len(parts) > 1:
                query = parts[1].strip()
                return query.split("\n")[0].strip()
        return None

    def send_message(
        self, message: str, files: list[Path] = None, stream: bool = True
    ) -> Generator[str, None, None] | str:
        """Send a message to the assistant."""
        file_paths = []
        image_b64_list = []
        additional_context = []

        # Process attached files
        if files:
            for filepath in files:
                filepath = Path(filepath).expanduser().resolve()
                is_valid, error = validate_file(filepath)
                if not is_valid:
                    if stream:
                        yield f"⚠️ {error}\n"
                    continue

                file_paths.append(str(filepath))
                file_type = get_file_type(filepath)

                if file_type == "image":
                    img = load_image(filepath)
                    b64 = image_to_base64(img)
                    image_b64_list.append(b64)
                else:
                    _, content = read_file_content(filepath)
                    additional_context.append(
                        f"[Content of {filepath.name}]:\n{content}"
                    )

        # If we have images, use vision model
        if image_b64_list:
            full_prompt = message
            if additional_context:
                full_prompt += "\n\n" + "\n\n".join(additional_context)

            result = self._analyze_with_vision(image_b64_list[0], full_prompt)

            self._chat_history.append(
                {"role": "user", "content": f"[Image attached] {message}"}
            )
            self._chat_history.append({"role": "assistant", "content": result})
            self.memory.add_message("user", message, file_paths)
            self.memory.add_message("assistant", result)

            if stream:
                yield result
            else:
                return result
            return

        # Text-only: use main chat client
        full_message = message
        if additional_context:
            full_message += "\n\n" + "\n\n".join(additional_context)

        self._chat_history.append({"role": "user", "content": full_message})
        self.memory.add_message("user", message, file_paths)

        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT}
            ] + self._chat_history

            if stream:
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                    max_tokens=4096,
                    stream=True,
                )
                full_response = []
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response.append(chunk.choices[0].delta.content)
                result = "".join(full_response)
            else:
                response = self.client.chat.completions.create(
                    model=self.chat_model, messages=messages, max_tokens=4096
                )
                result = response.choices[0].message.content

            # Check if AI wants to search (before yielding anything to user)
            search_query = self._extract_search_query(result)

            # DEBUG: Print what's happening
            print(f"\n[DEBUG] === AUTO SEARCH DEBUG ===")
            print(f"[DEBUG] Full AI response: {result}")
            print(f"[DEBUG] Search query extracted: {search_query}")
            print(f"[DEBUG] ======================\n")

            if search_query:
                # Perform web search silently
                from .utils import web_search

                search_results = web_search(search_query, num_results=5)

                if search_results and len(search_results) > 0:
                    # Format search results
                    search_context = f"Web search results for '{search_query}':\n\n"
                    for i, r in enumerate(search_results[:3], 1):
                        title = r.get("title", "No title")
                        snippet = r.get("snippet", r.get("body", "No snippet"))
                        url = r.get("url", r.get("href", ""))
                        search_context += f"[{i}] {title}\n{snippet}\nSource: {url}\n\n"

                    # Feed search results back to AI with explicit instructions
                    search_prompt = f"""You have performed a web search and received these results. EXTRACT the specific information requested and provide it directly to the user. Do NOT say "you can check" or "visit these websites" - give the actual answer based on the search results below.

Original question: {message}

Web search results:
{search_context}

Based on these results, provide the specific information requested. Include numbers, prices, dates, or facts found in the results. If multiple sources show different data, mention the range or most recent figure. Be direct and factual."""

                    # Strip SEARCH_FOR: from the result before adding to history
                    clean_result = result.split("SEARCH_FOR:")[0].strip()
                    if clean_result:
                        self._chat_history.append(
                            {"role": "assistant", "content": clean_result}
                        )
                    self._chat_history.append(
                        {"role": "user", "content": search_prompt}
                    )

                    messages = [
                        {"role": "system", "content": self.SYSTEM_PROMPT}
                    ] + self._chat_history

                    # Get final answer with search results
                    if stream:
                        response = self.client.chat.completions.create(
                            model=self.chat_model,
                            messages=messages,
                            max_tokens=4096,
                            stream=True,
                        )
                        final_response = []
                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                final_response.append(chunk.choices[0].delta.content)
                                yield chunk.choices[0].delta.content
                        result = "".join(final_response)
                    else:
                        response = self.client.chat.completions.create(
                            model=self.chat_model, messages=messages, max_tokens=4096
                        )
                        result = response.choices[0].message.content
                else:
                    # No results - provide a fallback message
                    result = "I couldn't find any relevant information for that query. Please try a different search or ask me something else."
                    if stream:
                        yield result
            else:
                # No search needed - yield/return the original response
                if stream:
                    yield result

            self._chat_history.append({"role": "assistant", "content": result})
            self.memory.add_message("assistant", result)

            if not stream:
                return result

        except Exception as e:
            msg = f"❌ Error: {str(e)}"
            if stream:
                yield msg
            else:
                return msg

    def reset_chat(self):
        """Start a new conversation."""
        self._chat_history = []
        self.memory.new_conversation()

    def analyze_image(self, image_path: Path, question: str = None) -> str:
        """Analyze an image."""
        image_path = Path(image_path).expanduser().resolve()

        if not self.vision_client:
            return "❌ Image analysis not available. Set OPENROUTER_API_KEY or run Ollama with a vision model."

        try:
            img = load_image(image_path)
            b64 = image_to_base64(img)
            prompt = question or "Describe this image in detail. What do you see?"
            return self._analyze_with_vision(b64, prompt)

        except Exception as e:
            return f"❌ Error analyzing image: {str(e)}"

    def generate_image(
        self, prompt: str, output_path: Path = None, aspect_ratio: str = "1:1"
    ) -> Optional[Path]:
        """Generate an image from a text prompt."""
        if not self.image_model:
            raise RuntimeError(
                f"Image generation not available with {self.provider}.\n"
                "Use Together.ai or OpenAI for image generation."
            )

        try:
            dimensions = {
                "1:1": (1024, 1024),
                "16:9": (1792, 1024),
                "9:16": (1024, 1792),
                "4:3": (1024, 768),
                "3:4": (768, 1024),
            }
            width, height = dimensions.get(aspect_ratio, (1024, 1024))

            response = self.client.images.generate(
                model=self.image_model, prompt=prompt, n=1, size=f"{width}x{height}"
            )
            image_result = response.data[0]

            if output_path is None:
                output_path = (
                    Path(tempfile.gettempdir()) / f"ai_image_{os.urandom(4).hex()}.png"
                )
            output_path = Path(output_path)

            if hasattr(image_result, "b64_json") and image_result.b64_json:
                image_bytes = base64.b64decode(image_result.b64_json)
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
            elif hasattr(image_result, "url") and image_result.url:
                img_response = httpx.get(image_result.url)
                with open(output_path, "wb") as f:
                    f.write(img_response.content)
            else:
                raise RuntimeError("No image data in response")

            return output_path

        except Exception as e:
            raise RuntimeError(f"Image generation failed: {e}")

    def analyze_document(self, file_path: Path, question: str = None) -> str:
        """Analyze a document and optionally answer a question about it."""
        file_path = Path(file_path).expanduser().resolve()
        _, content = read_file_content(file_path)

        if question:
            prompt = f"Based on this document:\n\n{content}\n\nQuestion: {question}"
        else:
            prompt = f"Summarize this document:\n\n{content}"

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def listen_and_respond(self, speak_response: bool = True) -> tuple[str, str]:
        """Listen for voice input and respond."""
        if not self.voice.input_available:
            raise RuntimeError("Voice input not available")

        user_input = self.voice.listen()
        if not user_input:
            return None, None

        response_parts = []
        for chunk in self.send_message(user_input, stream=True):
            response_parts.append(chunk)
        response = "".join(response_parts)

        if speak_response and self.voice.output_available:
            clean = response.replace("*", "").replace("#", "").replace("`", "")
            self.voice.speak(clean[:500])

        return user_input, response

    def speak(self, text: str):
        """Speak text aloud."""
        if not self.voice.output_available:
            raise RuntimeError("Voice output not available")
        self.voice.speak(text)

    def get_conversation_history(self) -> list[dict]:
        """Get the current conversation history."""
        if self.memory.current_conversation:
            return [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in self.memory.current_conversation.messages
            ]
        return []

    def list_conversations(self) -> list[dict]:
        """List all saved conversations."""
        return self.memory.list_conversations()

    def load_conversation(self, conv_id: str) -> bool:
        """Load a previous conversation."""
        conv = self.memory.load_conversation(conv_id)
        if conv:
            self._chat_history = []
            for msg in conv.messages:
                self._chat_history.append({"role": msg.role, "content": msg.content})
            return True
        return False

    def info(self) -> dict:
        """Get assistant configuration info."""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "chat_model": self.chat_model,
            "vision": self.vision_provider or "Not available",
            "vision_model": self.vision_model,
            "image_model": self.image_model or "Not available",
            "ollama_url": self.ollama_url,
            "voice_input": self.voice.input_available,
            "voice_output": self.voice.output_available,
            "voice_tts": self.voice.tts_method,
            "voice_stt": self.voice.stt_method,
        }

    def set_vision_provider(self, provider: str, model: str = None) -> tuple[bool, str]:
        """
        Switch vision provider.

        Args:
            provider: 'openrouter' or 'ollama'
            model: Optional model override

        Returns:
            (success, message)
        """
        provider = provider.lower().strip()

        if provider == "openrouter":
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_key:
                return False, "OPENROUTER_API_KEY not set"

            self.vision_client = OpenAI(
                api_key=openrouter_key, base_url="https://openrouter.ai/api/v1"
            )
            self.vision_model = model or self.OPENROUTER_VISION_MODEL
            self.vision_provider = "OpenRouter"
            return True, f"Vision switched to OpenRouter ({self.vision_model})"

        elif provider == "ollama":
            if not self.ollama_url:
                self.ollama_url = self._find_ollama_url()

            if not self.ollama_url:
                return False, "Ollama not found. Make sure it's running."

            ollama_model = model or self._get_ollama_vision_model()
            if not ollama_model:
                available = self.list_ollama_models()
                if available:
                    return (
                        False,
                        f"No vision model found. Available: {', '.join(available[:5])}",
                    )
                return False, "No models found in Ollama."

            self.vision_client = OpenAI(
                api_key="ollama", base_url=f"{self.ollama_url}/v1"
            )
            self.vision_model = ollama_model
            self.vision_provider = "Ollama"
            return True, f"Vision switched to Ollama ({self.vision_model})"

        else:
            return False, f"Unknown provider: {provider}. Use 'openrouter' or 'ollama'"

    def list_ollama_models(self) -> list[str]:
        """List available Ollama models."""
        if not self.ollama_url:
            return []
        try:
            resp = httpx.get(f"{self.ollama_url}/api/tags", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except:
            pass
        return []

    def set_chat_provider(self, provider: str, model: str = None) -> tuple[bool, str]:
        """
        Switch chat provider.

        Args:
            provider: 'together', 'openrouter', 'groq', 'openai', or 'ollama'
            model: Optional model override

        Returns:
            (success, message)
        """
        provider = provider.lower().strip()

        if provider == "ollama":
            if not self.ollama_url:
                self.ollama_url = self._find_ollama_url()

            if not self.ollama_url:
                return False, "Ollama not found. Make sure it's running."

            available = self.list_ollama_models()
            if not available:
                return False, "No models found in Ollama."

            # Use provided model or first available
            chat_model = model
            if not chat_model:
                # Prefer non-vision models for chat
                for m in available:
                    if not any(v in m.lower() for v in ["vl", "vision", "llava"]):
                        chat_model = m
                        break
                if not chat_model:
                    chat_model = available[0]

            self.client = OpenAI(api_key="ollama", base_url=f"{self.ollama_url}/v1")
            self.chat_model = chat_model
            self.provider = "ollama"
            self.base_url = f"{self.ollama_url}/v1"
            return True, f"Chat switched to Ollama ({self.chat_model})"

        elif provider in self.MODEL_PRESETS:
            preset = self.MODEL_PRESETS[provider]

            # Get API key
            env_map = {
                "together": "TOGETHER_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "groq": "GROQ_API_KEY",
                "openai": "OPENAI_API_KEY",
            }
            api_key = os.getenv(env_map.get(provider, ""))
            if not api_key:
                return False, f"{env_map.get(provider, 'API_KEY')} not set"

            self.client = OpenAI(api_key=api_key, base_url=preset["base_url"])
            self.chat_model = model or preset["chat_model"]
            self.provider = provider
            self.base_url = preset["base_url"]
            self.image_model = preset.get("image_model")
            return True, f"Chat switched to {provider.title()} ({self.chat_model})"

        else:
            providers = ", ".join(list(self.MODEL_PRESETS.keys()))
            return False, f"Unknown provider: {provider}. Use: {providers}"

    def agentic_search(
        self, query: str, max_iterations: int = 3, stream: bool = True
    ) -> Generator[str, None, None] | str:
        """
        Perform an agentic web search - the AI searches, analyzes results, and synthesizes.

        Args:
            query: The search query
            max_iterations: Maximum number of search rounds (default 3)
            stream: Whether to stream the response

        Yields/Returns the synthesized answer
        """
        from .utils import web_search

        if stream:
            yield f"🔍 Searching for information about: {query}\n\n"

        gathered_context = []
        search_history = []
        current_query = query

        for iteration in range(max_iterations):
            search_history.append(current_query)

            if stream:
                yield f"Iteration {iteration + 1}: Searching '{current_query}'...\n"

            results = web_search(current_query, num_results=5)

            if not results:
                if stream:
                    yield "No search results found. Proceeding with available context.\n\n"
                break

            if stream:
                yield f"Found {len(results)} results. Analyzing...\n"

            relevant_info = []
            for i, result in enumerate(results[:5]):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                url = result.get("url", "")
                relevant_info.append(f"[{i + 1}] {title}\n{snippet}\nURL: {url}")

            search_context = "\n\n".join(relevant_info)
            gathered_context.append(search_context)

            if iteration < max_iterations - 1:
                context_to_send = (
                    self.SYSTEM_PROMPT
                    + f"\n\nSearch Results for iteration {iteration + 1}:\n{search_context}\n\nBased on these results, do we have sufficient information to answer '{query}'? If not, what should we search for next? Respond with either 'ANSWER: <your answer>' if we have enough info, or 'SEARCH: <next search query>' if we need more information."
                )

                try:
                    messages = [{"role": "user", "content": context_to_send}]
                    response = self.client.chat.completions.create(
                        model=self.chat_model,
                        messages=messages,
                        max_tokens=500,
                        stream=False,
                    )

                    response_text = response.choices[0].message.content.strip()

                    if response_text.upper().startswith("ANSWER:"):
                        if stream:
                            yield "Sufficient information gathered. Synthesizing answer...\n\n"
                        break
                    elif response_text.upper().startswith("SEARCH:"):
                        current_query = response_text[7:].strip()
                        if stream:
                            yield f"Refining search to: {current_query}\n\n"
                    else:
                        if stream:
                            yield "Sufficient information gathered. Synthesizing answer...\n\n"
                        break
                except Exception as e:
                    if stream:
                        yield f"Error in agentic flow: {e}. Proceeding with gathered information...\n\n"
                    break
            else:
                if stream:
                    yield "Final iteration reached. Synthesizing answer...\n\n"

        full_search_context = (
            f"Original Query: {query}\n\nSearch Queries Used:\n"
            + "\n".join(f"- {q}" for q in search_history)
            + f"\n\nGathered Information:\n\n"
            + "\n\n--- Next Iteration ---\n\n".join(gathered_context)
        )

        synthesis_prompt = f"""Based on the following search results, provide a comprehensive and well-structured answer to the user's query: "{query}"

{full_search_context}

Synthesize the information into a clear, helpful response. Include relevant details, cite sources where appropriate, and organize the information logically."""

        self._chat_history.append({"role": "user", "content": synthesis_prompt})

        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT}
            ] + self._chat_history

            if stream:
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                    max_tokens=4096,
                    stream=True,
                )
                full_response = []
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        full_response.append(text)
                        yield text
                result = "".join(full_response)
            else:
                response = self.client.chat.completions.create(
                    model=self.chat_model, messages=messages, max_tokens=4096
                )
                result = response.choices[0].message.content

            self._chat_history.append({"role": "assistant", "content": result})
            self.memory.add_message("assistant", result)

            if not stream:
                return result

        except Exception as e:
            msg = f"❌ Error synthesizing answer: {str(e)}"
            if stream:
                yield msg
            else:
                return msg


# Backwards compatibility
GeminiAssistant = AIAssistant
