# ✨ JARVIS: Multimodal AI Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gemini](https://img.shields.io/badge/Built%20with-Gemini-blue.svg)](https://deepmind.google/technologies/gemini/)

A premium, feature-rich CLI AI assistant with a polished "JARVIS" personality. Built for productivity, it handles text, images, documents, and real-time web searches with ease.

## 🚀 Key Features

- 💬 **Multimodal Conversations** - Talk to your AI about anything, including images and files.
- 🔍 **Intelligent Web Search** - Real-time access to the web via DuckDuckGo.
- 🧠 **Agentic Search** - Multi-step research where the AI searches, analyzes, and synthesizes information.
- 📄 **Document Intelligence** - Native support for PDF, Python, JSON, Markdown, and more.
- 🖼️ **Vision & Art** - High-fidelity image analysis (Vision) and creation (DALL-E 3, FLUX).
- 🎤 **Voice Interaction** - Full voice input and ultra-realistic TTS output via Microsoft Edge.
- 📱 **System Integration** - Launch apps and run shell commands directly from the assistant.
- 💾 **Persistent Memory** - Conversations are saved automatically for later recall.
- ✨ **Beautiful Interface** - Rich markdown rendering, syntax highlighting, and smooth animations.

## 🛠️ Supported Providers

JARVIS works with a wide range of providers using native SDKs or OpenAI-compatible APIs:

| Provider | Chat | Vision | Image Gen | Note |
|----------|------|--------|-----------|------|
| **Google Gemini** | ✅ | ✅ | ❌ | Native `google-genai` support |
| **Together.ai** | ✅ | ✅ | ✅ | Recommended for speed & FLUX |
| **OpenRouter** | ✅ | ✅ | ❌ | Access to Llama 3, Claude, etc. |
| **Groq** | ✅ | ✅ | ❌ | Ultra-fast inference |
| **OpenAI** | ✅ | ✅ | ✅ | Industry standard (GPT-4o, DALL-E 3) |
| **Ollama** | ✅ | ✅ | ❌ | **100% Local** via your hardware |

## ⚙️ Installation

### 1. Requirements
Ensure you have **Python 3.10+** installed.

### 2. Setup
```bash
# Clone the project
git clone https://github.com/your-repo/gemini-assistant.git
cd gemini-assistant

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[all]"
```

### 3. Configuration
Create a `.env` file in the root directory (see `.env.example`):
```env
TOGETHER_API_KEY=your_key
GEMINI_API_KEY=your_key
OPENROUTER_API_KEY=your_key
GROQ_API_KEY=your_key
OPENAI_API_KEY=your_key
```

## 🎮 Usage

### Start Interactive Mode
```bash
ai
# or
assistant
```

### Quick Commands
```bash
# Ask a direct question
ai ask "Explain quantum entanglement like I'm five."

# Search the web
ai search "Latest news on space exploration"

# Analyze an image
ai analyze path/to/photo.jpg -q "What's happening in this scene?"

# Generate an image
ai generate "A futuristic cyberpunk city at night with neon lights"
```

## ⌨️ Command Reference

While in interactive mode, use these slash commands:

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/new` | Start a fresh conversation |
| `/history` | Browse past conversation titles |
| `/load <id>` | Resume a specific conversation |
| `/image analyze <path>` | Deep analysis of an image |
| `/image generate <prompt>` | Create images using AI |
| `/file <path>` | Read and discuss any document |
| `/search <query>` | Real-time DuckDuckGo search |
| `/asearch <query>` | **Agentic Search**: Multistep reasoning & synthesis |
| `/open <app>` | Launch apps (e.g., `chrome`, `vscode`, `notepad`) |
| `/voice` | Toggle Voice Input (Microphone) |
| `/speak` | Toggle Voice Output (TTS) |
| `/continuous` | Hands-free conversation mode |
| `/model` | Inspect current AI & system configuration |
| `/quit` | Save and exit |

## 🎙️ Voice Setup

For the best experience, we recommend using the `edge-tts` engine for high-quality, human-like responses.

- **Windows**: Works out of the box with `pip install .[voice]`.
- **WSL/Linux**: Requires `portaudio` for microphone support:
  ```bash
  sudo apt install portaudio19-dev
  ```

## 📂 Project Structure

```text
gemini-assistant/
├── src/
│   ├── main.py        # CLI Interface & Command Handling
│   ├── assistant.py   # Core AI Logic & Multi-provider Engine
│   ├── memory.py      # Persistence & History Management
│   ├── utils.py       # Web Search, Image Ops, App Launching
│   └── voice.py       # Speech-to-Text & Text-to-Speech
├── conversations/     # Your saved chat history
├── pyproject.toml     # Project metadata & entry points
└── requirements.txt   # Dependency list
```

## 📄 License

This project is licensed under the **MIT License**. Use it, tweak it, and make it your own!

---
*Built with ❤️ for the Vibe Coding community.*
