#!/usr/bin/env python3
"""
AI Assistant - A polished CLI AI assistant

Features:
- 💬 Natural conversations with memory
- 🖼️ Image understanding & generation
- 📄 Document analysis (PDF, text, code)
- 🔍 Web search
- 📱 App launching
- 🎤 Voice input/output
- 🧠 Persistent conversation history

Supports: Together.ai, OpenRouter, Groq, OpenAI, Ollama
"""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.live import Live
from rich.theme import Theme
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style

# Load environment variables
load_dotenv()

# Rich console with custom theme
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "user": "bold blue",
        "assistant": "bold magenta",
    }
)

console = Console(theme=custom_theme)

# Prompt toolkit style
prompt_style = Style.from_dict(
    {
        "prompt": "#00aa00 bold",
    }
)


def get_assistant(mic_device_name: str = None):
    """Get or create the assistant instance."""
    from .assistant import AIAssistant

    # Get mic device name from environment if not provided
    if mic_device_name is None:
        mic_device_name = os.getenv("AI_MIC_DEVICE")

    try:
        return AIAssistant(mic_device_name=mic_device_name)
    except ValueError as e:
        console.print(f"[error]❌ {e}[/error]")
        console.print("\n[info]Set an API key for your preferred provider:[/info]")
        console.print("  export TOGETHER_API_KEY=your_key    # together.ai")
        console.print("  export OPENROUTER_API_KEY=your_key  # openrouter.ai")
        console.print("  export GROQ_API_KEY=your_key        # groq.com")
        console.print("  export OPENAI_API_KEY=your_key      # openai.com")
        console.print("\n[dim]Or run Ollama locally: ollama serve[/dim]")
        sys.exit(1)


def print_banner(assistant=None):
    """Print the welcome banner."""
    console.clear()

    provider_info = ""
    if assistant:
        info = assistant.info()
        vision = f" • Vision: {info['vision']}" if info.get("vision") else ""
        provider_info = (
            f"\n[dim]{info['provider'].title()} • {info['chat_model']}{vision}[/dim]"
        )

    console.print(
        Panel.fit(
            f"[bold cyan]✨ AI Assistant[/bold cyan]{provider_info}",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print("[dim]Type /help for commands. /quit to exit.[/dim]\n")


def print_help():
    """Print detailed help information."""
    help_text = """
## 💬 Chat Commands

- Just type your message to chat
- `/new` - Start a new conversation  
- `/history` - View past conversations
- `/load <id>` - Load a previous conversation
- `/context` - Show current conversation

## 🖼️ Image Commands

- `/image analyze <path>` - Analyze an image
- `/image generate <prompt>` - Generate an image

## 📄 File Commands  

- `/file <path>` - Analyze any file (PDF, code, text)
- `/file <path> <question>` - Ask a question about a file
- Or just paste a file path in your message!

## 🔍 Search & Apps

- `/search <query>` - Search the web
- `/asearch <query>` - Agentic search (AI searches, analyzes, and synthesizes multi-step)
- `/open <app>` - Open an application
- `/run <command>` - Run a shell command

## 🎤 Voice Commands

- `/voice` - Toggle voice input mode
- `/speak` - Toggle voice output
- `/listen` - Listen for one voice command
- `/continuous` - Start hands-free conversation mode (say "stop" to exit)
- `/mics` - List available microphones
- `/mics <name|index>` - Select microphone device

## ⚙️ Settings

- `/model` - Show current model info
- `/chat` - Show/switch chat provider (ollama/together/openrouter/groq)
- `/vision` - Show/switch vision provider (openrouter/ollama)
- `/clear` - Clear the screen
- `/quit` or `/exit` - Exit the assistant
"""
    console.print(Markdown(help_text))


def stream_response(assistant, message: str, files: list = None):
    """Stream and display the assistant's response."""
    full_response = []

    with Live(
        Panel(
            "",
            title="[bold magenta]JARVIS[/bold magenta]",
            border_style="magenta",
            width=80,
        ),
        refresh_per_second=10,
    ) as live:
        for chunk in assistant.send_message(message, files=files, stream=True):
            full_response.append(chunk)
            content = Markdown("".join(full_response))
            live.update(
                Panel(
                    content,
                    title="[bold magenta]JARVIS[/bold magenta]",
                    border_style="magenta",
                    width=80,
                )
            )

    console.print()


def handle_image_command(assistant, args: str):
    """Handle /image commands."""
    parts = args.strip().split(maxsplit=1)

    if not parts:
        console.print(
            "[warning]Usage: /image analyze <path> OR /image generate <prompt>[/warning]"
        )
        return

    action = parts[0].lower()

    if action == "analyze" and len(parts) > 1:
        path = Path(parts[1].strip().strip("\"'"))
        if not path.exists():
            console.print(f"[error]File not found: {path}[/error]")
            return

        with console.status("Analyzing image...", spinner="dots"):
            result = assistant.analyze_image(path)
        console.print(
            Panel(Markdown(result), title=f"📷 {path.name}", border_style="green")
        )

    elif action == "generate" and len(parts) > 1:
        prompt = parts[1].strip()

        with console.status("Generating image...", spinner="dots"):
            try:
                output_path = assistant.generate_image(prompt)
                if output_path:
                    console.print(
                        f"[success]✅ Image saved to: {output_path}[/success]"
                    )
                else:
                    console.print("[error]Failed to generate image[/error]")
            except RuntimeError as e:
                console.print(f"[error]{e}[/error]")
    else:
        console.print(
            "[warning]Usage: /image analyze <path> OR /image generate <prompt>[/warning]"
        )


def handle_file_command(assistant, args: str):
    """Handle /file commands."""
    if not args.strip():
        console.print("[warning]Usage: /file <path> [question][/warning]")
        return

    parts = args.strip().split(maxsplit=1)
    path = Path(parts[0].strip().strip("\"'"))
    question = parts[1] if len(parts) > 1 else None

    if not path.exists():
        console.print(f"[error]File not found: {path}[/error]")
        return

    with console.status(f"Analyzing {path.name}...", spinner="dots"):
        result = assistant.analyze_document(path, question)

    title = f"📄 {path.name}"
    if question:
        title += f" - {question[:50]}..."
    console.print(Panel(Markdown(result), title=title, border_style="blue"))


def handle_search_command(args: str):
    """Handle /search commands."""
    from .utils import web_search, fetch_url_content

    if not args.strip():
        console.print("[warning]Usage: /search <query>[/warning]")
        return

    query = args.strip()

    with console.status(f"Searching for '{query}'...", spinner="dots"):
        results = web_search(query)

    if not results:
        console.print("[warning]No results found[/warning]")
        return

    table = Table(title=f"🔍 Search: {query}", border_style="cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="white")
    table.add_column("Snippet", style="dim")

    for i, result in enumerate(results, 1):
        table.add_row(
            str(i), result.get("title", "")[:50], result.get("snippet", "")[:80] + "..."
        )

    console.print(table)
    console.print(
        "\n[info]Enter a number to view content, or press Enter to continue[/info]"
    )

    try:
        choice = Prompt.ask("View result", default="")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                url = results[idx].get("url", "")
                title = results[idx].get("title", "")[:50]
                if url:
                    console.print(f"\n[info]Fetching content from: {title}[/info]")
                    with console.status("Fetching page content...", spinner="dots"):
                        content = fetch_url_content(url)
                    if content:
                        summary = content[:2000]
                        if len(content) > 2000:
                            summary += "\n\n[... content truncated ...]"
                        console.print(
                            Panel(
                                Markdown(summary),
                                title=f"📄 {title}",
                                border_style="green",
                            )
                        )
                        console.print(f"\n[dim]URL: {url}[/dim]")
                    else:
                        console.print("[warning]Could not fetch page content[/warning]")
    except KeyboardInterrupt:
        pass


def handle_open_command(args: str):
    """Handle /open commands."""
    from .utils import open_app

    if not args.strip():
        console.print("[warning]Usage: /open <app name>[/warning]")
        console.print(
            "[info]Examples: /open chrome, /open calculator, /open vscode[/info]"
        )
        return

    app_name = args.strip()
    success, message = open_app(app_name)

    if success:
        console.print(f"[success]{message}[/success]")
    else:
        console.print(f"[error]{message}[/error]")


def handle_agentic_search_command(assistant, args: str):
    """Handle /asearch commands - agentic search with multi-step analysis."""
    from .utils import web_search

    if not args.strip():
        console.print("[warning]Usage: /asearch <query>[/warning]")
        return

    query = args.strip()
    full_response = []

    with console.status("Starting agentic search...", spinner="dots"):
        pass

    with Live(
        Panel(
            "",
            title="[bold cyan]🔍 Agentic Search[/bold cyan]",
            border_style="cyan",
            width=80,
        ),
        refresh_per_second=10,
    ) as live:
        for chunk in assistant.agentic_search(query, stream=True):
            full_response.append(chunk)
            content = Markdown("".join(full_response))
            live.update(
                Panel(
                    content,
                    title="[bold cyan]🔍 Agentic Search[/bold cyan]",
                    border_style="cyan",
                    width=80,
                )
            )


def handle_run_command(args: str):
    """Handle /run commands."""
    from .utils import run_command

    if not args.strip():
        console.print("[warning]Usage: /run <command>[/warning]")
        return

    command = args.strip()
    console.print(f"[info]Running: {command}[/info]")

    success, output = run_command(command)

    if output:
        console.print(
            Panel(output, title="Output", border_style="green" if success else "red")
        )
    else:
        console.print(
            "[success]Command completed[/success]"
            if success
            else "[error]Command failed[/error]"
        )


def handle_history_command(assistant, args: str):
    """Handle /history commands."""
    conversations = assistant.list_conversations()

    if not conversations:
        console.print("[info]No saved conversations yet.[/info]")
        return

    table = Table(title="📚 Conversation History", border_style="cyan")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Messages", justify="right")
    table.add_column("Created", style="dim")

    for conv in conversations[:20]:
        table.add_row(
            conv["id"],
            conv["title"][:40],
            str(conv["message_count"]),
            conv["created_at"][:16],
        )

    console.print(table)
    console.print("\n[info]Use /load <id> to load a conversation[/info]")


def handle_listen_command(assistant, speak_mode: bool):
    """Handle /listen command."""
    if not assistant.voice.input_available:
        console.print("[error]Voice input not available.[/error]")
        console.print(
            "[info]To enable, install: pip install SpeechRecognition sounddevice[/info]"
        )
        return

    console.print("[info]🎤 Listening...[/info]")

    try:
        text = assistant.voice.listen()
        if text:
            console.print(f"[user]You said:[/user] {text}")
            stream_response(assistant, text)

            if speak_mode and assistant.voice.output_available:
                history = assistant.get_conversation_history()
                if history:
                    try:
                        clean = history[-1]["content"].replace("*", "").replace("#", "")
                        assistant.speak(clean[:500])
                    except Exception as e:
                        console.print(f"[warning]Could not speak: {e}[/warning]")
        else:
            console.print("[warning]Didn't catch that. Try again.[/warning]")
    except Exception as e:
        console.print(f"[error]Voice error: {e}[/error]")


def handle_mics_command(assistant, args: str):
    """Handle /mics command to list and select microphones."""
    if not assistant.voice.input_available:
        console.print("[error]Voice input not available.[/error]")
        console.print(
            "[info]To enable, install: pip install SpeechRecognition sounddevice[/info]"
        )
        return

    mics = assistant.voice.list_microphones()

    if not mics:
        console.print("[warning]No microphones found.[/warning]")
        return

    if not args.strip():
        # List all microphones
        table = Table(title="🎤 Available Microphones", border_style="cyan")
        table.add_column("Index", style="dim", justify="right")
        table.add_column("Name", style="white")

        for idx, name in mics:
            table.add_row(str(idx), name)

        console.print(table)
        console.print("\n[info]Usage:[/info]")
        console.print("  /mics <index>     - Select microphone by index")
        console.print("  /mics <name>      - Select microphone by name (partial match)")
        console.print("\n[dim]Example: /mics 2 or /mics 'Headset'[/dim]")
    else:
        # Try to select microphone
        arg = args.strip()

        # Try as index first
        try:
            device_index = int(arg)
            if 0 <= device_index < len(mics):
                assistant.voice.set_microphone(device_index=device_index)
                console.print(
                    f"[success]✅ Microphone set to [{device_index}]: {mics[device_index][1]}[/success]"
                )
            else:
                console.print(f"[error]Invalid index. Use 0-{len(mics) - 1}[/error]")
        except ValueError:
            # Try as name
            assistant.voice.set_microphone(device_name=arg)
            # Check if it was set successfully
            mics_after = assistant.voice.list_microphones()
            for idx, name in mics_after:
                if arg.lower() in name.lower():
                    console.print(
                        f"[success]✅ Microphone set to [{idx}]: {name}[/success]"
                    )
                    return
            console.print(f"[error]No microphone found matching '{arg}'[/error]")


def handle_continuous_command(assistant):
    """Handle /continuous command - hands-free conversation mode."""
    from .voice import ContinuousVoiceMode
    from rich.status import Status
    import datetime

    if not assistant.voice.input_available:
        console.print("[error]Voice input not available.[/error]")
        console.print("[info]Install: pip install SpeechRecognition sounddevice[/info]")
        return

    if not assistant.voice.output_available:
        console.print("[error]Voice output not available.[/error]")
        console.print("[info]Install: pip install edge-tts[/info]")
        return

    # State indicator styles (spinner style, text)
    state_config = {
        "idle": ("dots", "[dim]⏸️  Idle[/dim]"),
        "listening": ("dots12", "[bold cyan]🎤 Listening...[/bold cyan]"),
        "processing": ("dots", "[bold yellow]🧠 Processing...[/bold yellow]"),
        "speaking": ("speaker", "[bold magenta]🔊 Speaking...[/bold magenta]"),
        "stopped": ("dots", "[dim]⏹️  Stopped[/dim]"),
    }

    # Shared state for the status widget
    current_state = {"text": "[cyan]🎤 Starting...[/cyan]", "spinner": "dots12"}

    def on_state_change(state: str):
        """Update the shared state (status widget reads this)."""
        spinner, text = state_config.get(state, ("dots", f"[dim]{state}[/dim]"))
        now = datetime.datetime.now().strftime("%H:%M:%S")
        current_state["text"] = f"[dim][{now}][/dim] {text}"
        current_state["spinner"] = spinner
        # Print state changes on new lines for visibility
        console.print(f"[dim][{now}][/dim] {text}")

    def on_user_speech(text: str):
        """Display what user said."""
        console.print(f"\n[user]You said:[/user] {text}")

    def on_ai_response(user_text: str) -> str:
        """Process user input with AI and return response."""
        # Use streaming for better UX, but collect full response
        full_response = []

        # Use Live display for the response
        panel_content = ""
        with Live(
            Panel(
                "",
                title="[bold magenta]JARVIS[/bold magenta]",
                border_style="magenta",
                width=80,
            ),
            refresh_per_second=10,
        ) as live:
            for chunk in assistant.send_message(user_text, stream=True):
                full_response.append(chunk)
                panel_content = "".join(full_response)
                live.update(
                    Panel(
                        Markdown(panel_content),
                        title="[bold magenta]JARVIS[/bold magenta]",
                        border_style="magenta",
                        width=80,
                    )
                )

        return panel_content

    def on_error(error: Exception):
        """Handle errors."""
        console.print(f"\n[error]Error: {error}[/error]")

    # Create and start continuous mode
    console.print(
        Panel(
            "[bold cyan]🎙️ Continuous Voice Mode[/bold cyan]\n\n"
            "• Speak naturally - I'm always listening\n"
            "• Say [bold]'stop'[/bold], [bold]'exit'[/bold], or [bold]'quit'[/bold] to end\n"
            "• Press [bold]Ctrl+C[/bold] to interrupt",
            border_style="cyan",
        )
    )

    continuous = ContinuousVoiceMode(
        voice_manager=assistant.voice,
        on_state_change=on_state_change,
        on_user_speech=on_user_speech,
        on_ai_response=on_ai_response,
        on_error=on_error,
        listen_timeout=8.0,
        phrase_time_limit=30.0,
    )

    try:
        continuous.start(blocking=True)
    except KeyboardInterrupt:
        continuous.stop()
        console.print("\n[info]Continuous mode interrupted.[/info]")
    finally:
        console.print("\n[success]✅ Continuous mode ended.[/success]")


def interactive_session(assistant):
    """Run the interactive chat session."""
    print_banner(assistant)

    # Setup prompt history
    history_file = Path.home() / ".ai-assistant" / "prompt_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    session = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
    )

    voice_mode = False
    speak_mode = False

    while True:
        try:
            # Get input (voice or text)
            if voice_mode and assistant.voice.input_available:
                console.print("[dim]🎤 Listening...[/dim]", end=" ")
                try:
                    user_input = assistant.voice.listen()
                    if user_input:
                        console.print(f"[user]{user_input}[/user]")
                    else:
                        console.print("[warning]Didn't catch that.[/warning]")
                        continue
                except Exception as e:
                    console.print(f"[error]Voice error: {e}[/error]")
                    voice_mode = False
                    console.print(
                        "[info]Voice mode disabled. Type your message.[/info]"
                    )
                    continue
            else:
                user_input = session.prompt(
                    [("class:prompt", "> ")], style=prompt_style
                ).strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd_parts = user_input[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                args = cmd_parts[1] if len(cmd_parts) > 1 else ""

                if cmd in ("quit", "exit", "q"):
                    console.print("[info]👋 Goodbye![/info]")
                    break

                elif cmd == "help":
                    print_help()

                elif cmd == "new":
                    assistant.reset_chat()
                    console.print("[success]✨ Started new conversation[/success]")

                elif cmd == "clear":
                    print_banner(assistant)

                elif cmd == "image":
                    handle_image_command(assistant, args)

                elif cmd == "file":
                    handle_file_command(assistant, args)

                elif cmd == "search":
                    handle_search_command(args)

                elif cmd == "asearch":
                    handle_agentic_search_command(assistant, args)

                elif cmd == "open":
                    handle_open_command(args)

                elif cmd == "run":
                    handle_run_command(args)

                elif cmd == "history":
                    handle_history_command(assistant, args)

                elif cmd == "load":
                    if args:
                        if assistant.load_conversation(args.strip()):
                            console.print(
                                f"[success]✅ Loaded conversation: {args}[/success]"
                            )
                        else:
                            console.print(
                                f"[error]Conversation not found: {args}[/error]"
                            )
                    else:
                        console.print(
                            "[warning]Usage: /load <conversation_id>[/warning]"
                        )

                elif cmd == "voice":
                    if assistant.voice.input_available:
                        voice_mode = not voice_mode
                        status = "enabled 🎤" if voice_mode else "disabled"
                        console.print(f"[info]Voice input {status}[/info]")
                    else:
                        console.print("[error]Voice input not available.[/error]")
                        console.print(
                            "[info]Install: pip install SpeechRecognition sounddevice[/info]"
                        )

                elif cmd == "speak":
                    if assistant.voice.output_available:
                        speak_mode = not speak_mode
                        status = "enabled 🔊" if speak_mode else "disabled"
                        console.print(f"[info]Voice output {status}[/info]")
                        if speak_mode:
                            console.print(
                                f"[dim]Using: {assistant.voice.tts_method}[/dim]"
                            )
                    else:
                        console.print("[error]Voice output not available.[/error]")
                        console.print(
                            "[info]Install: pip install pyttsx3 OR pip install gtts pygame[/info]"
                        )

                elif cmd == "listen":
                    handle_listen_command(assistant, speak_mode)

                elif cmd == "mics":
                    handle_mics_command(assistant, args)

                elif cmd in ("continuous", "convo"):
                    handle_continuous_command(assistant)

                elif cmd == "context":
                    history = assistant.get_conversation_history()
                    if history:
                        for msg in history[-10:]:
                            role = "You" if msg["role"] == "user" else "JARVIS"
                            style = "user" if msg["role"] == "user" else "assistant"
                            content = msg["content"][:200]
                            if len(msg["content"]) > 200:
                                content += "..."
                            console.print(f"[{style}]{role}:[/{style}] {content}")
                    else:
                        console.print(
                            "[info]No messages in current conversation[/info]"
                        )

                elif cmd == "model":
                    info = assistant.info()
                    table = Table(title="⚙️ Configuration", border_style="cyan")
                    table.add_column("Setting", style="cyan")
                    table.add_column("Value", style="white")
                    for key, value in info.items():
                        table.add_row(key.replace("_", " ").title(), str(value))
                    console.print(table)

                elif cmd == "vision":
                    if not args:
                        # Show current vision config
                        info = assistant.info()
                        console.print(f"[info]Vision Provider: {info['vision']}[/info]")
                        console.print(
                            f"[info]Vision Model: {info['vision_model']}[/info]"
                        )
                        console.print(
                            "\n[dim]Switch with: /vision openrouter OR /vision ollama[/dim]"
                        )
                        console.print(
                            "[dim]Custom model: /vision ollama llava:13b[/dim]"
                        )

                        # Show Ollama models if available
                        ollama_models = assistant.list_ollama_models()
                        if ollama_models:
                            console.print(
                                f"\n[dim]Ollama models: {', '.join(ollama_models[:10])}[/dim]"
                            )
                    else:
                        parts = args.strip().split(maxsplit=1)
                        provider = parts[0]
                        model = parts[1] if len(parts) > 1 else None
                        success, message = assistant.set_vision_provider(
                            provider, model
                        )
                        if success:
                            console.print(f"[success]✅ {message}[/success]")
                        else:
                            console.print(f"[error]❌ {message}[/error]")

                elif cmd == "chat":
                    if not args:
                        # Show current chat config
                        info = assistant.info()
                        console.print(
                            f"[info]Chat Provider: {info['provider'].title()}[/info]"
                        )
                        console.print(f"[info]Chat Model: {info['chat_model']}[/info]")
                        console.print(
                            "\n[dim]Switch with: /chat ollama OR /chat together OR /chat openrouter[/dim]"
                        )
                        console.print(
                            "[dim]Custom model: /chat ollama deepseek-v3.1:671b-cloud[/dim]"
                        )

                        # Show Ollama models if available
                        ollama_models = assistant.list_ollama_models()
                        if ollama_models:
                            console.print(
                                f"\n[dim]Ollama models: {', '.join(ollama_models[:10])}[/dim]"
                            )
                    else:
                        parts = args.strip().split(maxsplit=1)
                        provider = parts[0]
                        model = parts[1] if len(parts) > 1 else None
                        success, message = assistant.set_chat_provider(provider, model)
                        if success:
                            console.print(f"[success]✅ {message}[/success]")
                        else:
                            console.print(f"[error]❌ {message}[/error]")

                else:
                    console.print(
                        f"[warning]Unknown command: /{cmd}. Type /help for commands.[/warning]"
                    )

            else:
                # Display user message
                console.print(
                    Panel(
                        user_input,
                        title="[bold blue]You[/bold blue]",
                        border_style="blue",
                        width=80,
                    )
                )

                # Check if input contains file paths
                files = []
                words = user_input.split()
                remaining_words = []

                for word in words:
                    clean_word = word.strip("\"'")
                    potential_path = Path(clean_word).expanduser()
                    if potential_path.exists() and potential_path.is_file():
                        files.append(potential_path)
                        console.print(
                            f"[info]📎 Attached: {potential_path.name}[/info]"
                        )
                    else:
                        remaining_words.append(word)

                message = (
                    " ".join(remaining_words)
                    if remaining_words
                    else "Describe this file"
                )

                # Stream the response
                stream_response(assistant, message, files if files else None)

                # Speak response if enabled
                if speak_mode and assistant.voice.output_available:
                    history = assistant.get_conversation_history()
                    if history:
                        try:
                            clean = (
                                history[-1]["content"]
                                .replace("*", "")
                                .replace("#", "")
                                .replace("`", "")
                            )
                            assistant.speak(clean[:500])
                        except Exception as e:
                            console.print(f"[warning]Could not speak: {e}[/warning]")

        except KeyboardInterrupt:
            console.print("\n[info]Use /quit to exit[/info]")
            continue
        except EOFError:
            console.print("\n[info]👋 Goodbye![/info]")
            break
        except Exception as e:
            console.print(f"[error]Error: {e}[/error]")
            continue


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """AI Assistant - Your AI-powered multimodal assistant."""
    if ctx.invoked_subcommand is None:
        assistant = get_assistant()
        interactive_session(assistant)


@main.command()
@click.argument("message", nargs=-1, required=True)
@click.option(
    "--file", "-f", multiple=True, type=click.Path(exists=True), help="Attach file(s)"
)
def ask(message, file):
    """Ask a single question."""
    assistant = get_assistant()
    message_text = " ".join(message)
    files = [Path(f) for f in file] if file else None

    for chunk in assistant.send_message(message_text, files=files, stream=True):
        console.print(chunk, end="")
    console.print()


@main.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--question", "-q", help="Question about the image")
def analyze(image_path, question):
    """Analyze an image."""
    assistant = get_assistant()

    with console.status("Analyzing image..."):
        result = assistant.analyze_image(Path(image_path), question)

    console.print(Markdown(result))


@main.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option(
    "--output", "-o", type=click.Path(), help="Output path for generated image"
)
@click.option(
    "--aspect", "-a", default="1:1", help="Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4)"
)
def generate(prompt, output, aspect):
    """Generate an image from a prompt."""
    assistant = get_assistant()
    prompt_text = " ".join(prompt)

    with console.status("Generating image..."):
        try:
            output_path = assistant.generate_image(
                prompt_text,
                output_path=Path(output) if output else None,
                aspect_ratio=aspect,
            )
            if output_path:
                console.print(f"[success]✅ Image saved to: {output_path}[/success]")
            else:
                console.print("[error]Failed to generate image[/error]")
        except RuntimeError as e:
            console.print(f"[error]{e}[/error]")


@main.command()
@click.argument("query", nargs=-1, required=True)
def search(query):
    """Search the web."""
    handle_search_command(" ".join(query))


@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option(
    "--iterations", "-i", default=3, help="Maximum search iterations (default 3)"
)
def asearch(query, iterations):
    """Agentic search - AI searches, analyzes, and synthesizes multi-step."""
    assistant = get_assistant()
    query_text = " ".join(query)

    full_response = []
    with console.status("Starting agentic search...", spinner="dots"):
        pass

    with Live(
        Panel(
            "",
            title="[bold cyan]Agentic Search[/bold cyan]",
            border_style="cyan",
            width=80,
        ),
        refresh_per_second=10,
    ) as live:
        for chunk in assistant.agentic_search(
            query_text, max_iterations=iterations, stream=True
        ):
            full_response.append(chunk)
            content = Markdown("".join(full_response))
            live.update(
                Panel(
                    content,
                    title="[bold cyan]Agentic Search[/bold cyan]",
                    border_style="cyan",
                    width=80,
                )
            )


@main.command()
def history():
    """List conversation history."""
    assistant = get_assistant()
    handle_history_command(assistant, "")


@main.command()
def info():
    """Show assistant configuration."""
    assistant = get_assistant()
    info_dict = assistant.info()

    table = Table(title="⚙️ AI Assistant Configuration", border_style="cyan")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    for key, value in info_dict.items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


if __name__ == "__main__":
    main()
