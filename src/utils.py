"""Utility functions for file handling, web search, and app launching."""

import base64
import mimetypes
import subprocess
import platform
import webbrowser
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import quote_plus

from PIL import Image
import pypdf


# Supported file types
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".css",
}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def get_file_type(filepath: Path) -> Optional[str]:
    """Determine the type of file."""
    ext = filepath.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in DOCUMENT_EXTENSIONS:
        return "document"
    elif ext in AUDIO_EXTENSIONS:
        return "audio"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def get_mime_type(filepath: Path) -> str:
    """Get MIME type for a file."""
    mime_type, _ = mimetypes.guess_type(str(filepath))
    return mime_type or "application/octet-stream"


def load_image(filepath: Path, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
    """Load and optionally resize an image."""
    img = Image.open(filepath)

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    if img.width > max_size[0] or img.height > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

    return img


def image_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    import io

    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def read_pdf(filepath: Path, max_pages: int = 50) -> str:
    """Extract text content from a PDF file."""
    reader = pypdf.PdfReader(str(filepath))
    pages = min(len(reader.pages), max_pages)

    text_parts = []
    for i in range(pages):
        page = reader.pages[i]
        text = page.extract_text()
        if text:
            text_parts.append(f"--- Page {i + 1} ---\n{text}")

    return "\n\n".join(text_parts)


def read_text_file(filepath: Path, max_chars: int = 100000) -> str:
    """Read text content from a file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read(max_chars)

    if len(content) == max_chars:
        content += "\n\n[... content truncated ...]"

    return content


def read_file_content(filepath: Path) -> Tuple[str, str]:
    """Read file content based on its type."""
    file_type = get_file_type(filepath)

    if file_type == "image":
        img = load_image(filepath)
        return "image", img

    elif file_type == "document":
        if filepath.suffix.lower() == ".pdf":
            content = read_pdf(filepath)
        else:
            content = read_text_file(filepath)
        return "text", content

    elif file_type == "audio":
        return "audio", filepath

    elif file_type == "video":
        return "video", filepath

    else:
        try:
            content = read_text_file(filepath)
            return "text", content
        except Exception:
            raise ValueError(f"Unsupported file type: {filepath.suffix}")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def validate_file(filepath: Path) -> Tuple[bool, str]:
    """Validate that a file exists and is readable."""
    if not filepath.exists():
        return False, f"File not found: {filepath}"

    if not filepath.is_file():
        return False, f"Not a file: {filepath}"

    if filepath.stat().st_size == 0:
        return False, f"File is empty: {filepath}"

    max_size = 20 * 1024 * 1024
    if filepath.stat().st_size > max_size:
        return (
            False,
            f"File too large (max 20MB): {format_file_size(filepath.stat().st_size)}",
        )

    return True, ""


# ============ Web Search ============


def web_search(
    query: str,
    num_results: int = 5,
    timeout: int = 10,
    max_retries: int = 3,
) -> list[dict]:
    """
    Perform a web search using DuckDuckGo.

    Args:
        query: The search query string
        num_results: Maximum number of results to return (default: 5)
        timeout: Request timeout in seconds (default: 10)
        max_retries: Number of retries on rate limit or timeout (default: 3)

    Returns:
        List of result dicts with 'title', 'url', 'snippet' keys

    Raises:
        ImportError: If duckduckgo_search package is not installed (handled internally)
    """
    import time

    DDGS = None

    # Try to import from ddgs (new package) first
    try:
        from ddgs import DDGS
    except ImportError:
        pass

    if DDGS is None:
        # Fallback: return a search URL if package not installed
        search_url = f"https://duckduckgo.com/?q={quote_plus(query)}"
        return [
            {
                "title": "Search Results",
                "url": search_url,
                "snippet": "Install 'ddgs' for real results: pip install duckduckgo-search",
            }
        ]

    last_error = None
    for attempt in range(max_retries):
        try:
            with DDGS(timeout=timeout) as ddgs:
                results = list(ddgs.text(query, max_results=num_results))

            if not results:
                return [
                    {
                        "title": "No Results",
                        "url": f"https://duckduckgo.com/?q={quote_plus(query)}",
                        "snippet": "No results found for this query.",
                    }
                ]

            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                }
                for r in results
            ]

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Check for rate limiting or timeout errors
            if "ratelimit" in error_str or "rate" in error_str or "429" in error_str:
                # Exponential backoff for rate limiting
                wait_time = 2**attempt
                time.sleep(wait_time)
                continue
            elif "timeout" in error_str or "timed out" in error_str:
                # Short delay before retry on timeout
                time.sleep(1)
                continue
            else:
                # Non-recoverable error, don't retry
                break

    # All retries failed
    return [
        {
            "title": "Search Error",
            "url": f"https://duckduckgo.com/?q={quote_plus(query)}",
            "snippet": f"Error: {last_error}. Click URL to search manually.",
        }
    ]


async def web_search_async(
    query: str,
    num_results: int = 5,
    timeout: int = 10,
) -> list[dict]:
    """
    Async version of web search using DuckDuckGo.

    Args:
        query: The search query string
        num_results: Maximum number of results to return (default: 5)
        timeout: Request timeout in seconds (default: 10)

    Returns:
        List of result dicts with 'title', 'url', 'snippet' keys
    """
    import asyncio

    # Run the sync version in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: web_search(query, num_results, timeout, max_retries=2),
    )


def open_url(url: str) -> bool:
    """Open a URL in the default browser."""
    try:
        webbrowser.open(url)
        return True
    except Exception:
        return False


# ============ App Launching ============


def get_system() -> str:
    """Get the current operating system."""
    system = platform.system().lower()

    # Check for WSL
    if system == "linux":
        try:
            with open("/proc/version", "r") as f:
                if "microsoft" in f.read().lower():
                    return "wsl"
        except:
            pass

    if system == "darwin":
        return "macos"
    elif system == "windows":
        return "windows"
    else:
        return "linux"


def open_app(app_name: str) -> Tuple[bool, str]:
    """
    Open an application by name.

    Returns (success, message).
    """
    system = get_system()
    app_name_lower = app_name.lower().strip()

    # Common app mappings
    windows_apps = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "explorer": "explorer.exe",
        "file explorer": "explorer.exe",
        "files": "explorer.exe",
        "cmd": "cmd.exe",
        "command prompt": "cmd.exe",
        "terminal": "cmd.exe",
        "powershell": "powershell.exe",
        "chrome": "chrome.exe",
        "google chrome": "chrome.exe",
        "firefox": "firefox.exe",
        "edge": "msedge.exe",
        "microsoft edge": "msedge.exe",
        "code": "code",  # 'code' is usually in PATH in WSL
        "vscode": "code",
        "visual studio code": "code",
        "word": "winword.exe",
        "excel": "excel.exe",
        "powerpoint": "powerpnt.exe",
        "outlook": "outlook.exe",
        "spotify": "spotify.exe",
        "discord": "discord.exe",
        "slack": "slack.exe",
        "teams": "teams.exe",
        "zoom": "zoom.exe",
        "paint": "mspaint.exe",
        "snipping tool": "snippingtool.exe",
        "settings": "ms-settings:",
        "control panel": "control.exe",
        # Store Apps / Protocols
        "whatsapp": "whatsapp:",
        "telegram": "telegram:",  # Or 'tg:'
        "instagram": "instagram:",
        "messenger": "messenger:",
        "netflix": "netflix:",
        "tiktok": "tiktok:",
        "spotify": "spotify:",  # Protocol often faster than exe
        "todo": "ms-todo:",
        "store": "ms-windows-store:",
        "weather": "bingweather:",
        "photos": "ms-photos:",
        "camera": "microsoft.windows.camera:",
        "calculator": "calculator:",  # Modern calc uses this too
    }

    app_mappings = {
        "windows": windows_apps,
        "wsl": windows_apps,  # WSL uses Windows apps
        "macos": {
            "finder": "Finder",
            "safari": "Safari",
            "chrome": "Google Chrome",
            "google chrome": "Google Chrome",
            "firefox": "Firefox",
            "terminal": "Terminal",
            "iterm": "iTerm",
            "code": "Visual Studio Code",
            "vscode": "Visual Studio Code",
            "visual studio code": "Visual Studio Code",
            "notes": "Notes",
            "messages": "Messages",
            "mail": "Mail",
            "calendar": "Calendar",
            "music": "Music",
            "spotify": "Spotify",
            "discord": "Discord",
            "slack": "Slack",
            "zoom": "zoom.us",
            "photos": "Photos",
            "preview": "Preview",
            "settings": "System Preferences",
            "system preferences": "System Preferences",
            "system settings": "System Settings",
        },
        "linux": {
            "files": "nautilus",
            "file manager": "nautilus",
            "nautilus": "nautilus",
            "terminal": "gnome-terminal",
            "chrome": "google-chrome",
            "google chrome": "google-chrome",
            "firefox": "firefox",
            "code": "code",
            "vscode": "code",
            "visual studio code": "code",
            "spotify": "spotify",
            "discord": "discord",
            "slack": "slack",
            "settings": "gnome-control-center",
            "calculator": "gnome-calculator",
            "gedit": "gedit",
            "text editor": "gedit",
        },
    }

    # Get the app command
    mappings = app_mappings.get(system, {})
    app_cmd = mappings.get(app_name_lower, app_name)

    try:
        if system == "windows":
            # Windows: use start command
            if app_cmd.startswith("ms-"):
                subprocess.Popen(["start", app_cmd], shell=True)
            else:
                subprocess.Popen(["start", "", app_cmd], shell=True)
            return True, f"Opening {app_name}..."

        elif system == "wsl":
            # WSL: use cmd.exe /c start to launch Windows apps
            # For 'code', it's usually a shell script in WSL path, so run directly
            if app_cmd == "code":
                subprocess.Popen(
                    [app_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            else:
                subprocess.Popen(
                    ["cmd.exe", "/c", "start", "", app_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return True, f"Opening {app_name} (Windows)..."

        elif system == "macos":
            # macOS: use open -a
            subprocess.Popen(["open", "-a", app_cmd])
            return True, f"Opening {app_name}..."

        else:
            # Linux: try direct command
            subprocess.Popen(
                [app_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return True, f"Opening {app_name}..."

    except FileNotFoundError:
        return False, f"App not found: {app_name}"
    except Exception as e:
        return False, f"Error opening {app_name}: {e}"


def run_command(command: str) -> Tuple[bool, str]:
    """
    Run a shell command and return output.

    Returns (success, output).
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )

        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr

        return result.returncode == 0, output.strip()

    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, f"Error: {e}"


def fetch_url_content(url: str, max_length: int = 10000) -> Optional[str]:
    """
    Fetch and extract text content from a URL.

    Returns the content or None if failed.
    """
    try:
        import httpx
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        cleaned_text = "\n".join(lines)

        if len(cleaned_text) > max_length:
            cleaned_text = cleaned_text[:max_length]

        return cleaned_text

    except Exception as e:
        return None
