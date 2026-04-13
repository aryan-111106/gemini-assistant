"""Voice input/output capabilities with edge-tts as primary backend."""

import os
import tempfile
import subprocess
import shutil
import asyncio
import threading
import queue
from pathlib import Path
from typing import Optional

# Voice dependencies - graceful degradation
VOICE_INPUT_AVAILABLE = False
VOICE_OUTPUT_AVAILABLE = False
TTS_METHOD = None  # 'edge-tts', 'pyttsx3', 'gtts', 'say', 'espeak'
STT_METHOD = None  # 'speech_recognition'

DEFAULT_VOICE = (
    "en-US-GuyNeural"  # Default male voice (change to en-US-JennyNeural for female)
)
EDGE_TTS_VOICES = {
    "male": [
        "en-US-GuyNeural",  # US Male
        "en-GB-RyanNeural",  # UK Male
        "en-AU-WilliamNeural",  # Australian Male
        "en-CA-LiamNeural",  # Canadian Male
    ],
    "female": [
        "en-US-JennyNeural",  # US Female
        "en-GB-SoniaNeural",  # UK Female
        "en-AU-NatashaNeural",  # Australian Female
        "en-CA-ClaraNeural",  # Canadian Female
    ],
}


def _check_command(cmd: str) -> bool:
    """Check if a command exists."""
    return shutil.which(cmd) is not None


# ============ TTS Detection ============

# Try edge-tts FIRST (high quality, preferred)
try:
    import edge_tts

    VOICE_OUTPUT_AVAILABLE = True
    TTS_METHOD = "edge-tts"
except ImportError:
    pass

# Try pyttsx3 as fallback (offline, cross-platform)
if not VOICE_OUTPUT_AVAILABLE:
    try:
        import pyttsx3

        VOICE_OUTPUT_AVAILABLE = True
        TTS_METHOD = "pyttsx3"
    except ImportError:
        pass

# Try gtts + pygame as another fallback
if not VOICE_OUTPUT_AVAILABLE:
    try:
        from gtts import gTTS
        import pygame

        pygame.mixer.init()
        VOICE_OUTPUT_AVAILABLE = True
        TTS_METHOD = "gtts"
    except ImportError:
        pass

# Try system commands as last resort
if not VOICE_OUTPUT_AVAILABLE:
    if _check_command("say"):  # macOS
        VOICE_OUTPUT_AVAILABLE = True
        TTS_METHOD = "say"
    elif _check_command("espeak"):  # Linux
        VOICE_OUTPUT_AVAILABLE = True
        TTS_METHOD = "espeak"
    elif _check_command("espeak-ng"):  # Linux (newer)
        VOICE_OUTPUT_AVAILABLE = True
        TTS_METHOD = "espeak-ng"


# ============ STT Detection ============

# Try speech_recognition with sounddevice
try:
    import speech_recognition as sr

    # Try to create a recognizer
    _test = sr.Recognizer()
    VOICE_INPUT_AVAILABLE = True
    STT_METHOD = "speech_recognition"
except (ImportError, OSError):
    pass


class VoiceInput:
    """Handle voice input via microphone."""

    def __init__(
        self, device_index: int = None, device_name: str = None, verbose: bool = False
    ):
        """
        Initialize voice input.

        Args:
            device_index: Specific microphone device index (None for default)
            device_name: Specific microphone name (or partial name) to search for
            verbose: Print debug information
        """
        if not VOICE_INPUT_AVAILABLE:
            raise RuntimeError(
                "Voice input not available.\n"
                "Install: pip install SpeechRecognition sounddevice\n"
                "On Linux: sudo apt install portaudio19-dev"
            )

        import speech_recognition as sr

        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.verbose = verbose
        self.device_index = device_index
        self.device_name = device_name

        # Configure recognizer for better detection
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 300  # Default, will adjust
        self.recognizer.pause_threshold = 0.5  # Faster response
        self.recognizer.phrase_threshold = 0.3  # Minimum seconds of speaking

        # If device_name is provided, find the matching device_index
        if device_name and device_index is None:
            self.device_index = self.find_microphone_by_name(device_name)
            if self.device_index is None:
                if self.verbose:
                    print(
                        f"[Voice] Warning: Could not find microphone matching '{device_name}'"
                    )
                    print("[Voice] Available microphones:")
                    for idx, name in self.list_microphones():
                        print(f"  [{idx}] {name}")

        # Initialize microphone
        self._init_microphone()

    def list_microphones(self) -> list[tuple[int, str]]:
        """List all available microphones."""
        import speech_recognition as sr

        mics = sr.Microphone.list_microphone_names()
        return [(i, name) for i, name in enumerate(mics)]

    def find_microphone_by_name(self, name_query: str) -> int | None:
        """
        Find microphone device index by name (partial match).

        Args:
            name_query: Partial name to search for (case-insensitive)

        Returns:
            Device index if found, None otherwise
        """
        name_query_lower = name_query.lower()
        mics = self.list_microphones()

        # First try exact match
        for idx, name in mics:
            if name_query_lower == name.lower():
                return idx

        # Then try partial match
        for idx, name in mics:
            if name_query_lower in name.lower():
                return idx

        return None

    def _init_microphone(self):
        """Initialize microphone with fallback options."""
        import speech_recognition as sr

        # Try specific device first if provided
        if self.device_index is not None:
            try:
                self.microphone = sr.Microphone(device_index=self.device_index)
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                if self.verbose:
                    mic_names = sr.Microphone.list_microphone_names()
                    print(
                        f"[Voice] Using microphone [{self.device_index}]: {mic_names[self.device_index]}"
                    )
                    print(
                        f"[Voice] Energy threshold: {self.recognizer.energy_threshold}"
                    )
                return
            except Exception as e:
                if self.verbose:
                    print(f"[Voice] Failed to use device {self.device_index}: {e}")

        # Try default microphone
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            if self.verbose:
                print(f"[Voice] Using default microphone")
                print(f"[Voice] Energy threshold: {self.recognizer.energy_threshold}")
            return
        except OSError as e:
            if self.verbose:
                print(f"[Voice] Default microphone failed: {e}")

        # Fallback: try all available microphones
        mic_names = sr.Microphone.list_microphone_names()
        if self.verbose:
            print(f"[Voice] Trying {len(mic_names)} available microphones...")

        for idx, name in enumerate(mic_names):
            try:
                self.microphone = sr.Microphone(device_index=idx)
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                if self.verbose:
                    print(f"[Voice] Using microphone [{idx}]: {name}")
                    print(
                        f"[Voice] Energy threshold: {self.recognizer.energy_threshold}"
                    )
                self.device_index = idx
                return
            except Exception as e:
                if self.verbose:
                    print(f"[Voice] Microphone [{idx}] '{name}' failed: {e}")
                continue

        raise RuntimeError(
            "Could not access any microphone. Check your audio settings."
        )

    def test_microphone(self, duration: float = 3) -> bool:
        """
        Test if microphone is working by recording audio.

        Args:
            duration: How long to record in seconds

        Returns:
            True if audio was captured, False otherwise
        """
        import speech_recognition as sr

        if self.verbose:
            print(f"[Voice] Testing microphone for {duration}s...")
            print("[Voice] Speak something now!")

        try:
            with self.microphone as source:
                # Don't adjust for ambient noise - just record
                audio = self.recognizer.listen(
                    source, timeout=duration + 2, phrase_time_limit=duration
                )

                audio_size = len(audio.frame_data)
                if self.verbose:
                    print(f"[Voice] Captured {audio_size} bytes of audio")

                # Check if we got meaningful audio (not just silence)
                if audio_size > 1000:  # Arbitrary threshold
                    if self.verbose:
                        print("[Voice] Microphone test PASSED - audio detected")
                    return True
                else:
                    if self.verbose:
                        print("[Voice] Microphone test FAILED - no audio or too quiet")
                    return False

        except sr.WaitTimeoutError:
            if self.verbose:
                print("[Voice] Microphone test FAILED - timeout (no sound detected)")
            return False
        except Exception as e:
            if self.verbose:
                print(f"[Voice] Microphone test FAILED - error: {e}")
            return False

    def listen(
        self,
        timeout: float = 10,
        phrase_time_limit: float = 30,
        verbose: bool = None,
        ambient_noise_duration: float = 0.5,
    ) -> Optional[str]:
        """
        Listen for speech and return transcribed text.

        Args:
            timeout: Max seconds to wait for speech to start
            phrase_time_limit: Max seconds of speech to record
            verbose: Override verbose setting for this call
            ambient_noise_duration: Seconds to sample ambient noise (0 to disable)

        Returns:
            Transcribed text or None if no speech detected
        """
        import speech_recognition as sr

        verbose = verbose if verbose is not None else self.verbose

        try:
            if verbose:
                print(f"[Voice] Listening... (timeout: {timeout}s)")
                print(f"[Voice] Speak now!")

            with self.microphone as source:
                # Adjust for ambient noise before listening
                if ambient_noise_duration > 0:
                    self.recognizer.adjust_for_ambient_noise(
                        source, duration=ambient_noise_duration
                    )

                if verbose:
                    print(
                        f"[Voice] Energy threshold: {self.recognizer.energy_threshold:.1f}"
                    )
                    print(f"[Voice] Waiting for speech...")

                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )

                if verbose:
                    duration = (
                        len(audio.frame_data) / audio.sample_rate / audio.sample_width
                    )
                    print(f"[Voice] Audio captured: {duration:.1f}s")

            # Try Google's free API first
            try:
                if verbose:
                    print("[Voice] Recognizing with Google Speech Recognition...")
                text = self.recognizer.recognize_google(audio)
                if verbose:
                    print(f"[Voice] Recognized: '{text}'")
                return text
            except sr.UnknownValueError:
                if verbose:
                    print("[Voice] Could not understand audio (Google)")
                # Try Whisper as fallback
                try:
                    if verbose:
                        print("[Voice] Trying Whisper fallback...")
                    text = self.recognizer.recognize_whisper(audio, model="base")
                    if verbose:
                        print(f"[Voice] Recognized (Whisper): '{text}'")
                    return text
                except Exception as e:
                    if verbose:
                        print(f"[Voice] Whisper failed: {e}")
                    return None
            except sr.RequestError as e:
                if verbose:
                    print(f"[Voice] Google API error: {e}")
                # Try Whisper on API error
                try:
                    if verbose:
                        print("[Voice] Trying Whisper fallback...")
                    text = self.recognizer.recognize_whisper(audio, model="base")
                    return text
                except:
                    return None

        except sr.WaitTimeoutError:
            if verbose:
                print("[Voice] Timeout - no speech detected")
                print("[Voice] Troubleshooting:")
                print("  - Check if microphone is muted in Windows settings")
                print("  - Speak louder or closer to microphone")
                print("  - Try a different microphone device")
            return None
        except Exception as e:
            if verbose:
                print(f"[Voice] Error during listening: {e}")
            return None


class VoiceOutput:
    """Handle text-to-speech output with edge-tts as primary."""

    def __init__(self, voice: str = None):
        if not VOICE_OUTPUT_AVAILABLE:
            raise RuntimeError(
                "Voice output not available.\n"
                "Install: pip install edge-tts\n"
                "Or: pip install pyttsx3\n"
                "Or: pip install gtts pygame"
            )

        self.method = TTS_METHOD
        self.engine = None
        self._voice = voice or DEFAULT_VOICE
        self._rate = 1.0  # Speed multiplier for edge-tts

        if self.method == "pyttsx3":
            self._init_pyttsx3()

    @property
    def voice(self) -> str:
        """Get current voice."""
        return self._voice

    @voice.setter
    def voice(self, voice_id: str):
        """Set voice (edge-tts voice ID)."""
        self._voice = voice_id

    def set_voice(self, gender: str = "male", accent: str = "us"):
        """
        Set voice by gender and accent.

        Args:
            gender: 'male' or 'female'
            accent: 'us', 'gb' (UK), 'au' (Australia), 'ca' (Canada)
        """
        if self.method == "edge-tts":
            gender = gender.lower()
            accent = accent.lower()

            voices = EDGE_TTS_VOICES.get(gender, EDGE_TTS_VOICES["male"])

            # Find voice matching accent
            accent_map = {
                "us": "en-US",
                "gb": "en-GB",
                "uk": "en-GB",
                "au": "en-AU",
                "ca": "en-CA",
            }
            prefix = accent_map.get(accent, "en-US")

            for v in voices:
                if v.startswith(prefix):
                    self._voice = v
                    return

            # Fallback to first voice in list
            self._voice = voices[0] if voices else DEFAULT_VOICE

    def set_rate(self, rate: float):
        """Set speech rate/speed.

        Args:
            rate: Speed multiplier (0.5 = half speed, 1.0 = normal, 2.0 = double)
        """
        if self.method == "edge-tts":
            self._rate = max(0.5, min(2.0, rate))
        elif self.method == "pyttsx3" and self.engine:
            # pyttsx3 uses words per minute
            self.engine.setProperty("rate", int(175 * rate))

    def set_volume(self, volume: float):
        """Set volume 0.0-1.0 (pyttsx3 only)."""
        if self.method == "pyttsx3" and self.engine:
            self.engine.setProperty("volume", max(0.0, min(1.0, volume)))

    def _init_pyttsx3(self):
        """Initialize pyttsx3 engine."""
        import pyttsx3

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 175)
        self.engine.setProperty("volume", 0.9)

        # Try to find a good voice
        voices = self.engine.getProperty("voices")
        if voices:
            for voice in voices:
                name = voice.name.lower()
                # Prefer natural-sounding voices
                if any(x in name for x in ["zira", "david", "mark", "hazel", "george"]):
                    self.engine.setProperty("voice", voice.id)
                    break

    def speak(self, text: str, block: bool = True):
        """Speak the given text."""
        if not text or not text.strip():
            return

        text = text.strip()

        if self.method == "edge-tts":
            self._speak_edge_tts(text, block)
        elif self.method == "pyttsx3":
            self._speak_pyttsx3(text, block)
        elif self.method == "gtts":
            self._speak_gtts(text)
        elif self.method == "say":
            subprocess.run(["say", text], check=True)
        elif self.method in ("espeak", "espeak-ng"):
            subprocess.run([self.method, text], check=True)

    def _speak_pyttsx3(self, text: str, block: bool):
        """Speak using pyttsx3."""
        self.engine.say(text)
        if block:
            self.engine.runAndWait()

    def _speak_edge_tts(self, text: str, block: bool = True):
        """Speak using edge-tts (Microsoft voices) - PRIMARY METHOD."""
        import edge_tts

        # Split long text into chunks (edge-tts has limits around 5000 chars)
        # We increase this slightly but keep chunks for safety
        max_chars = 4500
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

        temp_files = []

        try:
            # Run TTS generation for ALL chunks in a single event loop/thread
            result_queue = queue.Queue()

            def run_all_tts():
                async def _generate_all():
                    try:
                        for i, chunk in enumerate(chunks):
                            with tempfile.NamedTemporaryFile(
                                suffix=f"_{i}.mp3", delete=False
                            ) as f:
                                temp_path = f.name
                                temp_files.append(temp_path)

                            # Build kwargs for edge_tts.Communicate
                            kwargs = {"text": chunk, "voice": self._voice}
                            if self._rate != 1.0:
                                kwargs["rate"] = f"{int((self._rate - 1) * 100):+d}%"

                            communicate = edge_tts.Communicate(**kwargs)
                            await communicate.save(temp_path)

                        result_queue.put(("success", None))
                    except Exception as e:
                        result_queue.put(("error", str(e)))

                try:
                    asyncio.run(_generate_all())
                except Exception as e:
                    result_queue.put(("error", str(e)))

            # Run the background generation thread
            gen_thread = threading.Thread(target=run_all_tts)
            gen_thread.start()
            gen_thread.join(timeout=120)

            if gen_thread.is_alive():
                raise RuntimeError("TTS generation timed out (120s)")

            status, error = result_queue.get()
            if status == "error":
                raise RuntimeError(f"TTS generation failed: {error}")

            # Play all audio files
            if block:
                for temp_path in temp_files:
                    self._play_audio(temp_path)
            else:
                # Play in background thread
                def play_async():
                    for temp_path in temp_files:
                        self._play_audio(temp_path)

                thread = threading.Thread(target=play_async, daemon=True)
                thread.start()

        finally:
            # Cleanup temp files (only if blocking, otherwise cleanup happens in thread)
            if block:
                for temp_path in temp_files:
                    try:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                    except:
                        pass
            else:
                # For async playback, we need a way to cleanup.
                # Simplest is to let them persist in temp dir or add a cleanup hook.
                # Since they are in tempfile, they will eventually be cleaned by OS.
                pass

    def _play_audio(self, audio_path: str):
        """Play audio file with multiple fallback methods."""
        played = False

        # Try playsound (pure Python, cross-platform)
        if not played:
            try:
                from playsound import playsound

                playsound(audio_path, block=True)
                played = True
            except Exception:
                pass

        # Try ffplay (FFmpeg)
        if not played and _check_command("ffplay"):
            try:
                subprocess.run(
                    [
                        "ffplay",
                        "-nodisp",
                        "-autoexit",
                        "-loglevel",
                        "quiet",
                        audio_path,
                    ],
                    check=True,
                    timeout=300,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                played = True
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                pass

        # Try mpv
        if not played and _check_command("mpv"):
            try:
                subprocess.run(
                    ["mpv", "--no-video", "--really-quiet", audio_path],
                    check=True,
                    timeout=300,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                played = True
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                pass

        # Try afplay (macOS)
        if not played and _check_command("afplay"):
            try:
                subprocess.run(
                    ["afplay", audio_path],
                    check=True,
                    timeout=300,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                played = True
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                pass

        # Try Windows PowerShell with MediaPlayer
        if not played and os.name == "nt":
            try:
                ps_cmd = f'''
                $player = New-Object System.Media.SoundPlayer "{audio_path}"
                $player.PlaySync()
                '''
                subprocess.run(
                    ["powershell", "-Command", ps_cmd],
                    check=True,
                    timeout=300,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                played = True
            except (
                subprocess.TimeoutExpired,
                subprocess.CalledProcessError,
                FileNotFoundError,
            ):
                pass

        # Try pygame
        if not played:
            try:
                import pygame

                if not pygame.mixer.get_init():
                    pygame.mixer.init()

                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()

                # Wait for playback to finish
                clock = pygame.time.Clock()
                while pygame.mixer.music.get_busy():
                    clock.tick(10)

                played = True
            except Exception:
                pass

        # Try pydub as last resort
        if not played:
            try:
                from pydub import AudioSegment
                from pydub.playback import play

                audio = AudioSegment.from_mp3(audio_path)
                play(audio)
                played = True
            except Exception:
                pass

        if not played:
            print(
                f"[Voice] Could not play audio. Install one of: playsound, ffplay, mpv, pygame, or pydub"
            )

    def _speak_gtts(self, text: str):
        """Speak using Google TTS."""
        from gtts import gTTS
        import pygame

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name

        try:
            tts = gTTS(text=text, lang="en")
            tts.save(temp_path)

            if not pygame.mixer.get_init():
                pygame.mixer.init()

            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        finally:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass


class VoiceManager:
    """Unified interface for voice I/O."""

    def __init__(
        self,
        voice: str = None,
        verbose: bool = False,
        mic_device: int = None,
        mic_device_name: str = None,
    ):
        """
        Initialize voice manager.

        Args:
            voice: Default voice ID for TTS
            verbose: Enable verbose output for debugging
            mic_device: Specific microphone device index (None for auto)
            mic_device_name: Specific microphone name to search for (e.g., "Headset", "USB")
        """
        self._input: Optional[VoiceInput] = None
        self._output: Optional[VoiceOutput] = None
        self._default_voice = voice
        self._verbose = verbose
        self._mic_device = mic_device
        self._mic_device_name = mic_device_name

    @property
    def input_available(self) -> bool:
        return VOICE_INPUT_AVAILABLE

    @property
    def output_available(self) -> bool:
        return VOICE_OUTPUT_AVAILABLE

    @property
    def tts_method(self) -> Optional[str]:
        return TTS_METHOD

    @property
    def stt_method(self) -> Optional[str]:
        return STT_METHOD

    def list_microphones(self) -> list[tuple[int, str]]:
        """List all available microphones."""
        if VOICE_INPUT_AVAILABLE:
            import speech_recognition as sr

            mics = sr.Microphone.list_microphone_names()
            return [(i, name) for i, name in enumerate(mics)]
        return []

    def get_input(self) -> VoiceInput:
        """Get voice input handler (lazy init)."""
        if self._input is None:
            self._input = VoiceInput(
                device_index=self._mic_device,
                device_name=self._mic_device_name,
                verbose=self._verbose,
            )
        return self._input

    def get_output(self) -> VoiceOutput:
        """Get voice output handler (lazy init)."""
        if self._output is None:
            self._output = VoiceOutput(voice=self._default_voice)
        return self._output

    def listen(self, **kwargs) -> Optional[str]:
        """Listen for voice input."""
        return self.get_input().listen(**kwargs)

    def speak(self, text: str, **kwargs):
        """Speak text aloud."""
        self.get_output().speak(text, **kwargs)

    def set_voice(self, gender: str = "male", accent: str = "us"):
        """Set voice by gender and accent."""
        self.get_output().set_voice(gender, accent)

    def set_rate(self, rate: float):
        """Set speech rate/speed."""
        self.get_output().set_rate(rate)

    def set_volume(self, volume: float):
        """Set volume (0.0-1.0)."""
        self.get_output().set_volume(volume)

    def set_microphone(self, device_index: int = None, device_name: str = None):
        """
        Set microphone device.

        Args:
            device_index: Specific device index
            device_name: Device name (or partial name) to search for
        """
        if device_index is not None:
            self._mic_device = device_index
            self._mic_device_name = None
        elif device_name:
            self._mic_device = None
            self._mic_device_name = device_name

        # Reset input to reinitialize with new device
        self._input = None

    def status(self) -> dict:
        """Get voice capabilities status."""
        return {
            "input_available": VOICE_INPUT_AVAILABLE,
            "output_available": VOICE_OUTPUT_AVAILABLE,
            "tts_method": TTS_METHOD,
            "stt_method": STT_METHOD,
        }


# ============ Continuous Voice Mode ============


class ContinuousVoiceMode:
    """
    Hands-free continuous voice conversation mode.

    Provides a conversation loop that:
    - Continuously listens for user speech
    - Processes with AI and speaks the response
    - Supports interrupt commands ("stop", "exit", "quit")
    - Visual feedback during different states
    """

    # States for the conversation loop
    STATE_IDLE = "idle"
    STATE_LISTENING = "listening"
    STATE_PROCESSING = "processing"
    STATE_SPEAKING = "speaking"
    STATE_STOPPED = "stopped"

    # Commands that stop the conversation
    STOP_COMMANDS = {"stop", "exit", "quit", "bye", "goodbye", "end"}

    def __init__(
        self,
        voice_manager: VoiceManager,
        on_state_change: callable = None,
        on_user_speech: callable = None,
        on_ai_response: callable = None,
        on_error: callable = None,
        listen_timeout: float = 10.0,
        phrase_time_limit: float = 30.0,
        silence_between_phrases: float = 1.0,
    ):
        """
        Initialize continuous voice mode.

        Args:
            voice_manager: VoiceManager instance for I/O
            on_state_change: Callback(state: str) when state changes
            on_user_speech: Callback(text: str) when user speaks
            on_ai_response: Callback(text: str) -> str for AI processing (must return response)
            on_error: Callback(error: Exception) on errors
            listen_timeout: Seconds to wait for speech before prompting
            phrase_time_limit: Max seconds per phrase
            silence_between_phrases: Pause between listening cycles
        """
        self.voice = voice_manager
        self.on_state_change = on_state_change
        self.on_user_speech = on_user_speech
        self.on_ai_response = on_ai_response
        self.on_error = on_error
        self.listen_timeout = listen_timeout
        self.phrase_time_limit = phrase_time_limit
        self.silence_between_phrases = silence_between_phrases

        self._state = self.STATE_IDLE
        self._running = False
        self._stop_requested = False
        self._thread: Optional[threading.Thread] = None

    @property
    def state(self) -> str:
        """Get current state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if conversation loop is running."""
        return self._running

    def _set_state(self, new_state: str):
        """Update state and notify callback."""
        self._state = new_state
        if self.on_state_change:
            try:
                self.on_state_change(new_state)
            except Exception:
                pass  # Don't let callback errors break the loop

    def _is_stop_command(self, text: str) -> bool:
        """Check if text is a stop command."""
        if not text:
            return False
        # Check if the entire message or first word is a stop command
        words = text.lower().strip().split()
        if words and words[0] in self.STOP_COMMANDS:
            return True
        # Also check if full text is just a stop command
        return text.lower().strip() in self.STOP_COMMANDS

    def start(self, blocking: bool = True):
        """
        Start the continuous conversation loop.

        Args:
            blocking: If True, block until stopped. If False, run in background thread.
        """
        if self._running:
            return

        if not self.voice.input_available:
            raise RuntimeError("Voice input not available for continuous mode")

        if not self.voice.output_available:
            raise RuntimeError("Voice output not available for continuous mode")

        self._running = True
        self._stop_requested = False

        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the conversation loop."""
        self._stop_requested = True
        self._set_state(self.STATE_STOPPED)

    def _run_loop(self):
        """Main conversation loop."""
        import time

        consecutive_timeouts = 0
        max_consecutive_timeouts = 3

        try:
            # Note: adjust_for_ambient_noise is already done in VoiceInput._init_microphone
            # We don't want to do it again here as it might hang some drivers.

            while self._running and not self._stop_requested:
                # === LISTENING PHASE ===
                self._set_state(self.STATE_LISTENING)

                try:
                    # Listen with a timeout.
                    user_text = self.voice.listen(
                        timeout=self.listen_timeout,
                        phrase_time_limit=self.phrase_time_limit,
                        ambient_noise_duration=0,  # Use existing calibration
                    )
                except Exception as e:
                    if self.on_error:
                        self.on_error(f"Listen error: {e}")
                    time.sleep(1)
                    continue

                # Handle timeout (no speech detected)
                if not user_text:
                    consecutive_timeouts += 1
                    time.sleep(0.5)
                    continue

                # Reset timeout counter on successful speech
                consecutive_timeouts = 0

                # Notify callback
                if self.on_user_speech:
                    try:
                        self.on_user_speech(user_text)
                    except Exception:
                        pass

                # Check for stop commands
                if self._is_stop_command(user_text):
                    self._set_state(self.STATE_SPEAKING)
                    self.voice.speak("Ending continuous mode. Goodbye!")
                    break

                # === PROCESSING PHASE ===
                self._set_state(self.STATE_PROCESSING)

                if self.on_ai_response:
                    try:
                        response_text = self.on_ai_response(user_text)
                    except Exception as e:
                        if self.on_error:
                            self.on_error(f"AI error: {e}")
                        response_text = "Sorry, I encountered an error processing that."
                else:
                    response_text = f"You said: {user_text}"

                if not response_text:
                    continue

                # === SPEAKING PHASE ===
                self._set_state(self.STATE_SPEAKING)

                try:
                    # Clean response for TTS (remove markdown, use larger limit)
                    clean_response = self._clean_for_speech(
                        response_text, max_length=5000
                    )
                    self.voice.speak(clean_response)
                except Exception as e:
                    if self.on_error:
                        self.on_error(f"TTS error: {e}")

                # Brief pause before next listening cycle
                time.sleep(0.5)

        finally:
            self._running = False
            self._set_state(self.STATE_STOPPED)

    def _clean_for_speech(self, text: str, max_length: int = 5000) -> str:
        """
        Clean text for TTS output.

        Args:
            text: Raw response text (may contain markdown)
            max_length: Maximum characters to speak

        Returns:
            Cleaned text suitable for speech
        """
        if not text:
            return ""

        # Remove common markdown
        import re

        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", " (code block omitted) ", text)
        text = re.sub(r"`[^`]+`", "", text)

        # Remove headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # Remove bold/italic markers
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)

        # Remove links but keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove bullet points
        text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"  +", " ", text)
        text = text.strip()

        # Truncate if too long
        if len(text) > max_length:
            # Try to truncate at sentence boundary
            truncated = text[:max_length]
            last_period = truncated.rfind(".")
            last_question = truncated.rfind("?")
            last_exclaim = truncated.rfind("!")
            last_sentence = max(last_period, last_question, last_exclaim)

            if last_sentence > max_length * 0.5:
                text = truncated[: last_sentence + 1]
            else:
                text = truncated + "..."

        return text
