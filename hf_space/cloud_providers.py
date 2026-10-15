"""Cloud transcription providers with automatic fallback.

Keys are read from the .env file next to this module (or from the environment).
Recognised variable names (case-insensitive):

    GROQ_API_KEY      (also gros_api_key)         console.groq.com
    OPENAI_API_KEY                                 platform.openai.com
    DEEPGRAM_API_KEY                               console.deepgram.com
    GEMINI_API_KEY    (also GOOGLE_API_KEY)        aistudio.google.com
    HF_TOKEN          (also HUGGINGFACE_API_KEY)   huggingface.co/settings/tokens

Order of use: Groq -> OpenAI -> Deepgram -> Gemini -> HuggingFace.
When a provider answers 429 (rate limit / quota) it is put on cooldown and the
next chunk goes straight to the next provider that has a key. Once the cooldown
expires the provider is tried again. Providers without a key are skipped.
"""

import base64
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"), override=False)
except ImportError:  # python-dotenv not installed: rely on the environment
    pass

RATE_LIMIT_COOLDOWN = 60      # seconds to skip a provider after HTTP 429
BUSY_COOLDOWN = 30            # seconds to skip a provider after HTTP 503 (high demand)
ERROR_COOLDOWN = 300          # seconds to skip a provider after another failure
NO_CREDITS_COOLDOWN = 3600    # account has no credits / quota: check again in an hour
AUTH_COOLDOWN = 24 * 3600     # a bad key is not retried in this session
_NO_CREDITS_WORDS = ("no credits", "no remaining credits", "insufficient_quota", "billing", "exceeded your current quota")
REQUEST_TIMEOUT = httpx.Timeout(300.0)


def _env(*names):
    wanted = {n.lower() for n in names}
    for key, value in os.environ.items():
        if key.lower() in wanted and value.strip():
            return value.strip()
    return ""


class ProviderExhausted(Exception):
    """Rate limit or quota hit; try the next provider."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderAuthError(Exception):
    """The key was rejected."""


class ProviderError(Exception):
    """Any other failure from a provider."""


def _retry_after(response):
    value = response.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _raise_for_status(name, response):
    if response.status_code == 200:
        return
    body = response.text[:300]
    lowered = response.text[:2000].lower()
    if response.status_code == 402 or any(word in lowered for word in _NO_CREDITS_WORDS):
        raise ProviderExhausted(f"{name} has no credits/quota left: {body}", NO_CREDITS_COOLDOWN)
    if response.status_code == 429:
        raise ProviderExhausted(f"{name} rate limit: {body}", _retry_after(response))
    if response.status_code == 503:
        raise ProviderExhausted(f"{name} is busy (HTTP 503): {body}", BUSY_COOLDOWN)
    if response.status_code in (401, 403):
        raise ProviderAuthError(f"{name} rejected the API key (HTTP {response.status_code}): {body}")
    raise ProviderError(f"{name} returned HTTP {response.status_code}: {body}")


class Provider:
    name = "base"
    env_names = ()

    def __init__(self):
        self.cooldown_until = 0.0
        self.key_override = ""
        self.last_error = ""

    def key(self):
        return self.key_override or _env(*self.env_names)

    def has_key(self):
        return bool(self.key())

    def available(self):
        return self.has_key() and time.time() >= self.cooldown_until

    def cooldown(self, seconds, reason):
        self.cooldown_until = time.time() + seconds
        self.last_error = reason

    def transcribe_chunk(self, client, chunk_path, duration, language=None):
        """Return a list of {"start", "end", "text"} segments relative to the chunk.

        ``language`` is an ISO code such as "hi" or "en", or None for auto-detect.
        """
        raise NotImplementedError


# Whisper's own default: segments decoded with a lower average log-probability are
# usually hallucinations on ring tones, music or noise and are dropped.
LOGPROB_THRESHOLD = -1.0

LANGUAGE_NAMES = {"hi": "Hindi", "en": "English", "pa": "Punjabi", "mr": "Marathi", "gu": "Gujarati",
                  "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "ur": "Urdu"}


class GroqProvider(Provider):
    name = "Groq whisper-large-v3-turbo"
    env_names = ("GROQ_API_KEY", "gros_api_key")
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    model = "whisper-large-v3-turbo"

    def transcribe_chunk(self, client, chunk_path, duration, language=None):
        data = {"model": self.model, "response_format": "verbose_json", "temperature": "0"}
        if language:
            data["language"] = language
        with open(chunk_path, "rb") as handle:
            response = client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.key()}"},
                data=data,
                files={"file": (Path(chunk_path).name, handle, "audio/flac")},
            )
        _raise_for_status(self.name, response)
        payload = response.json()
        segments = []
        for s in payload.get("segments", []):
            logprob = s.get("avg_logprob")
            if logprob is not None and float(logprob) < LOGPROB_THRESHOLD:
                continue  # hallucinated text on ring tone / noise
            segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text", "")})
        return segments


class OpenAIProvider(GroqProvider):
    name = "OpenAI whisper-1"
    env_names = ("OPENAI_API_KEY",)
    url = "https://api.openai.com/v1/audio/transcriptions"
    model = "whisper-1"


class DeepgramProvider(Provider):
    name = "Deepgram nova-3"
    env_names = ("DEEPGRAM_API_KEY",)
    url = "https://api.deepgram.com/v1/listen"

    def transcribe_chunk(self, client, chunk_path, duration, language=None):
        # nova-3 "multi" covers Hindi, English and code-switched Hinglish; "hi" alone
        # returned nothing on a telephony recording in testing.
        dg_language = "en" if language == "en" else "multi"
        with open(chunk_path, "rb") as handle:
            response = client.post(
                self.url,
                params={"model": "nova-3", "smart_format": "true", "utterances": "true", "language": dg_language},
                headers={"Authorization": f"Token {self.key()}", "Content-Type": "audio/flac"},
                content=handle.read(),
            )
        _raise_for_status(self.name, response)
        payload = response.json()
        results = payload.get("results", {})
        utterances = results.get("utterances") or []
        if utterances:
            return [
                {"start": float(u["start"]), "end": float(u["end"]), "text": u.get("transcript", "")}
                for u in utterances
            ]
        channels = results.get("channels") or []
        text = ""
        if channels and channels[0].get("alternatives"):
            text = channels[0]["alternatives"][0].get("transcript", "")
        return [{"start": 0.0, "end": float(duration), "text": text}] if text.strip() else []


class GeminiProvider(Provider):
    name = "Gemini 2.5 Flash"
    env_names = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    prompt = (
        "Transcribe this audio exactly as spoken, in the original language. "
        "Split it into short segments and give the start and end time of each segment "
        "in seconds from the beginning of the audio. Return only the JSON array."
    )
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "start": {"type": "NUMBER"},
                "end": {"type": "NUMBER"},
                "text": {"type": "STRING"},
            },
            "required": ["start", "end", "text"],
        },
    }

    def transcribe_chunk(self, client, chunk_path, duration, language=None):
        # Inline audio must stay under 20 MB including base64, so re-encode to 32 kbps MP3.
        mp3_path = Path(chunk_path).with_suffix(".gemini.mp3")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ProviderError("ffmpeg not found; cannot prepare audio for Gemini")
        subprocess.run(
            [ffmpeg, "-nostdin", "-y", "-i", str(chunk_path), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "libmp3lame", "-b:a", "32k", str(mp3_path)],
            check=True, capture_output=True, text=True,
        )
        try:
            audio_b64 = base64.b64encode(mp3_path.read_bytes()).decode("ascii")
        finally:
            mp3_path.unlink(missing_ok=True)
        prompt = self.prompt
        if language:
            prompt += f" The audio is mainly in {LANGUAGE_NAMES.get(language, language)}; write it in that language's native script."
        body = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "audio/mp3", "data": audio_b64}},
            ]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": self.schema,
            },
        }
        response = client.post(self.url, params={"key": self.key()}, json=body)
        _raise_for_status(self.name, response)
        payload = response.json()
        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            items = json.loads(text)
        except (KeyError, IndexError, ValueError) as error:
            raise ProviderError(f"{self.name} returned an unexpected response: {str(payload)[:300]}") from error
        segments = []
        for item in items:
            try:
                segments.append({"start": float(item["start"]), "end": float(item["end"]), "text": str(item["text"])})
            except (KeyError, TypeError, ValueError):
                continue
        return segments


class HuggingFaceProvider(Provider):
    name = "HuggingFace whisper-large-v3"
    env_names = ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN")
    url = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"

    def transcribe_chunk(self, client, chunk_path, duration, language=None):
        with open(chunk_path, "rb") as handle:
            response = client.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.key()}",
                    "Content-Type": "audio/flac",
                    "x-wait-for-model": "true",
                },
                content=handle.read(),
            )
        if response.status_code == 503:
            raise ProviderExhausted(f"{self.name} model is loading: {response.text[:200]}", 30)
        _raise_for_status(self.name, response)
        payload = response.json()
        text = payload.get("text", "") if isinstance(payload, dict) else ""
        # The serverless endpoint returns plain text only, so the chunk becomes one segment.
        return [{"start": 0.0, "end": float(duration), "text": text}] if text.strip() else []


PROVIDERS = [GroqProvider(), OpenAIProvider(), DeepgramProvider(), GeminiProvider(), HuggingFaceProvider()]


def set_groq_key_override(key):
    """Use a key typed into the app instead of the .env Groq key.

    Changing the key also clears any cooldown, so a bad key pasted once does not
    block the .env key afterwards.
    """
    groq = PROVIDERS[0]
    key = (key or "").strip()
    if key != groq.key_override:
        groq.key_override = key
        groq.cooldown_until = 0.0
        groq.last_error = ""


def provider_status():
    """Human-readable one-liner per provider, for the UI."""
    now = time.time()
    lines = []
    for provider in PROVIDERS:
        if not provider.has_key():
            state = "no key"
        elif now < provider.cooldown_until:
            state = f"cooling down {int(provider.cooldown_until - now)} s ({provider.last_error[:60]})"
        else:
            state = "ready"
        lines.append(f"{provider.name}: {state}")
    return lines


def transcribe_chunk_with_fallback(client, chunk_path, duration, language=None, log=print):
    """Try each provider in order; return (provider_name, segments).

    A provider that hits its rate limit is skipped immediately and cooled down,
    so the very next chunk already goes to the next key.
    """
    if not any(p.has_key() for p in PROVIDERS):
        raise RuntimeError(
            "No cloud API key found. Put GROQ_API_KEY, OPENAI_API_KEY, DEEPGRAM_API_KEY, "
            "GEMINI_API_KEY or HF_TOKEN in the .env file next to the notebook."
        )
    tried = []
    for provider in PROVIDERS:
        if not provider.available():
            continue
        try:
            segments = provider.transcribe_chunk(client, chunk_path, duration, language)
            return provider.name, segments
        except ProviderExhausted as error:
            wait = error.retry_after if error.retry_after else RATE_LIMIT_COOLDOWN
            wait = max(wait, BUSY_COOLDOWN)
            provider.cooldown(wait, str(error))
            log(f"{provider.name}: unavailable ({str(error)[:80]}), switching to the next provider for {int(wait)} s")
        except ProviderAuthError as error:
            provider.cooldown(AUTH_COOLDOWN, str(error))
            log(f"{provider.name}: key rejected, disabled for this session")
        except (ProviderError, httpx.HTTPError, subprocess.CalledProcessError) as error:
            provider.cooldown(ERROR_COOLDOWN, str(error))
            log(f"{provider.name}: failed ({str(error)[:120]}), switching to the next provider")
        tried.append(provider.name)
    raise RuntimeError(
        "Every cloud provider with a key is rate-limited or failing right now. "
        "Status: " + "; ".join(provider_status())
    )


def new_client():
    return httpx.Client(timeout=REQUEST_TIMEOUT)
