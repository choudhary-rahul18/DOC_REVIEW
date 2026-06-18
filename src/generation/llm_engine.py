import abc
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class BaseLLMEngine(abc.ABC):
    """Abstract LLM engine — instantiate once at startup, call complete() per turn."""

    @abc.abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict],  # [{"role": "user"|"assistant", "content": str}]
        max_tokens: int = 1024,
    ) -> str: ...


class AnthropicEngine(BaseLLMEngine):
    def __init__(self, model: str, api_key: Optional[str] = None):
        import anthropic
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        logger.info("[llm] Anthropic engine ready — model=%s", model)

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.content[0].text


class GeminiEngine(BaseLLMEngine):
    """Google Gemini via google-generativeai. Install: pip install google-generativeai"""

    def __init__(self, model: str, api_key: Optional[str] = None):
        import google.generativeai as genai
        self.model = model
        self._genai = genai
        genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        logger.info("[llm] Gemini engine ready — model=%s", model)

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
        client = self._genai.GenerativeModel(self.model, system_instruction=system)
        # All turns except the last become chat history
        history = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in messages[:-1]
        ]
        chat = client.start_chat(history=history)
        response = chat.send_message(
            messages[-1]["content"],
            generation_config={"max_output_tokens": max_tokens},
        )
        return response.text


class OllamaEngine(BaseLLMEngine):
    """Local Ollama server. Install: pip install ollama  (server must be running)."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        import ollama
        self.model = model
        self._client = ollama.Client(host=base_url)
        logger.info("[llm] Ollama engine ready — model=%s @ %s", model, base_url)

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
        full_messages = [{"role": "system", "content": system}] + messages
        response = self._client.chat(
            model=self.model,
            messages=full_messages,
            options={"num_predict": max_tokens},
        )
        return response.message.content


# ── Factory ────────────────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type[BaseLLMEngine]] = {
    "anthropic": AnthropicEngine,
    "gemini": GeminiEngine,
    "ollama": OllamaEngine,
}


def build_llm_engine(provider: str, model: str, **kwargs) -> BaseLLMEngine:
    """
    Instantiate and pre-launch an LLM engine.

    provider ∈ {"anthropic", "gemini", "ollama"}
    Extra kwargs (api_key, base_url) are forwarded to the engine constructor.

    Example:
        llm = build_llm_engine("anthropic", "claude-sonnet-4-6")
        llm = build_llm_engine("gemini", "gemini-1.5-pro")
        llm = build_llm_engine("ollama", "llama3", base_url="http://localhost:11434")
    """
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](model=model, **kwargs)
