"""LLM providers for the config generator.

  anthropic   Claude via the official anthropic SDK              ANTHROPIC_API_KEY
  openai      OpenAI                                             OPENAI_API_KEY
  openrouter  OpenRouter (any model it hosts)                    OPENROUTER_API_KEY
  custom      any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, Together, Groq, DeepSeek, ...)
              or Anthropic-compatible one (FORGE_LLM_API_STYLE=anthropic)
              FORGE_LLM_BASE_URL + FORGE_LLM_API_KEY (+ FORGE_LLM_MODEL)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from . import ui


class LLMError(RuntimeError):
    pass


class ModelNotFound(LLMError):
    pass


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    key_env: str
    base_url: str | None
    default_model: str


SPECS = {
    "anthropic": ProviderSpec("anthropic", "Claude (Anthropic)", "ANTHROPIC_API_KEY", None, "claude-opus-5-5"),
    "openai": ProviderSpec("openai", "OpenAI", "OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-6.1-terra"),
    "openrouter": ProviderSpec(
        "openrouter", "OpenRouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "anthropic/claude-opus-5.5"
    ),
    "custom": ProviderSpec("custom", "Custom endpoint (OpenAI- or Anthropic-compatible)", "FORGE_LLM_API_KEY", None, ""),
}

# Claude models that accept the server-side refusal fallback ("default" routing)
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_NOT_CHAT = re.compile(r"(mini|nano|audio|realtime|image|tts|transcribe|search|embedding|whisper|dall-e|moderation|instruct|babbage|davinci)")


def _claude_version(model: str) -> float:
    m = re.match(r"claude-(?:opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d+))?", model)
    if not m:
        return 0.0
    return float(f"{m.group(1)}.{m.group(2) or 0}")


class Provider:
    name = "base"

    def __init__(self, model: str):
        self.model = model

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def list_models(self) -> list[tuple[str, int]]:  # (id, created)
        return []

    def label(self) -> str:
        return f"{self.name}:{self.model}"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str | None, model: str, base_url: str | None = None, effort: str = "high"):
        super().__init__(model)
        import anthropic

        self._anthropic = anthropic
        kwargs = {"max_retries": 3}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**kwargs)
        self.base_url = base_url
        self.effort = effort

    def complete(self, system: str, user: str) -> str:
        anthropic = self._anthropic
        params = {
            "model": self.model,
            "max_tokens": 32000,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if _claude_version(self.model) >= 4.6:
            params["thinking"] = {"type": "adaptive"}
            params["output_config"] = {"effort": self.effort}
        use_fallback = self.base_url is None and self.model in _FALLBACK_MODELS
        try:
            msg = self._stream(params, use_fallback)
        except anthropic.NotFoundError as exc:
            raise ModelNotFound(f"model {self.model!r} not found on Anthropic") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key (ANTHROPIC_API_KEY)") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError(f"Anthropic: no access to {self.model}") from exc
        except anthropic.BadRequestError as exc:
            if use_fallback:  # e.g. fallbacks not enabled for this org/endpoint: retry plain
                try:
                    msg = self._stream(params, False)
                except anthropic.BadRequestError as exc2:
                    raise LLMError(f"Anthropic bad request: {exc2.message}") from exc2
            else:
                raise LLMError(f"Anthropic bad request: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit hit - try again shortly") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("could not reach the Anthropic API (network?)") from exc

        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMError(f"Claude declined the request (category: {category})")
        if msg.stop_reason == "max_tokens":
            ui.warn("Claude hit max_tokens; the config may be cut short")
        return "".join(block.text for block in msg.content if block.type == "text")

    def _stream(self, params: dict, use_fallback: bool):
        if use_fallback:
            extra = {"betas": ["server-side-fallback-2026-07-01"]}
            try:
                with self.client.beta.messages.stream(**params, **extra, fallbacks="default") as stream:
                    return stream.get_final_message()
            except TypeError:  # older SDK without the typed parameter
                with self.client.beta.messages.stream(**params, **extra, extra_body={"fallbacks": "default"}) as stream:
                    return stream.get_final_message()
        with self.client.messages.stream(**params) as stream:
            return stream.get_final_message()

    def list_models(self) -> list[tuple[str, int]]:
        out = []
        for m in self.client.models.list(limit=100):
            created = getattr(m, "created_at", None)
            out.append((m.id, int(created.timestamp()) if created else 0))
        return out


class OpenAICompatProvider(Provider):
    def __init__(self, name: str, api_key: str | None, base_url: str, model: str, headers: dict | None = None):
        super().__init__(model)
        from openai import OpenAI

        self.name = name
        self._openai = __import__("openai")
        self.client = OpenAI(
            api_key=api_key or "not-needed", base_url=base_url, default_headers=headers, max_retries=3, timeout=900
        )

    def complete(self, system: str, user: str) -> str:
        openai = self._openai
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        attempts = [{}] if self.name == "openai" else [{"max_tokens": 16000}, {}]
        last: Exception | None = None
        for extra in attempts:
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages, **extra)
            except openai.NotFoundError as exc:
                raise ModelNotFound(f"model {self.model!r} not found on {self.name}") from exc
            except openai.AuthenticationError as exc:
                raise LLMError(f"{self.name} rejected the API key") from exc
            except openai.BadRequestError as exc:
                last = exc
                if "model" in str(exc).lower() and ("not exist" in str(exc).lower() or "invalid model" in str(exc).lower()):
                    raise ModelNotFound(str(exc)) from exc
                continue
            except openai.RateLimitError as exc:
                raise LLMError(f"{self.name} rate limit / quota hit") from exc
            except openai.APIConnectionError as exc:
                raise LLMError(f"could not reach {self.name} ({self.client.base_url})") from exc
            except openai.APIStatusError as exc:
                raise LLMError(f"{self.name} API error {exc.status_code}: {exc.message}") from exc
            if not resp.choices:
                raise LLMError(f"{self.name} returned no choices")
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                ui.warn(f"{self.name} hit its output limit; the config may be cut short")
            text = choice.message.content or ""
            if not text.strip():
                raise LLMError(f"{self.name} returned an empty answer")
            return text
        raise LLMError(f"{self.name} rejected the request: {last}")

    def list_models(self) -> list[tuple[str, int]]:
        return [(m.id, int(getattr(m, "created", 0) or 0)) for m in self.client.models.list()]


def _newest_chat_model(provider: Provider, prefer_prefix: str | None) -> str | None:
    try:
        models = provider.list_models()
    except Exception:
        return None
    if prefer_prefix:
        filtered = [m for m in models if m[0].startswith(prefer_prefix) and not _NOT_CHAT.search(m[0])]
        models = filtered or models
    models.sort(key=lambda m: m[1], reverse=True)
    return models[0][0] if models else None


def complete_with_recovery(provider: Provider, system: str, user: str) -> str:
    """Call the provider; if the configured model does not exist, switch to the newest one it lists."""
    try:
        return provider.complete(system, user)
    except ModelNotFound as exc:
        prefix = {"openai": "gpt-", "anthropic": "claude-opus", "openrouter": None}.get(provider.name)
        replacement = _newest_chat_model(provider, prefix)
        if not replacement or replacement == provider.model:
            raise
        ui.warn(f"{exc} - switching to {replacement} (set FORGE_LLM_MODEL to pin one)")
        provider.model = replacement
        return provider.complete(system, user)


def detect_provider_name(settings) -> str:
    if settings.provider and settings.provider != "auto":
        return settings.provider
    if settings.llm_base_url:
        return "custom"
    for name in ("anthropic", "openrouter", "openai"):
        if os.environ.get(SPECS[name].key_env):
            return name
    return "none"


def build(settings, name: str | None = None) -> Provider | None:
    name = name or detect_provider_name(settings)
    if name == "none":
        return None
    if name not in SPECS:
        raise LLMError(f"unknown provider {name!r} (anthropic | openai | openrouter | custom | none)")
    spec = SPECS[name]
    if name == "custom":
        if not settings.llm_base_url:
            raise LLMError("custom provider needs FORGE_LLM_BASE_URL")
        key = settings.llm_api_key or os.environ.get("FORGE_LLM_API_KEY")
        if settings.llm_api_style == "anthropic":
            model = settings.llm_model or "claude-opus-5-5"
            return AnthropicProvider(key, model, base_url=settings.llm_base_url, effort=settings.llm_effort)
        if not settings.llm_model:
            probe = OpenAICompatProvider("custom", key, settings.llm_base_url, "")
            settings.llm_model = _newest_chat_model(probe, None)
            if not settings.llm_model:
                raise LLMError("custom provider: set FORGE_LLM_MODEL (the endpoint did not list any models)")
        return OpenAICompatProvider("custom", key, settings.llm_base_url, settings.llm_model)
    key = settings.llm_api_key if settings.provider == name and settings.llm_api_key else os.environ.get(spec.key_env)
    if not key:
        raise LLMError(f"{spec.label} needs {spec.key_env}")
    model = settings.llm_model or spec.default_model
    if name == "anthropic":
        return AnthropicProvider(key, model, effort=settings.llm_effort)
    headers = {"X-Title": "axolotl-forge"} if name == "openrouter" else None
    return OpenAICompatProvider(name, key, spec.base_url or "", model, headers=headers)


def interactive_pick(settings) -> Provider | None:
    """Wizard: choose provider, key and model (models are fetched from the provider)."""
    detected = detect_provider_name(settings)
    options = [
        ("anthropic", "Claude (Anthropic)" + ("  [dim]key found[/dim]" if os.environ.get("ANTHROPIC_API_KEY") else "")),
        ("openai", "OpenAI" + ("  [dim]key found[/dim]" if os.environ.get("OPENAI_API_KEY") else "")),
        ("openrouter", "OpenRouter" + ("  [dim]key found[/dim]" if os.environ.get("OPENROUTER_API_KEY") else "")),
        ("custom", "Custom endpoint (vLLM, Ollama, LM Studio, Together, Groq, ...)"),
        ("none", "No AI - use forge's built-in rules engine"),
    ]
    default = next((i for i, (v, _) in enumerate(options) if v == detected), 0)
    name = ui.choose("Which AI should write your axolotl config?", options, default=default)
    if name == "none":
        settings.provider = "none"
        return None
    settings.provider = name
    if name == "custom":
        settings.llm_base_url = ui.ask("Base URL (e.g. http://localhost:11434/v1)", settings.llm_base_url)
        settings.llm_api_style = ui.choose(
            "API style", [("openai", "OpenAI-compatible /chat/completions"), ("anthropic", "Anthropic-compatible /v1/messages")]
        )
        settings.llm_api_key = ui.ask("API key (blank if none)", settings.llm_api_key or "", password=True) or None
    elif not os.environ.get(SPECS[name].key_env):
        key = ui.ask(f"{SPECS[name].label} API key", password=True)
        os.environ[SPECS[name].key_env] = key
    provider = build(settings, name)
    if provider is None:
        return None
    try:
        with ui.console.status("fetching model list…"):
            models = provider.list_models()
    except Exception:
        models = []
    if models:
        models.sort(key=lambda m: m[1], reverse=True)
        ids = [m[0] for m in models if not _NOT_CHAT.search(m[0]) or name == "custom"][:15] or [m[0] for m in models[:15]]
        default_model = provider.model if provider.model in ids else ids[0]
        if provider.model and provider.model not in ids:
            ids.insert(0, provider.model)
            default_model = provider.model
        choice = ui.choose(
            "Model", [(m, m) for m in ids], default=ids.index(default_model), allow_custom=True, custom_label="type a model id"
        )
        provider.model = choice
        settings.llm_model = choice
    return provider
