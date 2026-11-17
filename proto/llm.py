"""Minimal OpenRouter client with timing.

    r = chat(messages, schema=SCHEMA)     # -> {"content", "json", "ttft_s", "total_s", "usage", "cost"}

Reads OPENROUTER_API_KEY / OPENROUTER_MODEL from .env (via engines.load_env).
Streams the response so time-to-first-token is measured honestly.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rig"))
import engines  # noqa: E402,F401  (loads .env)

URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl"


class LLMError(RuntimeError):
    pass


def model_name() -> str:
    return os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL


def available() -> tuple[bool, str]:
    engines.load_env()
    ok = bool(os.environ.get("OPENROUTER_API_KEY"))
    return ok, "" if ok else "Paste OPENROUTER_API_KEY into .env, then restart the server"


def _extract_json(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < 0:
        raise LLMError(f"no JSON in reply: {text[:200]}")
    return json.loads(t[a:b + 1])


def chat(messages: list[dict], schema: dict | None = None, temperature: float = 0.3,
         max_tokens: int = 800, reasoning: str | None = "low", model: str | None = None,
         timeout: int = 120, retries: int = 2) -> dict:
    ok, why = available()
    if not ok:
        raise LLMError(why)
    body = {
        "model": model or model_name(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "usage": {"include": True},
        # personal data: route only to providers that don't store/train on prompts.
        # ':free' models have no such provider, so the filter is dropped for them: use fictional data only.
        **({} if (model or model_name()).endswith(":free") else {"provider": {"data_collection": "deny"}}),
    }
    if reasoning == "off":
        body["reasoning"] = {"enabled": False}
    elif reasoning:
        body["reasoning"] = {"effort": reasoning, "exclude": True}
    if schema:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "reply", "strict": True, "schema": schema}}
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
               "Content-Type": "application/json",
               "HTTP-Referer": "http://localhost/kaabil-voice-rig", "X-Title": "Kaabil voice rig"}
    last = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        ttft, parts, usage, used_model = None, [], {}, body["model"]
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    ev = json.loads(data)
                    if "error" in ev:
                        raise LLMError(str(ev["error"])[:300])
                    used_model = ev.get("model", used_model)
                    for ch in ev.get("choices", []):
                        piece = (ch.get("delta") or {}).get("content")
                        if piece:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            parts.append(piece)
                    if ev.get("usage"):
                        usage = ev["usage"]
            content = "".join(parts)
            out = {"content": content, "ttft_s": round(ttft or 0, 3),
                   "total_s": round(time.perf_counter() - t0, 3), "usage": usage,
                   "cost": usage.get("cost"), "model": used_model}
            if schema:
                out["json"] = _extract_json(content)
            return out
        except urllib.error.HTTPError as e:
            msg = e.read()[:400].decode("utf-8", "replace")
            last = LLMError(f"HTTP {e.code}: {msg}")
            if e.code in (400, 401, 402, 403):
                raise last from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, LLMError) as e:
            last = e if isinstance(e, LLMError) else LLMError(f"{type(e).__name__}: {e}")
        time.sleep(1.5 * (attempt + 1))
    raise last
