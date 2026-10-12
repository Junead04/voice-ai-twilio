"""
Explicit Gemini context caching for pipecat-ai 0.0.98 (GoogleLLMService).

Why this file exists
--------------------
pipecat 0.0.98 has no first-class `cached_content` support. Just passing
`extra={"cached_content": ...}` is NOT enough, because GoogleLLMService always
also sends `system_instruction`, `tools` and `tool_config` on every request, and
the Gemini API rejects that combination with:

    400 CachedContent can not be used with GenerateContent request setting
    system_instruction, tools or tool_config.

So this subclass:
  1. creates ONE cache (system prompt + tool declarations) per process, shared by
     every call, refreshed before its TTL runs out;
  2. on each request, swaps those three fields for `cached_content=<cache name>`;
  3. falls back to the normal (uncached) request if anything about the cache is
     unavailable, mismatched, expired, or rejected -- the bot never breaks because
     of caching;
  4. logs prompt / cached token counts per request so you can SEE whether it works.

Usage (see onetts.py edits):
    llm = CachedGoogleLLMService(api_key=..., model=..., params=..., tools=[...])
    ...
    context = LLMContext(messages)
    await llm.enable_prefix_cache(context)     # non-blocking by default
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

from google.genai import types
from loguru import logger
from pipecat.services.google.llm import GoogleLLMService

CACHE_TTL_SECONDS = 3600  # explicit caches are billed for storage while they live
REFRESH_BEFORE_EXPIRY_SECONDS = 300  # start building a replacement this early
EXPIRY_SAFETY_SECONDS = 15  # never *use* a cache this close to expiry
RETRY_AFTER_FAILURE_SECONDS = 600  # after a failed create (e.g. prefix too small), back off


@dataclass
class _Entry:
    name: Optional[str] = None
    expires_at: float = 0.0
    retry_at: float = 0.0


# Process-wide state: every call uses the same static prefix, so they share one cache.
_entries: Dict[str, _Entry] = {}
_locks: Dict[str, asyncio.Lock] = {}


def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, (list, tuple)):
        return [_dump(o) for o in obj]
    return obj


def _prefix_key(model: str, system_instruction: str, tools: Any, tool_config: Any) -> str:
    blob = json.dumps(
        {
            "model": model,
            "system": system_instruction,
            "tools": _dump(tools),
            "tool_config": _dump(tool_config),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class CachedGoogleLLMService(GoogleLLMService):
    """GoogleLLMService that serves system prompt + tools from a Gemini explicit cache."""

    def __init__(self, *args, cache_ttl_seconds: int = CACHE_TTL_SECONDS, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_ttl = cache_ttl_seconds
        self._cache_system_instruction: Optional[str] = None
        self._cache_key: Optional[str] = None
        self._cache_bg_task: Optional[asyncio.Task] = None
        self._request_used_cache = False

    # ------------------------------------------------------------------ setup
    async def enable_prefix_cache(self, context, wait: bool = False) -> None:
        """Register the static prefix (system prompt + tools) and make sure a cache exists.

        Pass the same LLMContext the pipeline uses: the system instruction is derived
        through pipecat's own adapter, so it is byte-identical to what gets sent per
        request. With wait=False (default) creation happens in the background and
        the first turn(s) simply run uncached until it is ready -- no added latency.
        """
        params = self.get_llm_adapter().get_llm_invocation_params(context)
        system_instruction = params["system_instruction"]
        if not system_instruction:
            logger.warning("[gemini-cache] context has no system message; caching disabled")
            return
        self._cache_system_instruction = system_instruction
        self._cache_key = _prefix_key(
            self._model_name, system_instruction, self._tools, self._tool_config
        )
        if wait:
            await self._ensure_cache()
        else:
            self._spawn_ensure()

    def _spawn_ensure(self) -> None:
        if self._cache_bg_task and not self._cache_bg_task.done():
            return
        self._cache_bg_task = asyncio.create_task(self._ensure_cache())

    async def _ensure_cache(self) -> None:
        """Create (or refresh) the shared cache. Never raises."""
        key = self._cache_key
        if not key:
            return
        lock = _locks.setdefault(key, asyncio.Lock())
        async with lock:
            entry = _entries.setdefault(key, _Entry())
            now = time.time()
            if entry.name and entry.expires_at - now > REFRESH_BEFORE_EXPIRY_SECONDS:
                return  # still fresh (another call already built it)
            if now < entry.retry_at:
                return  # recent failure, don't hammer the API
            try:
                cache = await self._client.aio.caches.create(
                    model=self._model_name,
                    config=types.CreateCachedContentConfig(
                        display_name=f"pipecat-prefix-{key}",
                        system_instruction=self._cache_system_instruction,
                        tools=self._tools or None,
                        tool_config=self._tool_config or None,
                        ttl=f"{self._cache_ttl}s",
                    ),
                )
            except Exception as e:  # e.g. prefix below the model's minimum, model unsupported
                entry.retry_at = time.time() + RETRY_AFTER_FAILURE_SECONDS
                logger.warning(
                    f"[gemini-cache] could not create cache for {self._model_name}; "
                    f"continuing UNCACHED (retry in {RETRY_AFTER_FAILURE_SECONDS}s). Reason: {str(e)[:400]}"
                )
                return
            old_name = entry.name
            entry.name = cache.name
            entry.expires_at = time.time() + self._cache_ttl
            tokens = getattr(getattr(cache, "usage_metadata", None), "total_token_count", None)
            logger.info(f"[gemini-cache] ready: {cache.name} ({tokens} tokens, ttl={self._cache_ttl}s)")
            if old_name:
                try:
                    await self._client.aio.caches.delete(name=old_name)
                except Exception:
                    pass  # it will expire on its own

    # ---------------------------------------------------------------- runtime
    def _active_cache_name(self) -> Optional[str]:
        entry = _entries.get(self._cache_key) if self._cache_key else None
        if not entry or not entry.name:
            if self._cache_key:
                self._spawn_ensure()  # expired / dropped / never built: rebuild in background
            return None
        remaining = entry.expires_at - time.time()
        if remaining <= EXPIRY_SAFETY_SECONDS:
            self._spawn_ensure()
            return None
        if remaining <= REFRESH_BEFORE_EXPIRY_SECONDS:
            self._spawn_ensure()  # keep serving the old one while a new one is built
        return entry.name

    def _drop_cache(self) -> None:
        entry = _entries.get(self._cache_key) if self._cache_key else None
        if entry:
            entry.name = None
            entry.expires_at = 0.0
        self._spawn_ensure()

    def _build_generation_params(
        self,
        system_instruction: Optional[str] = None,
        tools: Optional[list] = None,
        tool_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = super()._build_generation_params(
            system_instruction=system_instruction, tools=tools, tool_config=tool_config
        )
        self._request_used_cache = False
        name = self._active_cache_name()
        # Only use the cache when this request's prefix is exactly what was cached.
        if (
            name
            and system_instruction == self._cache_system_instruction
            and tools == self._tools
            and (tool_config or None) == (self._tool_config or None)
        ):
            params.pop("system_instruction", None)
            params.pop("tools", None)
            params.pop("tool_config", None)
            params["cached_content"] = name
            self._request_used_cache = True
        return params

    async def _stream_content(self, params_from_context) -> AsyncIterator[Any]:
        try:
            stream = await super()._stream_content(params_from_context)
        except Exception as e:
            # Cache deleted/expired server-side, wrong model, etc.: retry once uncached.
            if self._request_used_cache and "cache" in str(e).lower():
                logger.warning(f"[gemini-cache] cache rejected ({str(e)[:200]}); retrying uncached")
                self._drop_cache()
                stream = await super()._stream_content(params_from_context)
            else:
                raise
        return self._log_usage(stream, self._request_used_cache)

    async def _log_usage(self, stream, used_cache: bool) -> AsyncIterator[Any]:
        usage = None
        async for chunk in stream:
            if getattr(chunk, "usage_metadata", None):
                usage = chunk.usage_metadata
            yield chunk
        if usage:
            logger.info(
                f"[gemini-cache] explicit={'ON' if used_cache else 'off'} "
                f"prompt_tokens={usage.prompt_token_count} "
                f"cached_tokens={usage.cached_content_token_count or 0}"
            )
