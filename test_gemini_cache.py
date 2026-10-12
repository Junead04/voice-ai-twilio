"""
Offline tests for gemini_cache.py. No network: only the two Gemini calls are faked.
Run:  uv run --with pytest python -m pytest -q test_gemini_cache.py
"""

import asyncio
from types import SimpleNamespace

import pytest
from google.genai import types
from pipecat.processors.aggregators.llm_context import LLMContext

import gemini_cache
from gemini_cache import CachedGoogleLLMService

TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_products_by_name",
                description="Search for products",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "q"}},
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="end_call",
                description="End the call",
                parameters={"type": "object", "properties": {}, "required": []},
            ),
        ]
    )
]
SYSTEM = "You are Aarav, a customer support agent for Sash Clothing."


@pytest.fixture(autouse=True)
def _reset_state():
    gemini_cache._entries.clear()
    gemini_cache._locks.clear()


class FakeGenAI:
    """Stands in for genai.Client().aio; records what would be sent to Google."""

    def __init__(self, create_error=None, reject_cached_once=False):
        self.create_calls = 0
        self.delete_calls = []
        self.sent_configs = []
        self.sent_contents = []
        self.create_error = create_error
        self.reject_cached_once = reject_cached_once
        self.aio = SimpleNamespace(
            caches=SimpleNamespace(create=self._create, delete=self._delete),
            models=SimpleNamespace(generate_content_stream=self._stream),
        )

    async def _create(self, model, config):
        self.create_calls += 1
        self.last_create_config = config
        if self.create_error:
            raise self.create_error
        return SimpleNamespace(
            name=f"cachedContents/test{self.create_calls}",
            usage_metadata=SimpleNamespace(total_token_count=3300),
        )

    async def _delete(self, name):
        self.delete_calls.append(name)

    async def _stream(self, model, contents, config):
        self.sent_configs.append(config)
        self.sent_contents.append(contents)
        if config.cached_content and self.reject_cached_once:
            self.reject_cached_once = False
            raise RuntimeError("404 NOT_FOUND: CachedContent not found (or permission denied)")

        async def gen():
            yield SimpleNamespace(
                candidates=[],
                usage_metadata=SimpleNamespace(prompt_token_count=3300, cached_content_token_count=0),
            )

        return gen()


def make_service(fake):
    svc = CachedGoogleLLMService(
        api_key="fake",
        model="gemini-3.1-flash-lite",
        params=CachedGoogleLLMService.InputParams(temperature=0.7, max_tokens=400),
        tools=TOOLS,
    )
    svc._client = SimpleNamespace(aio=fake.aio)
    return svc


def make_context():
    return LLMContext(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Hi, I want a hoodie"},
        ]
    )


async def drain(stream):
    return [c async for c in stream]


def test_cached_request_strips_prefix_and_sets_cached_content():
    async def run():
        fake = FakeGenAI()
        svc, ctx = make_service(fake), make_context()
        await svc.enable_prefix_cache(ctx, wait=True)
        assert fake.create_calls == 1
        # cache built from exactly the prompt + tools we use per request
        assert fake.last_create_config.system_instruction == SYSTEM
        assert fake.last_create_config.tools == TOOLS
        assert fake.last_create_config.ttl == "3600s"

        await drain(await svc._stream_content_universal_context(ctx))
        cfg = fake.sent_configs[-1]
        assert cfg.cached_content == "cachedContents/test1"
        # these three MUST be absent or Gemini answers 400
        assert cfg.system_instruction is None
        assert cfg.tools is None
        assert cfg.tool_config is None
        # generation params and conversation still go through
        assert cfg.temperature == 0.7 and cfg.max_output_tokens == 400
        assert len(fake.sent_contents[-1]) == 1  # only the user turn; system prompt not in contents

    asyncio.run(run())


def test_uncached_until_ready_then_cached():
    async def run():
        fake = FakeGenAI()
        svc, ctx = make_service(fake), make_context()
        await svc.enable_prefix_cache(ctx)  # background
        # first turn goes out before the background task runs -> normal request
        await drain(await svc._stream_content_universal_context(ctx))
        first = fake.sent_configs[-1]
        assert first.cached_content is None and first.system_instruction == SYSTEM and first.tools
        await asyncio.sleep(0.05)  # background creation completes
        await drain(await svc._stream_content_universal_context(ctx))
        assert fake.sent_configs[-1].cached_content == "cachedContents/test1"
        assert fake.create_calls == 1

    asyncio.run(run())


def test_create_failure_falls_back_and_backs_off():
    async def run():
        fake = FakeGenAI(create_error=RuntimeError("Cached content is too small. min_total_token_count=4096"))
        svc, ctx = make_service(fake), make_context()
        await svc.enable_prefix_cache(ctx, wait=True)
        for _ in range(3):
            await drain(await svc._stream_content_universal_context(ctx))
            await asyncio.sleep(0.02)
        for cfg in fake.sent_configs:
            assert cfg.cached_content is None
            assert cfg.system_instruction == SYSTEM and cfg.tools  # original behaviour intact
        assert fake.create_calls == 1  # backoff: not retried on every turn

    asyncio.run(run())


def test_server_rejects_cache_then_retry_uncached_and_rebuilds():
    async def run():
        fake = FakeGenAI(reject_cached_once=True)
        svc, ctx = make_service(fake), make_context()
        await svc.enable_prefix_cache(ctx, wait=True)
        chunks = await drain(await svc._stream_content_universal_context(ctx))
        assert chunks  # request succeeded despite the rejection
        assert fake.sent_configs[0].cached_content == "cachedContents/test1"  # tried cached
        assert fake.sent_configs[1].cached_content is None  # retried uncached
        assert fake.sent_configs[1].system_instruction == SYSTEM
        await asyncio.sleep(0.05)
        assert fake.create_calls == 2  # replacement cache built in background
        await drain(await svc._stream_content_universal_context(ctx))
        assert fake.sent_configs[-1].cached_content == "cachedContents/test2"

    asyncio.run(run())


def test_changed_system_prompt_is_not_served_from_cache():
    async def run():
        fake = FakeGenAI()
        svc, ctx = make_service(fake), make_context()
        await svc.enable_prefix_cache(ctx, wait=True)
        ctx.messages[0]["content"] = SYSTEM + " (edited mid-call)"
        await drain(await svc._stream_content_universal_context(ctx))
        cfg = fake.sent_configs[-1]
        assert cfg.cached_content is None
        assert cfg.system_instruction.endswith("(edited mid-call)")

    asyncio.run(run())


def test_expiring_cache_is_not_used_and_is_refreshed():
    async def run():
        fake = FakeGenAI()
        svc, ctx = make_service(fake), make_context()
        await svc.enable_prefix_cache(ctx, wait=True)
        entry = gemini_cache._entries[svc._cache_key]
        entry.expires_at = gemini_cache.time.time() + 5  # inside safety window
        await drain(await svc._stream_content_universal_context(ctx))
        assert fake.sent_configs[-1].cached_content is None
        await asyncio.sleep(0.05)
        assert fake.create_calls == 2 and fake.delete_calls == ["cachedContents/test1"]

    asyncio.run(run())


def test_two_calls_share_one_cache():
    async def run():
        fake = FakeGenAI()
        a, b = make_service(fake), make_service(fake)
        await asyncio.gather(a.enable_prefix_cache(make_context(), wait=True),
                             b.enable_prefix_cache(make_context(), wait=True))
        assert fake.create_calls == 1  # second call reused the first call's cache

    asyncio.run(run())


def test_system_only_context_at_call_start_still_matches():
    """enable_prefix_cache() runs when the context holds ONLY the system message (call start)."""
    async def run():
        fake = FakeGenAI()
        svc = make_service(fake)
        ctx = LLMContext([{"role": "system", "content": SYSTEM}])
        await svc.enable_prefix_cache(ctx, wait=True)
        assert svc._cache_system_instruction == SYSTEM
        # first LLM request of the call (e.g. greeting) is built from the same system-only context
        await drain(await svc._stream_content_universal_context(ctx))
        cfg = fake.sent_configs[-1]
        assert cfg.cached_content == "cachedContents/test1"
        assert cfg.system_instruction is None and cfg.tools is None

    asyncio.run(run())
