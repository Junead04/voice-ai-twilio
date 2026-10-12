"""
cache_probe.py -- answers, with YOUR key and YOUR prompt, whether explicit context
caching is worth turning on for gemini-3.1-flash-lite.

It (1) pulls the real system prompt + 8 tool declarations out of onetts.py,
(2) tries to create the cache and prints the exact token count / error,
(3) alternates cached vs uncached requests and compares time-to-first-chunk,
(4) deletes the cache.

    uv run python cache_probe.py --dry-run     # offline: only checks extraction
    uv run python cache_probe.py               # real run (needs GEMINI_API_KEY in .env)
    uv run python cache_probe.py --n 15 --model gemini-3.1-flash-lite
"""

import argparse
import asyncio
import os
import statistics as st
import textwrap
import time
from pathlib import Path

from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

UTTERANCES = [
    "Hi, I want to buy a hoodie.",
    "What caps do you have?",
    "I want to check the status of my order.",
    "Do you have t-shirts in large?",
]


def _slice(src: str, start: str, end: str) -> str:
    a = src.index(start)
    b = src.index(end, a)
    return textwrap.dedent(src[a:b])


def load_prompt_and_tools(path: str = "onetts.py"):
    """Reads the same objects run_bot() builds, so the probe tests the real prefix."""
    src = Path(path).read_text(encoding="utf-8")
    ns = {"os": os, "types": types}
    exec(_slice(src, "    function_declarations = [", "\n    llm = "), ns)
    exec(_slice(src, "    messages = [", "\n    context = LLMContext(messages)"), ns)
    system_prompt = ns["messages"][0]["content"]
    tools = [types.Tool(function_declarations=ns["function_declarations"])]
    return system_prompt, tools, len(ns["function_declarations"])


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p * (len(values) - 1))))]


async def timed_request(client, model, contents, config):
    t0 = time.perf_counter()
    first, usage = None, None
    stream = await client.aio.models.generate_content_stream(model=model, contents=contents, config=config)
    async for chunk in stream:
        if first is None:
            first = time.perf_counter() - t0
        if chunk.usage_metadata:
            usage = chunk.usage_metadata
    return first, time.perf_counter() - t0, usage


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--n", type=int, default=10, help="measured requests per arm")
    ap.add_argument("--file", default="onetts.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    system_prompt, tools, n_tools = load_prompt_and_tools(args.file)
    print(f"Extracted system prompt: {len(system_prompt)} chars, {n_tools} tool declarations")
    if args.dry_run:
        print("Dry run OK (no API calls made).")
        return

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    gen = dict(temperature=0.7, max_output_tokens=400)
    uncached_cfg = types.GenerateContentConfig(system_instruction=system_prompt, tools=tools, **gen)

    print(f"\n[1/3] Creating explicit cache on {args.model} ...")
    try:
        cache = await client.aio.caches.create(
            model=args.model,
            config=types.CreateCachedContentConfig(
                display_name="probe", system_instruction=system_prompt, tools=tools, ttl="600s"
            ),
        )
    except Exception as e:
        print("\n  RESULT: cache creation FAILED. Explicit caching is not usable here as-is.")
        print(f"  Reason from Google: {e}")
        print("  -> Common causes: prefix below the model's minimum token count, or model does not")
        print("     support explicit caching. Your bot will simply keep running uncached (no harm).")
        return
    print(f"  OK: {cache.name}  tokens cached: {cache.usage_metadata.total_token_count}")
    cached_cfg = types.GenerateContentConfig(cached_content=cache.name, **gen)

    try:
        print(f"\n[2/3] Benchmark: {args.n} measured requests per arm (alternating, 1 warm-up each)")
        res = {"uncached": [], "cached": []}
        usage_seen = {"uncached": [], "cached": []}
        for i in range(args.n + 1):
            contents = UTTERANCES[i % len(UTTERANCES)]
            order = ["uncached", "cached"] if i % 2 == 0 else ["cached", "uncached"]
            for arm in order:
                cfg = cached_cfg if arm == "cached" else uncached_cfg
                ttft, total, usage = await timed_request(client, args.model, contents, cfg)
                if i == 0:
                    continue  # warm-up, discard
                res[arm].append(ttft)
                usage_seen[arm].append((usage.cached_content_token_count or 0) if usage else 0)
                await asyncio.sleep(0.3)

        print("\n[3/3] Results (time to FIRST chunk, seconds)")
        print(f"  {'arm':10s} {'median':>8s} {'p90':>8s} {'min':>8s}   cached tokens reported")
        for arm in ("uncached", "cached"):
            v = res[arm]
            print(
                f"  {arm:10s} {st.median(v):8.3f} {pct(v, .9):8.3f} {min(v):8.3f}   "
                f"{int(st.median(usage_seen[arm]))}"
            )
        delta = st.median(res["uncached"]) - st.median(res["cached"])
        print(f"\n  Median improvement from explicit cache: {delta * 1000:+.0f} ms")
        if int(st.median(usage_seen["uncached"])) > 0:
            print("  NOTE: the 'uncached' arm already shows cached tokens -> implicit caching is active,")
            print("        so explicit caching has less left to save (mostly a cost guarantee).")
        print("  Rule of thumb: under ~50-80 ms difference is within run-to-run noise; then skip it.")
    finally:
        try:
            await client.aio.caches.delete(name=cache.name)
            print("\nProbe cache deleted.")
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
