"""
Local test client for onetts.py — simulates an Exotel Voicebot call using your
mic and speakers. No Exotel, Twilio, or phone number needed.

It speaks Exotel's exact WebSocket handshake (connected -> start -> media chunks
of raw 16-bit/8kHz/mono PCM, base64-encoded) so onetts.py's parse_telephony_websocket
detects it as a real Exotel call.

Install (Windows, inside your project's venv):
    pip install websockets sounddevice numpy

Run (in a second terminal, after `uv run python onetts.py --proxy anything` is up):
    python test_call_client.py

Use headphones — this does no echo cancellation, so speaker bleed into the mic
will confuse the bot's turn-taking.
Ctrl+C to hang up.
"""

import asyncio
import base64
import json
import time
import uuid

import numpy as np
import sounddevice as sd
import websockets

WS_URL = "ws://localhost:7860/ws"
SAMPLE_RATE = 8000
CHUNK_MS = 100
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000  # 800 samples = 1600 bytes (16-bit)

SILENCE_RMS_THRESHOLD = 300   # tune up/down based on your mic's noise floor
SILENCE_HANG_MS = 600         # quiet duration before we consider "you stopped talking"

stream_sid = f"teststream{uuid.uuid4().hex[:20]}"
call_sid = f"testcall{uuid.uuid4().hex[:20]}"

seq = 0
last_speech_end = None
awaiting_response = False


def rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


async def mic_sender(ws):
    global seq, last_speech_end, awaiting_response
    silence_ms = 0
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def callback(indata, frames, time_info, status):
        loop.call_soon_threadsafe(q.put_nowait, indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        blocksize=CHUNK_SAMPLES, callback=callback,
    ):
        print("Mic live -- start talking (Ctrl+C to hang up)")
        while True:
            chunk = await q.get()
            level = rms(chunk)
            if level > SILENCE_RMS_THRESHOLD:
                silence_ms = 0
                last_speech_end = None
            else:
                silence_ms += CHUNK_MS
                if silence_ms >= SILENCE_HANG_MS and last_speech_end is None:
                    last_speech_end = time.time()
                    awaiting_response = True

            seq += 1
            msg = {
                "event": "media",
                "sequence_number": seq,
                "stream_sid": stream_sid,
                "media": {
                    "chunk": seq,
                    "timestamp": str(seq * CHUNK_MS),
                    "payload": base64.b64encode(chunk.tobytes()).decode(),
                },
            }
            await ws.send(json.dumps(msg))


async def bot_audio_receiver(ws):
    global awaiting_response
    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as out:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event = msg.get("event")
            if event == "media":
                pcm = np.frombuffer(base64.b64decode(msg["media"]["payload"]), dtype="int16")
                out.write(pcm)
                if awaiting_response and last_speech_end is not None:
                    latency_ms = (time.time() - last_speech_end) * 1000
                    print(f"[latency] silence -> first bot audio byte: {latency_ms:.0f} ms")
                    awaiting_response = False
            elif event == "clear":
                print("[bot interrupted -- playback cleared]")
            # "mark" events (bot finished speaking a turn) are ignored here


async def main():
    async with websockets.connect(WS_URL, max_size=None) as ws:
        await ws.send(json.dumps({"event": "connected"}))
        await ws.send(json.dumps({
            "event": "start",
            "sequence_number": 1,
            "stream_sid": stream_sid,
            "start": {
                "stream_sid": stream_sid,
                "call_sid": call_sid,
                "account_sid": "test_account",
                "from": "+911234567890",
                "to": "+919999999999",
                "custom_parameters": {},
                "media_format": {"encoding": "slin", "sample_rate": "8000", "bit_rate": "16"},
            },
        }))
        print(f"Call started -- stream_sid={stream_sid}")
        try:
            await asyncio.gather(mic_sender(ws), bot_audio_receiver(ws))
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        except websockets.exceptions.ConnectionClosedOK:
            print("Bot ended the call.")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed unexpectedly: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nHanging up...")
