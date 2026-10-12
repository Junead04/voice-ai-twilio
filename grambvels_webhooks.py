import httpx
import os
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

# ==================== CONFIG ====================
BASE_URL = "https://api.grambvels.ai/api/v1"
CLIENT_ID = os.getenv("GRAMBVELS_CLIENT_ID", "vc_live_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6")
CLIENT_SECRET = os.getenv("GRAMBVELS_CLIENT_SECRET", "vc-secret-2024-MyLeap-X7k9mPqR")
DESTINATION_NUMBER = "+0987654321"  # Registered demo number in staging DB

_access_token: str = None


# ==================== AUTH ====================
async def get_access_token() -> str:
    global _access_token
    if _access_token:
        return _access_token
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/auth/vendor-token",
                json={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET},
                timeout=10,
            )
            logger.info(f"Grambvels auth response: {resp.status_code} | body: {resp.text[:300]}")
            resp.raise_for_status()
            data = resp.json()
            # Try all common token key names
            _access_token = (
                data.get("access_token")
                or data.get("token")
                or data.get("accessToken")
                or data.get("data", {}).get("accessToken")
                or data.get("data", {}).get("access_token")
                or data.get("data", {}).get("token")
            )
            if _access_token:
                logger.info("Grambvels: access token fetched successfully")
            else:
                logger.error(f"Grambvels: token not found in response. Keys: {list(data.keys())}")
            return _access_token
    except Exception as e:
        logger.error(f"Grambvels: failed to get access token: {e}")
        return None


def _headers(token: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


# ==================== WEBHOOKS ====================
async def notify_call_started(call_id: str, caller_number: str):
    try:
        token = await get_access_token()
        if not token:
            logger.error("Grambvels call-started: skipped — no token")
            return
        async with httpx.AsyncClient() as client:
            payload = {
                "call_id": call_id,
                "caller_number": caller_number,
                "destination_number": DESTINATION_NUMBER,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            logger.info(f"Grambvels call-started payload: {payload}")
            resp = await client.post(
                f"{BASE_URL}/webhooks/call-started",
                headers=_headers(token),
                json=payload,
                timeout=10,
            )
            logger.info(f"Grambvels call-started: {resp.status_code} | {resp.text[:200]}")
    except Exception as e:
        logger.error(f"Grambvels call-started error: {e}")


async def notify_call_ended(call_id: str, caller_number: str):
    try:
        token = await get_access_token()
        if not token:
            logger.error("Grambvels call-ended: skipped — no token")
            return
        async with httpx.AsyncClient() as client:
            payload = {
                "call_id": call_id,
                "caller_number": caller_number,
                "destination_number": DESTINATION_NUMBER,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            logger.info(f"Grambvels call-ended payload: {payload}")
            resp = await client.post(
                f"{BASE_URL}/webhooks/call-ended",
                headers=_headers(token),
                json=payload,
                timeout=10,
            )
            logger.info(f"Grambvels call-ended: {resp.status_code} | {resp.text[:200]}")
    except Exception as e:
        logger.error(f"Grambvels call-ended error: {e}")


async def notify_call_transferred(call_id: str, caller_number: str, agent_number: str):
    try:
        token = await get_access_token()
        if not token:
            logger.error("Grambvels call-transferred: skipped — no token")
            return
        async with httpx.AsyncClient() as client:
            payload = {
                "call_id": call_id,
                "caller_number": caller_number,
                "destination_number": DESTINATION_NUMBER,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "agent_number": agent_number,
            }
            logger.info(f"Grambvels call-transferred payload: {payload}")
            resp = await client.post(
                f"{BASE_URL}/webhooks/call-transferred",
                headers=_headers(token),
                json=payload,
                timeout=10,
            )
            logger.info(f"Grambvels call-transferred: {resp.status_code} | {resp.text[:200]}")
    except Exception as e:
        logger.error(f"Grambvels call-transferred error: {e}")


async def notify_transcript(call_id: str, caller_number: str, messages: list):
    try:
        token = await get_access_token()
        if not token:
            logger.error("Grambvels transcript: skipped — no token")
            return
        async with httpx.AsyncClient() as client:
            payload = {
                "call_id": call_id,
                "caller_number": caller_number,
                "destination_number": DESTINATION_NUMBER,
                "messages": messages,
            }
            logger.info(f"Grambvels transcript: sending {len(messages)} messages")
            resp = await client.post(
                f"{BASE_URL}/webhooks/transcript",
                headers=_headers(token),
                json=payload,
                timeout=10,
            )
            logger.info(f"Grambvels transcript: {resp.status_code} | {resp.text[:200]}")
    except Exception as e:
        logger.error(f"Grambvels transcript error: {e}")