import os
import re
import asyncio
import json
import random
from typing import Optional, Dict
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger
import sys
import aiohttp

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(
    "logs/bot_debug_{time:YYYY-MM-DD_HH-mm-ss}.log",
    rotation="100 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair,LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.frames.frames import (
    Frame, TextFrame, LLMMessagesFrame, LLMFullResponseStartFrame,
    LLMFullResponseEndFrame, TTSSpeakFrame, EndFrame
)
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.services.google.llm import GoogleLLMService
from gemini_cache import CachedGoogleLLMService
from google.genai import types
from pipecat.transports.base_transport import BaseTransport
from pipecat.services.elevenlabs.stt import ElevenLabsSTTService
from pipecat_murf_tts import MurfTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from hubspot_inventory_orders import get_inventory_orders_manager
from grambvels_webhooks import notify_call_started, notify_call_ended, notify_call_transferred, notify_transcript

load_dotenv(override=True)

_last_detected_language = Language.EN

# ==================== PRODUCT CACHE ====================
_product_cache: list = []
_product_cache_loaded: bool = False

async def load_product_cache():
    global _product_cache, _product_cache_loaded
    try:
        manager = get_inventory_orders_manager()
        products = await manager.get_all_products()
        _product_cache = []
        for product in products:
            props = product.get("properties", {})
            _product_cache.append({
                "name": props.get("name"),
                "sku": props.get("hs_sku"),
                "price": props.get("price"),
                "description": props.get("description")
            })
        _product_cache_loaded = True
        logger.info(f"Product cache loaded: {len(_product_cache)} products")
    except Exception as e:
        logger.error(f"Failed to load product cache: {e}")
        _product_cache = []
        _product_cache_loaded = False

def _search_cache(query: str) -> list:
    query_lower = query.lower().strip()
    results = [p for p in _product_cache if query_lower in (p.get("name") or "").lower()]
    if not results and query_lower.endswith('s'):
        singular = query_lower[:-1]
        results = [p for p in _product_cache if singular in (p.get("name") or "").lower()]
    return results


# ==================== FRAME FLOW LOGGER ====================
class FrameFlowLogger(FrameProcessor):
    def __init__(self, position_name: str):
        super().__init__()
        self.position_name = position_name
        self.turn_counter = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        frame_type = type(frame).__name__
        dir_arrow = "→" if direction == FrameDirection.DOWNSTREAM else "←"
        skip_types = [
            'InputAudioRawFrame', 'OutputAudioRawFrame',
            'StartFrame', 'CancelFrame', 'StartInterruptionFrame',
            'StopInterruptionFrame', 'BotStoppedSpeakingFrame',
            'BotStartedSpeakingFrame', 'UserStartedSpeakingFrame',
            'UserStoppedSpeakingFrame', 'MetricsFrame','UserSpeakingFrame'
        ]
        if frame_type in skip_types:
            await self.push_frame(frame, direction)
            return
        logger.info(f"[{self.position_name}] {dir_arrow} {frame_type}")
        if isinstance(frame, TextFrame):
            logger.info(f"    📝 TEXT: '{frame.text[:100]}{'...' if len(frame.text) > 100 else ''}'")
        elif isinstance(frame, LLMMessagesFrame):
            self.turn_counter += 1
            logger.info(f"    🧠 LLM INPUT (Turn {self.turn_counter}):")
            logger.info(f"    " + "="*70)
            for i, msg in enumerate(frame.messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, str):
                    content_preview = content[:200] + "..." if len(content) > 200 else content
                elif isinstance(content, list):
                    content_preview = f"[{len(content)} parts]"
                else:
                    content_preview = str(content)[:200]
                logger.info(f"        [{i}] {role.upper()}: {content_preview}")
            logger.info(f"    " + "="*70)
            logger.info(f"    📊 Total messages in context: {len(frame.messages)}")
            # Rough token estimate: ~4 chars per token
            total_chars = sum(len(str(m.get('content', ''))) for m in frame.messages)
            estimated_tokens = total_chars // 4
            logger.info(f"    📊 Estimated tokens: ~{estimated_tokens}")
        elif isinstance(frame, TTSSpeakFrame):
            logger.info(f"    🔊 TTS: '{frame.text[:100]}{'...' if len(frame.text) > 100 else ''}'")
        elif isinstance(frame, EndFrame):
            logger.info(f"    🛑 END FRAME - Pipeline ending")
        elif frame_type == 'ErrorFrame':
            logger.error(f"    ❌ ERROR FRAME: {getattr(frame, 'error', 'Unknown error')}")
        await self.push_frame(frame, direction)


# ==================== TOOL CALL FILLER HELPER ====================
async def _speak_filler(params, text: str):
    await params.llm.push_frame(LLMFullResponseStartFrame())
    await params.llm.push_frame(TextFrame(text))
    await params.llm.push_frame(LLMFullResponseEndFrame())


# ==================== TRANSFER STATE MANAGEMENT ====================
_transfer_states: Dict[str, dict] = {}

def cleanup_old_transfer_states():
    global _transfer_states
    now = datetime.now()
    cutoff = now - timedelta(hours=1)
    old_states = [
        call_id for call_id, state in _transfer_states.items()
        if datetime.fromisoformat(state["timestamp"]) < cutoff
    ]
    for call_id in old_states:
        del _transfer_states[call_id]

def save_transfer_state(call_id: str, should_transfer: bool):
    _transfer_states[call_id] = {
        "should_transfer": should_transfer,
        "timestamp": datetime.now().isoformat()
    }
    if len(_transfer_states) % 100 == 0:
        cleanup_old_transfer_states()
    try:
        with open('transfer_state.json', 'w') as f:
            json.dump(_transfer_states, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving transfer state: {e}")

def get_transfer_state(call_id: str) -> bool:
    try:
        with open('transfer_state.json', 'r') as f:
            state = json.load(f)
            return state.get(call_id, {}).get("should_transfer", False)
    except (FileNotFoundError, json.JSONDecodeError):
        return _transfer_states.get(call_id, {}).get("should_transfer", False)


# ==================== HUBSPOT HELPERS ====================

async def search_products_by_name(query: str) -> list:
    if _product_cache_loaded and _product_cache:
        return _search_cache(query)
    logger.warning("Product cache empty, falling back to live HubSpot search")
    try:
        manager = get_inventory_orders_manager()
        products = await manager.search_products(query)
        results = []
        for product in products:
            props = product.get("properties", {})
            results.append({
                "name": props.get("name"),
                "sku": props.get("hs_sku"),
                "price": props.get("price"),
                "description": props.get("description")
            })
        return results
    except Exception as e:
        logger.error(f"Error searching products: {e}")
        return []


async def update_order_in_hubspot(order_id: str, update_fields: dict) -> bool:
    try:
        manager = get_inventory_orders_manager()
        order = await manager.get_order_by_name(order_id)
        if not order:
            logger.error(f"Order {order_id} not found")
            return False
        hubspot_internal_id = order.get("id")
        if not hubspot_internal_id:
            logger.error(f"Could not get HubSpot internal ID for order {order_id}")
            return False
        success = await manager.update_order(hubspot_internal_id, update_fields)
        if success:
            logger.info(f"Updated order {order_id} in HubSpot")
            return True
        return False
    except Exception as e:
        logger.error(f"Error updating order: {e}")
        return False


async def get_customer_orders(email: str) -> list:
    try:
        manager = get_inventory_orders_manager()
        orders = await manager.get_orders_by_email(email)
        results = []
        for order in orders:
            props = order.get("properties", {})
            results.append({
                "order_id": props.get("hs_order_name"),
                "total_price": props.get("hs_total_price"),
                "payment_status": props.get("hs_payment_status"),
                "order_date": props.get("createdate")
            })
        return results
    except Exception as e:
        logger.error(f"Error getting customer orders: {e}")
        return []


async def get_order_by_id(order_id: str) -> Optional[dict]:
    try:
        manager = get_inventory_orders_manager()
        order = await manager.get_order_by_name(order_id)
        if order:
            props = order.get("properties", {})
            return {
                "order_id": props.get("hs_order_name"),
                "email": props.get("hs_billing_address_email"),
                "sku": props.get("hs_tags"),
                "delivery_status": props.get("hs_external_order_status"),
                "order_date": props.get("hs_external_created_date")
            }
        return None
    except Exception as e:
        logger.error(f"Error getting order by ID: {e}")
        return None


async def get_all_products() -> list:
    if _product_cache_loaded and _product_cache:
        return _product_cache
    logger.warning("Product cache empty, falling back to live HubSpot call")
    try:
        manager = get_inventory_orders_manager()
        products = await manager.get_all_products()
        results = []
        for product in products:
            props = product.get("properties", {})
            results.append({
                "name": props.get("name"),
                "sku": props.get("hs_sku"),
                "price": props.get("price"),
                "description": props.get("description")
            })
        return results
    except Exception as e:
        logger.error(f"Error getting all products: {e}")
        return []


# ==================== FUNCTION HANDLERS ====================

async def handle_search_products_by_name(params):
    try:
        query = params.arguments.get("query")
        logger.info(f"🔧 TOOL: search_products_by_name(query='{query}')")
        await _speak_filler(params, "Let me look that up for you.")
        results = await search_products_by_name(query)
        if results:
            products_list = ", ".join([f"{p['name']} [SKU: {p['sku']}] (Rs.{int(float(p['price']))})" for p in results[:5]])
            response = f"Found {len(results)} products: {products_list}"
        else:
            response = f"No products found matching '{query}'"
        logger.info(f"    ✓ Found {len(results)} products")
        await params.result_callback(response)
    except Exception as e:
        logger.error(f"Tool error: {e}")
        await params.result_callback(f"Error searching products: {str(e)}")


async def handle_confirm_order(params):
    try:
        order_data = params.arguments
        logger.info(f"🔧 TOOL: confirm_order")

        customer_name = order_data.get('customer_name', '').strip()
        customer_email = order_data.get('customer_email', '').strip()
        customer_phone = order_data.get('customer_phone', '').strip()
        payment_method = order_data.get('payment_method', '').strip().upper()
        items = order_data.get('items', '').strip()

        missing = []
        if not customer_name:
            missing.append("full name")
        if not customer_email:
            missing.append("email address")
        if not customer_phone:
            missing.append("phone number")
        if not order_data.get('city', '').strip():
            missing.append("city")
        if not items:
            missing.append("product selection")
        if not order_data.get('total_price'):
            missing.append("total price (calculate unit price × quantity yourself)")
        if not payment_method:
            missing.append("payment method (UPI or Cash on Delivery)")
        elif payment_method not in ['UPI', 'COD']:
            missing.append("payment method (UPI or Cash on Delivery)")

        if missing:
            logger.warning(f"    ⚠️  Missing fields: {', '.join(missing)}")
            await params.result_callback(
                f"ERROR: Cannot place order yet. Still missing: {', '.join(missing)}. "
                f"Collect these from the customer first. Do NOT call confirm_order again until all are provided."
            )
            return

        # All fields present — now speak the filler and proceed
        await _speak_filler(params, "Placing your order now, one moment.")

        order_id = str(random.randint(100, 999))

        phone = customer_phone
        if phone and not phone.startswith('+'):
            phone = f"+91{phone}"
        payment_status = "CONFIRMED" if ('COD' in payment_method or 'CASH' in payment_method) else "PENDING"

        items = order_data.get('items', '')
        sku_value = items  # use item name(s) as SKU

        clean_order = {
            "order_id": order_id,
            "customer_name": customer_name,
            "customer_email": customer_email,
            "customer_phone": phone,
            "city": order_data.get('city', ''),
            "sku": sku_value,
            "items": items,
            "total_price": order_data.get('total_price', 0),
            "quantity": order_data.get('quantity', 1),
            "payment_method": payment_method,
            "payment_status": payment_status,
            "delivery_status": "Pending",
            "hubspot_uploaded": False,
            "created_at": datetime.now().isoformat()
        }

        try:
            with open('pending_orders.json', 'r') as f:
                orders = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            orders = []

        orders.append(clean_order)

        with open('pending_orders.json', 'w') as f:
            json.dump(orders, f, indent=2)

        logger.info(f"    ✓ Order {order_id} saved to pending_orders.json (will push to HubSpot post-call)")
        await params.result_callback(f"Order confirmed with order number {order_id}")

    except Exception as e:
        logger.error(f"Error confirming order: {e}")
        await params.result_callback(f"Error confirming order: {str(e)}")


async def handle_get_customer_orders(params):
    try:
        email = params.arguments.get("email")
        logger.info(f"🔧 TOOL: get_customer_orders(email='{email}')")
        await _speak_filler(params, "Let me check your order history.")
        orders = await get_customer_orders(email)
        if orders:
            orders_list = ", ".join([f"{o['order_id']} (Rs.{o['total_price']})" for o in orders])
            response = f"Found {len(orders)} orders: {orders_list}"
        else:
            response = f"No orders found for {email}"
        logger.info(f"    ✓ Found {len(orders)} orders")
        await params.result_callback(response)
    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        await params.result_callback(f"Error getting orders: {str(e)}")


async def handle_get_order_by_id(params):
    try:
        order_id = params.arguments.get("order_id")
        logger.info(f"🔧 TOOL: get_order_by_id(order_id='{order_id}')")
        await _speak_filler(params, "Let me pull up that order for you.")
        order = await get_order_by_id(order_id)
        if order:
            response = (
                f"Order {order['order_id']} found: "
                f"Email: {order['email']}, "
                f"SKU: {order['sku']}, "
                f"Delivery Status: {order['delivery_status']}, "
                f"Order Date: {order['order_date']}"
            )
        else:
            response = f"Order {order_id} not found"
        logger.info(f"    ✓ Order lookup: {response[:100]}")
        await params.result_callback(response)
    except Exception as e:
        logger.error(f"Error getting order by ID: {e}")
        await params.result_callback(f"Error getting order: {str(e)}")


async def handle_get_all_products(params):
    try:
        logger.info(f"🔧 TOOL: get_all_products")
        products = await get_all_products()
        if products:
            categories = set()
            for p in products:
                category = p.get('category')
                if category:
                    categories.add(category)
                else:
                    name_lower = p['name'].lower()
                    if 'jean' in name_lower or 'denim' in name_lower:
                        categories.add('Jeans')
                    elif 'shirt' in name_lower or 't-shirt' in name_lower or 'tshirt' in name_lower:
                        categories.add('T-Shirts')
                    elif 'hoodie' in name_lower or 'hood' in name_lower:
                        categories.add('Hoodies')
                    elif 'cap' in name_lower:
                        categories.add('Caps')
                    elif 'beanie' in name_lower or 'bean' in name_lower:
                        categories.add('Beanies')
            if categories:
                response = f"We have {', '.join(sorted(categories))} available"
            else:
                response = f"We have {len(products)} products available"
        else:
            response = "No products currently available in inventory"
        logger.info(f"    ✓ Product catalog: {len(products)} products")
        await params.result_callback(response)
    except Exception as e:
        logger.error(f"Error getting all products: {e}")
        await params.result_callback(f"Error retrieving products: {str(e)}")


async def handle_update_customer_order(params):
    try:
        order_id = params.arguments.get("order_id")
        logger.info(f"🔧 TOOL: update_customer_order(order_id='{order_id}')")
        await _speak_filler(params, "Updating your order details now.")

        order = await get_order_by_id(order_id)
        if not order:
            await params.result_callback(f"Order {order_id} not found")
            return

        delivery_status = order.get('delivery_status', '').lower()
        if delivery_status not in ['pending', '']:
            await params.result_callback(f"Sorry, order {order_id} has already {delivery_status}, cannot modify it now")
            return

        update_fields = {}
        if params.arguments.get('city'):
            update_fields['hs_shipping_address_city'] = params.arguments.get('city')
        if params.arguments.get('phone'):
            phone = params.arguments.get('phone')
            if not phone.startswith('+'):
                phone = f"+91{phone}"
            update_fields['hs_customer_phone'] = phone
        if params.arguments.get('email'):
            update_fields['hs_billing_address_email'] = params.arguments.get('email')
        if params.arguments.get('name'):
            update_fields['hs_billing_address_name'] = params.arguments.get('name')
        if params.arguments.get('items'):
            update_fields['hs_order_note'] = params.arguments.get('items')
        if params.arguments.get('sku'):
            sku_value = params.arguments.get('sku')
            update_fields['hs_tags'] = ", ".join(sku_value) if isinstance(sku_value, list) else sku_value
        if params.arguments.get('total_price'):
            update_fields['hs_total_price'] = params.arguments.get('total_price')
        if params.arguments.get('payment_method'):
            payment_method = params.arguments.get('payment_method').lower()
            update_fields['payment_processing_method'] = payment_method
            if 'cod' in payment_method or 'cash' in payment_method:
                update_fields['hs_payment_status'] = "CONFIRMED"
            else:
                update_fields['hs_payment_status'] = "PENDING"

        if not update_fields:
            await params.result_callback("No fields to update")
            return

        success = await update_order_in_hubspot(order_id, update_fields)
        if success:
            updated_items = []
            if 'hs_shipping_address_city' in update_fields:
                updated_items.append(f"city to {update_fields['hs_shipping_address_city']}")
            if 'hs_customer_phone' in update_fields:
                updated_items.append(f"phone to {update_fields['hs_customer_phone']}")
            if 'hs_billing_address_email' in update_fields:
                updated_items.append(f"email to {update_fields['hs_billing_address_email']}")
            if 'payment_processing_method' in update_fields:
                updated_items.append(f"payment method to {update_fields['payment_processing_method']}")
            if 'hs_order_note' in update_fields:
                updated_items.append("product details")
            response = f"Done, updated order {order_id}"
            if updated_items:
                response += f": {', '.join(updated_items)}"
            logger.info(f"    ✓ Order {order_id} updated")
            await params.result_callback(response)
        else:
            logger.error(f"    ✗ Failed to update order")
            await params.result_callback(f"Error updating order {order_id}, please try again")

    except Exception as e:
        logger.error(f"Error updating order: {e}")
        await params.result_callback(f"Error updating order: {str(e)}")


# ==================== POST-CALL ORDER PROCESSOR ====================

async def process_pending_orders():
    """Push all pending orders to HubSpot. Called after every call ends."""
    try:
        try:
            with open('pending_orders.json', 'r') as f:
                orders = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        unprocessed = [o for o in orders if not o.get("hubspot_uploaded")]
        if not unprocessed:
            return

        logger.info(f"Processing {len(unprocessed)} pending order(s) post-call...")

        for order in unprocessed:
            try:
                manager = get_inventory_orders_manager()
                hubspot_order_data = {
                    "hs_order_name": order.get("order_id"),
                    "hs_billing_address_email": order.get("customer_email"),
                    "hs_customer_phone": order.get("customer_phone"),
                    "hs_billing_address_name": order.get("customer_name"),
                    "hs_total_price": order.get("total_price", 0),
                    "hs_payment_status": order.get("payment_status", "PENDING"),
                    "payment_processing_method": order.get("payment_method", "").lower(),
                    "hs_order_note": order.get("items", ""),
                    "hs_tags": order.get("sku", ""),
                    "hs_shipping_address_city": order.get("city", ""),
                    "hs_external_order_status": "Pending"
                }
                hubspot_id = await manager.create_order(hubspot_order_data)
                if hubspot_id:
                    order["hubspot_uploaded"] = True
                    order["hubspot_id"] = hubspot_id
                    logger.info(f"    ✓ Order {order.get('order_id')} pushed to HubSpot: {hubspot_id}")
                else:
                    logger.error(f"    ✗ Failed to push order {order.get('order_id')} to HubSpot")
            except Exception as e:
                logger.error(f"    ✗ Error processing order {order.get('order_id')}: {e}")

        with open('pending_orders.json', 'w') as f:
            json.dump(orders, f, indent=2)

    except Exception as e:
        logger.error(f"Critical error in process_pending_orders: {e}")


# ==================== TRANSFER & END CALL HANDLERS ====================
_task_ref: dict = {"task": None}

async def handle_transfer_to_agent(params):
    try:
        call_id = _task_ref.get("call_id")
        caller_number = _task_ref.get("caller_number")
        transport_type = _task_ref.get("transport_type")
        agent_number = "+919606517174"
        logger.info(f"🔧 TOOL: transfer_to_agent (call_id={call_id}, transport={transport_type})")
        if call_id:
            save_transfer_state(call_id, should_transfer=True)
        asyncio.create_task(notify_call_transferred(call_id, caller_number or call_id, agent_number))

        async def _trigger():
            await asyncio.sleep(4.0)
            if transport_type == "twilio":
                try:
                    from twilio.rest import Client
                    client = Client(
                        os.getenv("TWILIO_ACCOUNT_SID"),
                        os.getenv("TWILIO_AUTH_TOKEN"),
                        region="ie1",
                        edge="dublin",
                    )
                    client.calls(call_id).update(
                        twiml=f"""<Response>
                        <Dial callerId="+448002294233" answerOnBridge="true" timeout="30">
                            <Number>{agent_number}</Number>
                        </Dial>
                    </Response>"""
                    )
                    logger.info(f"Twilio call {call_id} redirected to agent {agent_number}")
                except Exception as e:
                    logger.error(f"Twilio transfer redirect failed: {e}")
                    task = _task_ref.get("task")
                    if task:
                        await task.queue_frame(EndFrame())
            else:
                task = _task_ref.get("task")
                if task:
                    await task.queue_frame(EndFrame())
                    logger.info("EndFrame queued for transfer")

        asyncio.create_task(_trigger())
        await params.result_callback("")
    except Exception as e:
        logger.error(f"Error in transfer_to_agent: {e}")
        await params.result_callback("")


async def handle_end_call(params):
    try:
        call_id = _task_ref.get("call_id")
        logger.info(f"🔧 TOOL: end_call (call_id={call_id})")
        if call_id:
            save_transfer_state(call_id, should_transfer=False)
        async def _disconnect():
            await asyncio.sleep(4.0)
            task = _task_ref.get("task")
            if task:
                await task.queue_frame(EndFrame())
                logger.info("EndFrame queued for end_call")
        asyncio.create_task(_disconnect())
        await params.result_callback("")
    except Exception as e:
        logger.error(f"Error in end_call: {e}")
        await params.result_callback("")


# ==================== LANGUAGE DETECTION ====================
def detect_language(text: str) -> Language:
    global _last_detected_language
    if not text:
        return _last_detected_language
    text_for_detection = re.sub(r'[\s\d\.,!?;:\'"\-()]+', '', text)
    if not text_for_detection:
        return _last_detected_language
    if re.search(r'[\u0B80-\u0BFF]', text_for_detection):
        _last_detected_language = Language.TA
        return Language.TA
    if re.search(r'[\u0C00-\u0C7F]', text_for_detection):
        _last_detected_language = Language.TE
        return Language.TE
    if re.search(r'[\u0C80-\u0CFF]', text_for_detection):
        _last_detected_language = Language.KN
        return Language.KN
    if re.search(r'[\u0D00-\u0D7F]', text_for_detection):
        _last_detected_language = Language.ML
        return Language.ML
    if re.search(r'[\u0900-\u097F]', text_for_detection):
        _last_detected_language = Language.HI
        return Language.HI
    _last_detected_language = Language.EN
    return Language.EN


async def run_bot(transport: BaseTransport, handle_sigint: bool, call_id: str = None, stream_sid: str = None, caller_number: str = None, transport_type: str = None):
    from google.genai import types

    bot_is_speaking = False
    _task_ref["call_id"] = call_id
    _task_ref["caller_number"] = caller_number
    _task_ref["transport_type"] = transport_type
    _mark_sequence = 0
    transcript_messages = []

    function_declarations = [
        types.FunctionDeclaration(
            name="search_products_by_name",
            description="Search for products by name or category (e.g., jeans, tshirt, dress)",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for product name or category"}
                },
                "required": ["query"]
            }
        ),
        types.FunctionDeclaration(
            name="confirm_order",
            description="Call this once you have collected ALL customer details: name, email, phone, city, product, and payment method. Confirms the order with the customer and saves it.",
            parameters={
                "type": "object",
                "properties": {
                    "customer_email": {"type": "string", "description": "Customer email address"},
                    "customer_phone": {"type": "string", "description": "Customer phone number"},
                    "customer_name": {"type": "string", "description": "Customer full name"},
                    "total_price": {"type": "number", "description": "Total order price in rupees"},
                    "payment_method": {"type": "string", "description": "Payment method: UPI or COD. MUST be explicitly stated by the customer — never assume.", "enum": ["UPI", "COD"]},
                    "items": {"type": "string", "description": "Product name(s) ordered, e.g. 'Classic Black Hoodie'"},
                    "sku": {"type": "string", "description": "Same as items — the product name(s) ordered"},
                    "quantity": {"type": "number", "description": "Quantity ordered"},
                    "city": {"type": "string", "description": "Shipping city"}
                },
                "required": ["customer_email", "customer_name", "customer_phone", "payment_method", "items", "total_price"]
            }
        ),
        types.FunctionDeclaration(
            name="get_customer_orders",
            description="Get all orders for a customer by their email address",
            parameters={
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Customer email address"}
                },
                "required": ["email"]
            }
        ),
        types.FunctionDeclaration(
            name="get_order_by_id",
            description="Get order details by order ID",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID (e.g., ORD301)"}
                },
                "required": ["order_id"]
            }
        ),
        types.FunctionDeclaration(
            name="get_all_products",
            description="Get all available products from inventory",
            parameters={"type": "object", "properties": {}, "required": []}
        ),
        types.FunctionDeclaration(
            name="update_customer_order",
            description="Update order details like city, phone, email, product, or payment method",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID to update"},
                    "city": {"type": "string", "description": "New delivery city (optional)"},
                    "phone": {"type": "string", "description": "New phone number (optional)"},
                    "email": {"type": "string", "description": "New email address (optional)"},
                    "name": {"type": "string", "description": "New customer name (optional)"},
                    "items": {"type": "string", "description": "New product description (optional)"},
                    "sku": {"type": "string", "description": "New product SKU (optional)"},
                    "total_price": {"type": "number", "description": "New total price (optional)"},
                    "payment_method": {"type": "string", "enum": ["UPI", "COD"], "description": "New payment method (optional)"}
                },
                "required": ["order_id"]
            }
        ),
        types.FunctionDeclaration(
            name="transfer_to_agent",
            description="Transfer the customer to a human support agent.",
            parameters={"type": "object", "properties": {}, "required": []}
        ),
        types.FunctionDeclaration(
            name="end_call",
            description="End the call after a proper goodbye.",
            parameters={"type": "object", "properties": {}, "required": []}
        ),
    ]

    llm = CachedGoogleLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-3.1-flash-lite",
        params=GoogleLLMService.InputParams(
            temperature=0.7,
            max_tokens=400,
            stream=True
        ),
        tools=[types.Tool(function_declarations=function_declarations)]
    )

    llm.register_function("search_products_by_name", handle_search_products_by_name)
    llm.register_function("confirm_order", handle_confirm_order)
    llm.register_function("get_customer_orders", handle_get_customer_orders)
    llm.register_function("get_order_by_id", handle_get_order_by_id)
    llm.register_function("get_all_products", handle_get_all_products)
    llm.register_function("update_customer_order", handle_update_customer_order)
    llm.register_function("transfer_to_agent", handle_transfer_to_agent)
    llm.register_function("end_call", handle_end_call)

    logger.info("Registered 8 function handlers")

    # ==================== MURF AI TTS SERVICES ====================
    _murf_params = dict(
        voice_id="Aarav",
        style="Conversational",
        model="falcon-2",
        multi_native_locale="en-IN",
        sample_rate=8000,
        format="PCM",
    )

    english_tts = MurfTTSService(
        api_key=os.getenv("MURF_API_KEY"),
        url=os.getenv("MURF_WS_URL", "wss://uk.api.murf.ai/v1/speech/stream-input"),
        params=MurfTTSService.InputParams(**_murf_params),
    )

    aiohttp_session = aiohttp.ClientSession()
    stt = ElevenLabsSTTService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        model="scribe_v2",
        aiohttp_session=aiohttp_session,
    )

    async def handle_user_idle(processor):
        nonlocal bot_is_speaking
        if bot_is_speaking:
            return True
        idle_count = getattr(processor, '_idle_count', 0)
        processor._idle_count = idle_count + 1
        if idle_count == 0:
            await processor.push_frame(LLMFullResponseStartFrame())
            await processor.push_frame(TextFrame("Are you still there?"))
            await processor.push_frame(LLMFullResponseEndFrame())
            return True
        elif idle_count == 1:
            await processor.push_frame(LLMFullResponseStartFrame())
            await processor.push_frame(TextFrame("Hello, are you there, would you like to continue."))
            await processor.push_frame(LLMFullResponseEndFrame())
            return True
        else:
            await processor.push_frame(LLMFullResponseStartFrame())
            await processor.push_frame(TextFrame("Thank you for calling Sash Clothing, have a nice day."))
            await processor.push_frame(LLMFullResponseEndFrame())
            return False

    user_idle = UserIdleProcessor(callback=handle_user_idle, timeout=10.0)

    messages = [
        {
            "role": "system",
            "content": (
                "You are Aarav, a customer support agent for Sash Clothing, an Indian apparel brand.\n\n"

                "LANGUAGE HANDLING:\n"
                "- ALWAYS respond in English only, regardless of what language the customer uses\n"
                "- Do not switch languages under any circumstances\n"
                "- Supported: English only\n\n"

                "PRONUNCIATION & FORMATTING:\n"
                "- Currency: Always say 'rupeace' (Rs.) - never 'rupaya' or 'rupaye'\n"
                "- Time ranges: Use 'to' between numbers - '5 to 7 business days', '3 to 5 days'\n"
                "- Email addresses: @ as 'at', . as 'dot' (example: 'Aarav at sash dot com')\n"
                "- Phone numbers: Speak digits clearly with natural pauses\n"
                "- Order IDs: Speak as individual digits (e.g., 'order 3 0 1' not 'order three zero one')\n"
                "- NUMBERS IN TEXT: Always write numbers as words\n\n"

                "TOOLS YOU HAVE:\n"
                "1. get_all_products() - Shows product categories\n"
                "2. search_products_by_name(query) - Gets products by category\n"
                "3. confirm_order(order_data) - Call once you have ALL customer details collected. Saves the order and gives customer their order number.\n"
                "4. get_order_by_id(order_id) - Looks up order status\n"
                "5. get_customer_orders(email) - Gets customer order history\n"
                "6. update_customer_order(order_id, fields) - Update order details before shipment\n"
                "7. transfer_to_agent() - Transfer customer to a human agent (call when customer requests it or issue needs human expertise). IMPORTANT: Always say your transfer message OUT LOUD first (e.g. 'Let me connect you to our support team, please hold on.'), then call this tool.\n"
                "8. end_call() - End the call. IMPORTANT: Always say your farewell OUT LOUD first (e.g. 'Thank you for calling Sash Clothing, have a nice day!'), then call this tool. Never call this tool without speaking a goodbye phrase first.\n\n"

                "SMART TOOL USAGE:\n"
                "- When asked for 'lowest price', 'starting price', 'cheapest', or 'most affordable': Use get_all_products() ONCE, then analyze all prices yourself to find the minimum\n"
                "- When asked for 'highest price', 'most expensive', or 'maximum price': Use get_all_products() ONCE, then analyze all prices yourself to find the maximum\n"
                "- When asked for 'price range': Use get_all_products() ONCE, then calculate min and max yourself\n"
                "- DO NOT make multiple search_products_by_name() calls when you can get everything at once with get_all_products()\n"
                "- After getting all products, YOU analyze the data - don't make the customer wait through multiple searches\n\n"

                "Note: SKU codes are for internal use only - NEVER mention SKU codes to customers. When calling confirm_order(), pass the product name(s) the customer ordered into both the 'items' and 'sku' fields.\n"
                "Note: Keep acknowledgments brief, natural, and conversational\n\n"

                "POLICY GUIDELINES:\n"
                "Understanding our policies helps you guide customers effectively:\n\n"

                "Returns & Exchanges:\n"
                "- Return window: 7 days from delivery for standard returns\n"
                "- Items must be unused, unwashed, with original tags and packaging\n"
                "- Returns accepted for: wrong size, defective items, damaged during shipping, or not as described\n"
                "- Exchange process: For wrong sizes, offer direct replacement - no need for full return cycle\n"
                "- Defective items: Request photo/video proof, then process replacement or refund based on customer preference\n"
                "- Return shipping: Free for defective/damaged items; customer covers shipping for other reasons\n\n"

                "Cancellations:\n"
                "- Orders with 'Pending' status: Can be cancelled immediately with full refund\n"
                "- Orders with 'Shipped' status: Cannot be cancelled from our end - offer refuse delivery option for automatic refund\n"
                "- Refund timeline: 5-7 business days after item return or cancellation confirmation\n\n"

                "Modifications:\n"
                "- Orders can be modified (address, phone, items, payment method) only while status is 'Pending'\n"
                "- Once 'Shipped', modifications are not possible - order must be completed or refused at delivery\n"
                "- Use update_customer_order() tool to make changes to pending orders\n\n"

                "Delivery:\n"
                "- Standard delivery: 5-7 business days from order confirmation\n"
                "- Express delivery: Currently not available through standard ordering\n"
                "- Delivery delays: Share current status and realistic timeline - if customer wants to cancel due to delay, follow cancellation policy\n"
                "- Tracking: Customers receive tracking details via email once order ships\n\n"

                "Payments & Refunds:\n"
                "- Payment methods accepted: UPI, Cash on Delivery (COD)\n"
                "- UPI: We send a payment link via SMS after order placement. Customer uses any UPI app (Google Pay, PhonePe, Paytm) to complete payment. Order confirmed after payment verification.\n"
                "- COD: Customer pays cash when receiving delivery. Order confirmed immediately at placement.\n"
                "- Refunds: Processed to original payment method within 5 to 7 business days\n"
                "- Payment issues: Connect customer with payments team for transaction failures or double charges\n\n"

                "Pricing & Discounts:\n"
                "- All prices shown are final - includes applicable taxes\n"
                "- Product prices are retrieved from inventory - never invent or modify prices\n"
                "- Special discounts or custom pricing: Requires sales team approval - transfer to specialist\n"
                "- Bulk orders: Transfer to B2B team for corporate/bulk pricing discussions\n\n"

                "Policy Exceptions:\n"
                "- Situations outside standard policy: Connect customer with supervisor who has approval authority\n"
                "- Frame transfers positively: 'Let me connect you with someone who can review your specific case'\n\n"

                "What You Can Do Directly:\n"
                "- Process standard returns and replacements within policy window\n"
                "- Cancel orders with 'Pending' status\n"
                "- Modify pending orders (address, phone, items, payment method)\n"
                "- Share order status and tracking information\n"
                "- Answer product questions using inventory tools\n"
                "- Place new orders after collecting all required details\n\n"

                "When to Escalate:\n"
                "- Return requests beyond 7-day window\n"
                "- Cancellation requests for shipped orders (except refuse delivery option)\n"
                "- Payment disputes, refund amount questions, or transaction issues\n"
                "- Custom pricing or special discount requests\n"
                "- Any situation where customer insists after you've explained policy twice\n\n"

                "CONVERSATION STYLE - CRITICAL:\n"
                "- You're a professional but RELAXED support agent who knows their job perfectly\n"
                "- Sound NATURAL and CONVERSATIONAL - like a knowledgeable friend helping out\n"
                "- NEVER repeat customer details back to them\n"
                "- Vary your acknowledgments - don't sound repetitive\n"
                "- Keep responses SHORT and TO THE POINT - maximum 1-2 sentences\n"
                "- Don't over-explain - customers trust you know what you're doing\n\n"

                "INTENT IDENTIFICATION:\n"
                "Identify if query is pre-sales or post-sales\n"
                "If multiple issues mentioned, resolve post-sales first\n\n"

                "PRE-SALES (New Orders):\n"
                "- Use tools to show products - NEVER make up product names, prices, or suggest items we don't sell\n"
                "- We ONLY sell: Caps, Hoodies, T-Shirts (nothing else)\n"
                "- Flow: get_all_products() → customer picks category → search_products_by_name() → LIST ALL products with names and prices\n"
                "- When search_products_by_name returns results, YOU MUST list EVERY product found with its name and price\n"
                "- Mention price ONLY ONCE when first listing products - NEVER mention price again to the customer, but ALWAYS pass the correct price into confirm_order() as total_price\n"
                "- For T-Shirts/Hoodies: Ask size once. ALL sizes are ALWAYS available - never check inventory for sizes\n"
                "- For Caps: Skip size (one-size-fits-all)\n"
                "- Collect these details in order:\n"
                "  1. Ask for name and email together\n"
                "  2. Customer responds → acknowledge briefly\n"
                "  3. Ask for phone number and city together\n"
                "  4. Customer responds → acknowledge briefly\n"
                "  5. Ask for payment preference (UPI or Cash on delivery)\n"
                "  6. Customer responds → IMMEDIATELY place order\n"
                "- REMEMBER context - if customer already mentioned something, DON'T ask again\n"
                "- Once you have all 5 details (name, phone, email, city, payment), call confirm_order() immediately\n"
                "- TOTAL PRICE: Calculate it yourself before calling confirm_order(). Formula: unit price × quantity. Example: Cap Snapback Grey at Rs.449 × 2 = Rs.898. Pass this as total_price. NEVER leave total_price empty or zero.\n"
                "- CRITICAL: NEVER invent, guess, or assume any field. Payment method (UPI or COD) MUST be explicitly chosen by the customer — never assume COD or UPI. Do NOT call confirm_order() unless the customer has spoken the words 'UPI', 'Google Pay', 'PhonePe', 'Paytm', 'cash', or 'COD' in this conversation. If payment preference is missing, STOP and ask: 'Would you like to pay via UPI or Cash on Delivery?' before doing anything else.\n"
                "- When calling confirm_order(), pass the product name(s) the customer selected into BOTH the 'items' AND 'sku' fields.\n"
                "- After order confirmed, give customer their order number and let them know they will receive a confirmation shortly, then ask if they need anything else\n\n"

                "POST-SALES (Existing Orders):\n"
                "- Ask for order ID → call get_order_by_id(order_id)\n"
                "- For simple queries (status check): NO verification needed\n"
                "- For sensitive actions (cancel, refund, return, replacement): Ask for email to verify\n\n"

                "A) ORDER STATUS: Share status and expected delivery, ask if they need anything else\n\n"

                "B) CANCELLATION:\n"
                "   - If status is 'Pending': Confirm cancellation and refund timeline\n"
                "   - If status is 'Shipped': Cannot cancel - offer refuse delivery option\n"
                "   - If they keep insisting: Connect with specialist\n\n"

                "C) RETURN/REPLACEMENT:\n"
                "   - Wrong size: Offer replacement with correct size\n"
                "   - Defect: Request photo/video proof, then process\n"
                "   - Other: Process return and share refund timeline\n\n"

                "D) ORDER MODIFICATION:\n"
                "   - If 'Pending': Make the update, confirm done\n"
                "   - If 'Shipped': Cannot modify\n\n"

                "When transferring:\n"
                "- 'This needs specialist attention - let me connect you with our team'\n"
                "- 'Let me get you to someone who can fully help with this'\n\n"

                "CONVERSATION CLOSING:\n"
                "- After resolving ANY issue: Ask if they need anything else\n"
                "- If customer says no: Close with appreciation\n\n"

                "CRITICAL RULES:\n"
                "- LANGUAGE: Always respond in English only — never switch languages\n"
                "- Always use tools for product info - never fabricate data\n"
                "- NEVER mention SKU codes to customers\n"
                "- NEVER check inventory for sizes - all sizes always available\n"
                "- NEVER repeat customer details back\n"
                "- TRACK CONTEXT - don't ask for info they already provided\n"
                "- Keep responses brief - 1-2 sentences maximum\n\n"

                "Think like a skilled support agent having a natural conversation, not someone reading a script."
            ),
        },
    ]

    context = LLMContext(messages)
    if os.getenv("GEMINI_PREFIX_CACHE", "1") != "0":
        await llm.enable_prefix_cache(context)
    context_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(aggregation_timeout=0.2),
    )

    transcript = TranscriptProcessor()
    os.makedirs("transcripts", exist_ok=True)
    call_id_safe = call_id.replace('/', '_') if call_id else "unknown"
    transcript_filename = f"transcripts/transcript_{call_id_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    @transcript.event_handler("on_transcript_update")
    async def save_transcript(processor, frame):
        for message in frame.messages:
            timestamp = message.timestamp or datetime.now().isoformat()
            transcript_messages.append({
                "timestamp": timestamp,
                "role": message.role,
                "content": message.content
            })
            logger.info(f"{message.role.upper()}: {message.content}")
        with open(transcript_filename, "w", encoding="utf-8") as f:
            json.dump(transcript_messages, f, indent=2)

    logger_after_stt = FrameFlowLogger("AFTER_STT")
    logger_before_llm = FrameFlowLogger("BEFORE_LLM")
    logger_after_llm = FrameFlowLogger("AFTER_LLM")
    logger_before_tts = FrameFlowLogger("BEFORE_TTS")
    logger_after_tts = FrameFlowLogger("AFTER_TTS")

    pipeline = Pipeline([
        transport.input(),
        stt,
        logger_after_stt,
        transcript.user(),
        user_idle,
        context_aggregator.user(),
        logger_before_llm,
        llm,
        logger_after_llm,
        logger_before_tts,
        english_tts,
        logger_after_tts,
        transport.output(),
        transcript.assistant(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    _task_ref["task"] = task

    @transport.event_handler("on_bot_started_speaking")
    async def on_bot_started_speaking(transport):
        nonlocal bot_is_speaking
        bot_is_speaking = True
        logger.info("🗣️  BOT STARTED SPEAKING")

    @transport.event_handler("on_bot_stopped_speaking")
    async def on_bot_stopped_speaking(transport):
        nonlocal bot_is_speaking, _mark_sequence
        bot_is_speaking = False
        logger.info("🤐 BOT STOPPED SPEAKING")
        try:
            _mark_sequence += 1
            mark_event = {
                "event": "mark",
                "sequence_number": _mark_sequence,
                "stream_sid": stream_sid or "",
                "mark": {
                    "name": f"audio_chunk_{_mark_sequence}"
                }
            }
            await transport._websocket.send_text(json.dumps(mark_event))
            logger.info(f"📍 MARK sent: sequence={_mark_sequence}, name=audio_chunk_{_mark_sequence}")
        except Exception as e:
            logger.error(f"Failed to send mark event: {e}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"📞 CALL CONNECTED | Call ID: {call_id}")
        asyncio.create_task(notify_call_started(call_id, caller_number or call_id))
        with open(transcript_filename, "w", encoding="utf-8") as f:
            json.dump([], f)
        greeting = "Hi, I am Aarav from Sash Clothing, how may I assist you today?"
        await task.queue_frames([LLMFullResponseStartFrame(), TextFrame(greeting), LLMFullResponseEndFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"📞 CALL DISCONNECTED | Call ID: {call_id}")
        with open(transcript_filename, "w", encoding="utf-8") as f:
            json.dump(transcript_messages, f, indent=2)
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    tts_services = [english_tts]

    async def _tts_keepalive():
        while True:
            await asyncio.sleep(90)
            for svc in tts_services:
                try:
                    ws = getattr(svc, "_websocket", None)
                    if ws and not getattr(ws, "closed", True):
                        await ws.ping()
                except Exception:
                    pass

    keepalive_task = asyncio.create_task(_tts_keepalive())
    try:
        await runner.run(task)
    finally:
        keepalive_task.cancel()
        try:
            await keepalive_task
        except asyncio.CancelledError:
            pass
        # Always push pending orders to HubSpot after call ends,
        # regardless of whether user hung up, bot ended, or transfer occurred
        logger.info("Call ended — processing pending orders...")
        await process_pending_orders()
        await aiohttp_session.close()
        await notify_call_ended(call_id, caller_number or call_id)
        await notify_transcript(call_id, caller_number or call_id, transcript_messages)


async def bot(runner_args: RunnerArguments):
    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Auto-detected transport: {transport_type}")
    logger.info(f"Full call_data: {call_data}")
    caller_number = call_data.get("from") or call_data.get("caller_number") or call_data.get("from_number")

    if transport_type == "twilio":
        from pipecat.serializers.twilio import TwilioFrameSerializer
        serializer = TwilioFrameSerializer(
            stream_sid=call_data["stream_id"],
            call_sid=call_data["call_id"],
            account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        )
    else:
        serializer = ExotelFrameSerializer(
            stream_sid=call_data["stream_id"],
            call_sid=call_data["call_id"],
        )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )
    handle_sigint = runner_args.handle_sigint
    await run_bot(transport, handle_sigint, call_data["call_id"], stream_sid=call_data["stream_id"], caller_number=caller_number,transport_type=transport_type)


if __name__ == "__main__":
    import argparse
    from fastapi import FastAPI, WebSocket, Request, Response
    from fastapi.responses import JSONResponse, PlainTextResponse
    import uvicorn

    os.makedirs("logs", exist_ok=True)
    os.makedirs("transcripts", exist_ok=True)

    app = FastAPI()

    @app.api_route("/twilio/incoming-call", methods=["GET", "POST"])
    async def twilio_incoming_call():
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="wss://pastrami-unleash-boil.ngrok-free.dev/ws" />
        </Connect>
    </Response>"""
        return Response(content=twiml, media_type="application/xml")

    @app.on_event("startup")
    async def startup_event():
        await load_product_cache()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        runner_args = type('RunnerArguments', (), {'websocket': websocket, 'handle_sigint': False})()
        await bot(runner_args)

    @app.get("/check_transfer/{call_sid:path}")
    @app.get("/check_transfer")
    async def check_transfer_endpoint(request: Request, call_sid: str = None):
        if not call_sid:
            call_sid = request.query_params.get("CallSid")
        should_transfer = get_transfer_state(call_sid) if call_sid else False
        logger.info(f"Passthru: CallSid={call_sid}, transfer={should_transfer}")
        return JSONResponse({"status": "ok", "call_id": call_sid})

    @app.get("/get_transfer_number")
    async def get_transfer_number(request: Request):
        call_sid = request.query_params.get("CallSid")
        should_transfer = get_transfer_state(call_sid) if call_sid else False
        if should_transfer:
            logger.info(f"Returning agent number for {call_sid}")
            return PlainTextResponse(content="+919606517174", media_type="text/plain")
        else:
            return Response(status_code=404, content="No transfer")

    @app.get("/health")
    async def health_check():
        return JSONResponse({"status": "ok", "service": "bot+passthru"})

    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="exotel")
    parser.add_argument("--proxy", required=True)
    args = parser.parse_args()

    logger.info(f"Starting bot on port 7860 | Proxy: {args.proxy}")
    uvicorn.run(app, host="0.0.0.0", port=7860)