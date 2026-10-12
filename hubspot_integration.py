"""
HubSpot CRM Integration for Call Data

This module integrates call data with HubSpot CRM, creating:
- Contacts from caller phone numbers
- Call activities/engagements
- Custom properties for call metrics
- Notes with transcripts
"""

import os
import asyncio
import httpx
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from loguru import logger
from dotenv import load_dotenv

load_dotenv(override=True)


class HubSpotClient:
    """
    HubSpot API client for managing contacts, calls, and engagements.
    """
    
    def __init__(self):
        self.api_key = os.getenv("HUBSPOT_API_KEY", "")
        self.base_url = "https://api.hubapi.com"
        self.timeout = 30.0
        
        if not self.api_key:
            logger.warning("⚠️  HubSpot API key not configured - integration disabled")
            self.enabled = False
        else:
            logger.info("✓ HubSpot integration enabled")
            self.enabled = True
    
    async def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request to HubSpot API"""
        if not self.enabled:
            logger.debug(f"📊 HubSpot disabled - would {method} {endpoint}")
            return None
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if method == "GET":
                    response = await client.get(url, headers=headers)
                elif method == "POST":
                    response = await client.post(url, headers=headers, json=data)
                elif method == "PATCH":
                    response = await client.patch(url, headers=headers, json=data)
                else:
                    logger.error(f"Unsupported method: {method}")
                    return None
                
                if response.status_code in [200, 201, 204]:
                    logger.debug(f"✓ HubSpot {method} {endpoint} successful")
                    return response.json() if response.text else {}
                else:
                    logger.error(f"✗ HubSpot API error: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    return None
                    
        except httpx.TimeoutException:
            logger.error(f"✗ HubSpot API timeout: {endpoint}")
            return None
        except Exception as e:
            logger.error(f"✗ HubSpot API error: {e}")
            return None
    
    async def find_contact_by_phone(self, phone_number: str) -> Optional[str]:
        """Find contact by phone number, returns contact ID if found"""
        # Clean phone number
        clean_phone = phone_number.replace("+", "").replace(" ", "").replace("-", "")
        
        # Search for contact
        endpoint = "/crm/v3/objects/contacts/search"
        search_data = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "phone",
                            "operator": "CONTAINS_TOKEN",
                            "value": clean_phone[-10:]  # Last 10 digits
                        }
                    ]
                }
            ],
            "properties": ["phone", "firstname", "lastname"]
        }
        
        result = await self._request("POST", endpoint, search_data)
        
        if result and result.get("results"):
            contact_id = result["results"][0]["id"]
            logger.info(f"✓ Found existing HubSpot contact: {contact_id}")
            return contact_id
        
        return None
    
    async def create_contact(self, phone_number: str, additional_properties: Optional[Dict] = None) -> Optional[str]:
        """Create a new contact in HubSpot"""
        properties = {
            "phone": phone_number,
            "lifecyclestage": "lead"
        }
        
        if additional_properties:
            properties.update(additional_properties)
        
        endpoint = "/crm/v3/objects/contacts"
        data = {"properties": properties}
        
        result = await self._request("POST", endpoint, data)
        
        if result and result.get("id"):
            contact_id = result["id"]
            logger.info(f"✓ Created HubSpot contact: {contact_id} for {phone_number}")
            return contact_id
        
        return None
    
    async def get_or_create_contact(self, phone_number: str) -> Optional[str]:
        """Get existing contact or create new one"""
        # Try to find existing contact
        contact_id = await self.find_contact_by_phone(phone_number)
        
        if contact_id:
            return contact_id
        
        # Create new contact
        return await self.create_contact(phone_number)
    
    async def create_call_engagement(
        self,
        contact_id: str,
        call_data: Dict[str, Any]
    ) -> Optional[str]:
        """Create a call engagement/activity in HubSpot"""
        
        # Prepare engagement data
        timestamp_ms = int(datetime.fromisoformat(call_data["timestamp"]).timestamp() * 1000)
        duration_ms = (call_data.get("duration", 0) or 0) * 1000  # Convert seconds to milliseconds
        
        properties = {
            "hs_timestamp": timestamp_ms,
            "hs_call_title": f"AI Bot Call - {call_data['status']}",
            "hs_call_body": f"Call ID: {call_data['call_id']}\nStatus: {call_data['status']}",
            "hs_call_duration": duration_ms,
            "hs_call_from_number": call_data.get("caller_number", ""),
            "hs_call_to_number": os.getenv("EXOTEL_VIRTUAL_NUMBER", ""),
            "hs_call_status": "COMPLETED" if call_data["status"] == "ended" else "NO_ANSWER",
            "hs_call_direction": "INBOUND"
        }
        
        # Create engagement
        endpoint = "/crm/v3/objects/calls"
        data = {
            "properties": properties,
            "associations": [
                {
                    "to": {"id": contact_id},
                    "types": [
                        {
                            "associationCategory": "HUBSPOT_DEFINED",
                            "associationTypeId": 194  # Call to Contact association
                        }
                    ]
                }
            ]
        }
        
        result = await self._request("POST", endpoint, data)
        
        if result and result.get("id"):
            engagement_id = result["id"]
            logger.info(f"✓ Created HubSpot call engagement: {engagement_id}")
            return engagement_id
        
        return None
    
    async def add_note_to_contact(
        self,
        contact_id: str,
        note_body: str,
        call_id: str
    ) -> Optional[str]:
        """Add a note (transcript) to a contact"""
        
        timestamp_ms = int(datetime.now().timestamp() * 1000)
        
        properties = {
            "hs_timestamp": timestamp_ms,
            "hs_note_body": note_body
        }
        
        endpoint = "/crm/v3/objects/notes"
        data = {
            "properties": properties,
            "associations": [
                {
                    "to": {"id": contact_id},
                    "types": [
                        {
                            "associationCategory": "HUBSPOT_DEFINED",
                            "associationTypeId": 202  # Note to Contact association
                        }
                    ]
                }
            ]
        }
        
        result = await self._request("POST", endpoint, data)
        
        if result and result.get("id"):
            note_id = result["id"]
            logger.info(f"✓ Created HubSpot note: {note_id}")
            return note_id
        
        return None
    
    async def update_contact_properties(
        self,
        contact_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """Update contact properties"""
        
        endpoint = f"/crm/v3/objects/contacts/{contact_id}"
        data = {"properties": properties}
        
        result = await self._request("PATCH", endpoint, data)
        
        if result:
            logger.info(f"✓ Updated HubSpot contact {contact_id}")
            return True
        
        return False


class HubSpotIntegration:
    """
    Main integration class that handles call data to HubSpot sync
    """
    
    def __init__(self):
        self.client = HubSpotClient()
        self.contact_cache = {}  # Cache contact IDs to avoid repeated lookups
        logger.info("✓ HubSpot Integration initialized")
    
    async def on_call_started(self, call_id: str, caller_number: str, timestamp: str):
        """Handle call started event"""
        if not self.client.enabled:
            return
        
        logger.info(f"📞 Syncing call start to HubSpot: {call_id}")
        
        # Get or create contact
        contact_id = await self.client.get_or_create_contact(caller_number)
        
        if contact_id:
            self.contact_cache[call_id] = contact_id
            logger.info(f"✓ Contact {contact_id} cached for call {call_id}")
    
    async def on_call_ended(
        self,
        call_id: str,
        caller_number: str,
        duration: int,
        status: str,
        timestamp: str
    ):
        """Handle call ended event"""
        if not self.client.enabled:
            return
        
        logger.info(f"📴 Syncing call end to HubSpot: {call_id}")
        
        # Get contact ID from cache or lookup
        contact_id = self.contact_cache.get(call_id)
        if not contact_id:
            contact_id = await self.client.get_or_create_contact(caller_number)
        
        if not contact_id:
            logger.error(f"✗ Could not find/create contact for {caller_number}")
            return
        
        # Create call engagement
        call_data = {
            "call_id": call_id,
            "caller_number": caller_number,
            "duration": duration,
            "status": status,
            "timestamp": timestamp
        }
        
        await self.client.create_call_engagement(contact_id, call_data)
    
    async def on_call_transferred(
        self,
        call_id: str,
        caller_number: str,
        agent_number: str,
        timestamp: str
    ):
        """Handle call transferred event"""
        if not self.client.enabled:
            return
        
        logger.info(f"🔄 Syncing call transfer to HubSpot: {call_id}")
        
        # Get contact ID
        contact_id = self.contact_cache.get(call_id)
        if not contact_id:
            contact_id = await self.client.get_or_create_contact(caller_number)
        
        if contact_id:
            # Add note about transfer
            note_body = f"Call {call_id} was transferred to human agent {agent_number} at {timestamp}"
            await self.client.add_note_to_contact(contact_id, note_body, call_id)
    
    async def on_transcript_ready(
        self,
        call_id: str,
        caller_number: str,
        transcript_data: Dict[str, Any]
    ):
        """Handle transcript ready event"""
        if not self.client.enabled:
            return
        
        logger.info(f"📝 Syncing transcript to HubSpot: {call_id}")
        
        # Get contact ID
        contact_id = self.contact_cache.get(call_id)
        if not contact_id:
            contact_id = await self.client.get_or_create_contact(caller_number)
        
        if not contact_id:
            logger.error(f"✗ Could not find/create contact for {caller_number}")
            return
        
        # Format transcript as note
        messages = transcript_data.get("messages", [])
        transcript_text = f"Call Transcript - {call_id}\n"
        transcript_text += f"Duration: {transcript_data.get('duration', 0)}s\n"
        transcript_text += f"Date: {transcript_data.get('start_time', '')}\n\n"
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            transcript_text += f"{role}: {content}\n\n"
        
        # Add transcript as note
        await self.client.add_note_to_contact(contact_id, transcript_text, call_id)
        
        # Clean up cache
        if call_id in self.contact_cache:
            del self.contact_cache[call_id]


# Global instance
_hubspot_integration: Optional[HubSpotIntegration] = None


def get_hubspot_integration() -> HubSpotIntegration:
    """Get or create global HubSpotIntegration instance"""
    global _hubspot_integration
    if _hubspot_integration is None:
        _hubspot_integration = HubSpotIntegration()
    return _hubspot_integration
