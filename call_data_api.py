"""
Call Data API Integration Module

This module handles real-time call data collection and transmission to external APIs/webhooks.
It captures call events, metrics, and transcripts, then pushes them to configured endpoints.
"""

import os
import asyncio
import httpx
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger


class CallStatus(Enum):
    """Call status enumeration"""
    RINGING = "ringing"
    ANSWERED = "answered"
    ACTIVE = "active"
    ENDED = "ended"
    MISSED = "missed"
    ABANDONED = "abandoned"
    TRANSFERRED = "transferred"


@dataclass
class CallEvent:
    """Represents a call event"""
    event_type: str  # call_started, call_ended, call_transferred, status_changed
    call_id: str
    timestamp: str
    caller_number: str
    status: str
    duration: Optional[int] = None  # in seconds
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CallMetrics:
    """Aggregated call metrics"""
    total_calls_today: int
    active_calls_count: int  # Number of calls happening RIGHT NOW
    total_call_time: int  # in seconds
    successful_calls: int
    abandoned_calls: int
    average_duration: float
    timestamp: str


@dataclass
class TranscriptMessage:
    """Individual transcript message"""
    timestamp: str
    role: str  # USER or ASSISTANT
    content: str


@dataclass
class TranscriptData:
    """Complete transcript data"""
    call_id: str
    caller_number: str
    start_time: str
    end_time: str
    duration: int
    transcript_file_path: str
    messages: List[TranscriptMessage]


class CallDataCollector:
    """
    Collects call data and manages state for active calls.
    Tracks call events, metrics, and live call information.
    """
    
    def __init__(self):
        self.active_calls: Dict[str, Dict[str, Any]] = {}
        self.call_history: List[Dict[str, Any]] = []  # Store completed calls
        self.recent_activity: List[Dict[str, Any]] = []  # Store recent events
        self.max_history_size = 50  # Keep last 50 calls
        self.max_activity_size = 100  # Keep last 100 events
        self.daily_metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "abandoned_calls": 0,
            "total_duration": 0,
            "call_durations": []
        }
        self._reset_date = datetime.now().date()
    
    def _check_date_reset(self):
        """Reset daily metrics if date has changed"""
        current_date = datetime.now().date()
        if current_date != self._reset_date:
            logger.info("📅 New day detected - resetting daily metrics")
            self.daily_metrics = {
                "total_calls": 0,
                "successful_calls": 0,
                "abandoned_calls": 0,
                "total_duration": 0,
                "call_durations": []
            }
            self._reset_date = current_date
    
    def start_call(self, call_id: str, caller_number: str, stream_id: Optional[str] = None, call_type: str = "inbound") -> CallEvent:
        """Record call start event"""
        self._check_date_reset()
        
        timestamp = datetime.now().isoformat()
        self.active_calls[call_id] = {
            "call_id": call_id,
            "caller_number": caller_number,
            "stream_id": stream_id,
            "start_time": timestamp,
            "status": CallStatus.ANSWERED.value,
            "transcript_file": None,
            "call_type": call_type  # "inbound" or "outbound"
        }
        
        self.daily_metrics["total_calls"] += 1
        
        # Add to recent activity
        self._add_activity({
            "timestamp": timestamp,
            "event_type": "call_started",
            "description": f"New call from {caller_number}",
            "call_id": call_id,
            "call_type": call_type,  # Add call_type to activity
            "details": {"status": CallStatus.ANSWERED.value}
        })
        
        logger.info(f"📞 Call started: {call_id} from {caller_number}")
        
        return CallEvent(
            event_type="call_started",
            call_id=call_id,
            timestamp=timestamp,
            caller_number=caller_number,
            status=CallStatus.ANSWERED.value,
            metadata={"stream_id": stream_id}
        )
    
    def update_call_status(self, call_id: str, new_status: CallStatus) -> Optional[CallEvent]:
        """Update call status"""
        if call_id not in self.active_calls:
            logger.warning(f"⚠️  Attempted to update unknown call: {call_id}")
            return None
        
        call_data = self.active_calls[call_id]
        old_status = call_data["status"]
        call_data["status"] = new_status.value
        
        timestamp = datetime.now().isoformat()
        
        logger.info(f"🔄 Call {call_id} status: {old_status} → {new_status.value}")
        
        return CallEvent(
            event_type="status_changed",
            call_id=call_id,
            timestamp=timestamp,
            caller_number=call_data["caller_number"],
            status=new_status.value,
            metadata={"previous_status": old_status}
        )
    
    def end_call(self, call_id: str, final_status: Optional[CallStatus] = None) -> Optional[CallEvent]:
        """Record call end event"""
        if call_id not in self.active_calls:
            logger.warning(f"⚠️  Attempted to end unknown call: {call_id}")
            return None
        
        call_data = self.active_calls[call_id]
        end_time = datetime.now()
        start_time = datetime.fromisoformat(call_data["start_time"])
        duration = int((end_time - start_time).total_seconds())
        
        # Determine final status
        if final_status is None:
            final_status = CallStatus.ENDED
        
        call_data["end_time"] = end_time.isoformat()
        call_data["duration"] = duration
        call_data["status"] = final_status.value
        
        # Update metrics
        self.daily_metrics["total_duration"] += duration
        self.daily_metrics["call_durations"].append(duration)
        
        if final_status == CallStatus.ENDED:
            self.daily_metrics["successful_calls"] += 1
        elif final_status == CallStatus.ABANDONED:
            self.daily_metrics["abandoned_calls"] += 1
        
        # Add to call history
        self._add_to_history({
            "call_id": call_id,
            "caller_number": call_data["caller_number"],
            "start_time": call_data["start_time"],
            "end_time": call_data["end_time"],
            "duration": duration,
            "status": final_status.value,
            "transcript_file": call_data.get("transcript_file"),
            "transferred_to": call_data.get("transferred_to"),
            "call_type": call_data.get("call_type", "inbound")  # Add call_type to history
        })
        
        # Add to recent activity
        duration_formatted = self._format_duration(duration)
        self._add_activity({
            "timestamp": end_time.isoformat(),
            "event_type": "call_ended",
            "description": f"Call ended - Duration: {duration_formatted}",
            "call_id": call_id,
            "call_type": call_data.get("call_type", "inbound"),  # Add call_type to activity
            "details": {"duration": duration, "status": final_status.value}
        })
        
        logger.info(f"📴 Call ended: {call_id} - Duration: {duration}s - Status: {final_status.value}")
        
        event = CallEvent(
            event_type="call_ended",
            call_id=call_id,
            timestamp=end_time.isoformat(),
            caller_number=call_data["caller_number"],
            status=final_status.value,
            duration=duration,
            metadata={
                "start_time": call_data["start_time"],
                "transcript_file": call_data.get("transcript_file")
            }
        )
        
        # Remove from active calls
        del self.active_calls[call_id]
        
        return event
    
    def record_transfer(self, call_id: str, agent_number: str) -> Optional[CallEvent]:
        """Record call transfer event"""
        if call_id not in self.active_calls:
            logger.warning(f"⚠️  Attempted to record transfer for unknown call: {call_id}")
            return None
        
        call_data = self.active_calls[call_id]
        call_data["status"] = CallStatus.TRANSFERRED.value
        call_data["transferred_to"] = agent_number
        
        timestamp = datetime.now().isoformat()
        
        # Add to recent activity
        self._add_activity({
            "timestamp": timestamp,
            "event_type": "call_transferred",
            "description": f"Call transferred to {agent_number}",
            "call_id": call_id,
            "call_type": call_data.get("call_type", "inbound"),  # Add call_type to activity
            "details": {"agent_number": agent_number}
        })
        
        logger.info(f"🔄 Call {call_id} transferred to {agent_number}")
        
        return CallEvent(
            event_type="call_transferred",
            call_id=call_id,
            timestamp=timestamp,
            caller_number=call_data["caller_number"],
            status=CallStatus.TRANSFERRED.value,
            metadata={"agent_number": agent_number}
        )
    
    def set_transcript_file(self, call_id: str, transcript_file: str):
        """Associate transcript file with call"""
        if call_id in self.active_calls:
            self.active_calls[call_id]["transcript_file"] = transcript_file
            logger.info(f"📝 Transcript file set for {call_id}: {transcript_file}")
    
    def _add_to_history(self, call_record: Dict[str, Any]):
        """Add call to history (keep last N calls)"""
        self.call_history.insert(0, call_record)  # Add to beginning
        if len(self.call_history) > self.max_history_size:
            self.call_history = self.call_history[:self.max_history_size]
    
    def _add_activity(self, activity: Dict[str, Any]):
        """Add activity to recent activity feed (keep last N activities)"""
        self.recent_activity.insert(0, activity)  # Add to beginning
        if len(self.recent_activity) > self.max_activity_size:
            self.recent_activity = self.recent_activity[:self.max_activity_size]
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def get_current_metrics(self) -> CallMetrics:
        """Get current aggregated metrics"""
        self._check_date_reset()
        
        avg_duration = 0.0
        if self.daily_metrics["call_durations"]:
            avg_duration = sum(self.daily_metrics["call_durations"]) / len(self.daily_metrics["call_durations"])
        
        return CallMetrics(
            total_calls_today=self.daily_metrics["total_calls"],
            active_calls_count=len(self.active_calls),  # Number of calls happening RIGHT NOW
            total_call_time=self.daily_metrics["total_duration"],
            successful_calls=self.daily_metrics["successful_calls"],
            abandoned_calls=self.daily_metrics["abandoned_calls"],
            average_duration=round(avg_duration, 2),
            timestamp=datetime.now().isoformat()
        )
    
    def get_live_calls(self) -> List[Dict[str, Any]]:
        """Get list of currently active calls"""
        live_calls = []
        current_time = datetime.now()
        
        for call_id, call_data in self.active_calls.items():
            start_time = datetime.fromisoformat(call_data["start_time"])
            elapsed = int((current_time - start_time).total_seconds())
            
            live_calls.append({
                "call_id": call_id,
                "caller_number": call_data["caller_number"],
                "status": call_data["status"],
                "duration": elapsed,
                "start_time": call_data["start_time"],
                "stream_id": call_data.get("stream_id"),
                "call_type": call_data.get("call_type", "inbound")  # Add call_type to live calls
            })
        
        return live_calls
    
    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get call history (completed calls)"""
        return self.call_history.copy()
    
    def get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent activity feed"""
        return self.recent_activity.copy()


class APIClient:
    """
    Handles API communication with external endpoints.
    Sends call events, metrics, and transcript data to configured APIs.
    """
    
    def __init__(self):
        # Load configuration from environment
        self.base_url = os.getenv("DASHBOARD_API_URL", "")
        self.api_key = os.getenv("DASHBOARD_API_KEY", "")
        self.timeout = float(os.getenv("API_TIMEOUT", "10.0"))
        self.retry_attempts = int(os.getenv("API_RETRY_ATTEMPTS", "2"))
        self.enabled = bool(self.base_url)  # Only enabled if URL is configured
        
        if not self.enabled:
            logger.warning("⚠️  Dashboard API not configured - data will only be logged locally")
        else:
            logger.info(f"✓ Dashboard API configured: {self.base_url}")
    
    async def _send_request(self, endpoint: str, data: Dict[str, Any]) -> bool:
        """Send HTTP POST request to API endpoint"""
        if not self.enabled:
            logger.debug(f"📊 API disabled - would send to {endpoint}: {data}")
            return True
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        for attempt in range(self.retry_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=data, headers=headers)
                    
                    if response.status_code in [200, 201, 202]:
                        logger.info(f"✓ API request successful: {endpoint}")
                        return True
                    else:
                        logger.error(f"✗ API request failed: {endpoint} - Status: {response.status_code}")
                        logger.error(f"Response: {response.text}")
                        
            except httpx.TimeoutException:
                logger.error(f"✗ API request timeout: {endpoint} (attempt {attempt + 1}/{self.retry_attempts})")
            except Exception as e:
                logger.error(f"✗ API request error: {endpoint} - {e}")
            
            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(1)  # Wait before retry
        
        return False
    
    async def send_call_started(self, event: CallEvent) -> bool:
        """Send call started event"""
        data = asdict(event)
        return await self._send_request("/api/call-started", data)
    
    async def send_call_ended(self, event: CallEvent) -> bool:
        """Send call ended event"""
        data = asdict(event)
        return await self._send_request("/api/call-ended", data)
    
    async def send_call_transferred(self, event: CallEvent) -> bool:
        """Send call transferred event"""
        data = asdict(event)
        return await self._send_request("/api/call-transferred", data)
    
    async def send_status_changed(self, event: CallEvent) -> bool:
        """Send status changed event"""
        data = asdict(event)
        return await self._send_request("/api/status-changed", data)
    
    async def send_metrics(self, metrics: CallMetrics) -> bool:
        """Send aggregated metrics"""
        data = asdict(metrics)
        return await self._send_request("/api/metrics", data)
    
    async def send_transcript(self, transcript: TranscriptData) -> bool:
        """Send transcript data"""
        data = asdict(transcript)
        return await self._send_request("/api/transcript", data)
    
    async def send_live_calls(self, live_calls: List[Dict[str, Any]]) -> bool:
        """Send live calls snapshot"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "active_calls": live_calls
        }
        return await self._send_request("/api/live-calls", data)
    
    async def send_call_history(self, call_history: List[Dict[str, Any]]) -> bool:
        """Send call history"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "call_history": call_history
        }
        return await self._send_request("/api/call-history", data)
    
    async def send_recent_activity(self, recent_activity: List[Dict[str, Any]]) -> bool:
        """Send recent activity feed"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "recent_activity": recent_activity
        }
        return await self._send_request("/api/recent-activity", data)


class CallDataManager:
    """
    Main manager class that coordinates data collection and API transmission.
    Provides high-level interface for tracking calls and sending data.
    """
    
    def __init__(self):
        self.collector = CallDataCollector()
        self.api_client = APIClient()
        
        # Import HubSpot integration
        try:
            from hubspot_integration import get_hubspot_integration
            self.hubspot = get_hubspot_integration()
            self.hubspot_enabled = True
        except Exception as e:
            logger.warning(f"⚠️  HubSpot integration not available: {e}")
            self.hubspot = None
            self.hubspot_enabled = False
        
        # Import Dashboard Data Exporter
        try:
            from dashboard_data_exporter import get_dashboard_exporter
            self.dashboard_exporter = get_dashboard_exporter()
            self.dashboard_export_enabled = True
        except Exception as e:
            logger.warning(f"⚠️  Dashboard exporter not available: {e}")
            self.dashboard_exporter = None
            self.dashboard_export_enabled = False
        
        logger.info("✓ Call Data Manager initialized")
    
    async def on_call_started(self, call_id: str, caller_number: str, stream_id: Optional[str] = None, call_type: str = "inbound"):
        """Handle call start event"""
        event = self.collector.start_call(call_id, caller_number, stream_id, call_type)
        await self.api_client.send_call_started(event)
        
        # Send to HubSpot
        if self.hubspot_enabled and self.hubspot:
            await self.hubspot.on_call_started(call_id, caller_number, event.timestamp)
        
        # Send updated metrics
        metrics = self.collector.get_current_metrics()
        await self.api_client.send_metrics(metrics)
        
        # Send live calls, history, and activity
        await self._send_all_data()
        
        # Export dashboard data
        await self._export_dashboard_data()
    
    async def on_call_ended(self, call_id: str, final_status: Optional[CallStatus] = None):
        """Handle call end event"""
        event = self.collector.end_call(call_id, final_status)
        if event:
            await self.api_client.send_call_ended(event)
            
            # Send to HubSpot
            if self.hubspot_enabled and self.hubspot:
                await self.hubspot.on_call_ended(
                    call_id,
                    event.caller_number,
                    event.duration or 0,
                    event.status,
                    event.timestamp
                )
            
            # Send updated metrics
            metrics = self.collector.get_current_metrics()
            await self.api_client.send_metrics(metrics)
            
            # Send live calls, history, and activity
            await self._send_all_data()
            
            # Export dashboard data
            await self._export_dashboard_data()
            # Send updated metrics
            metrics = self.collector.get_current_metrics()
            await self.api_client.send_metrics(metrics)
            
            # Export dashboard data
            await self._export_dashboard_data()
    
    async def on_call_transferred(self, call_id: str, agent_number: str):
        """Handle call transfer event"""
        event = self.collector.record_transfer(call_id, agent_number)
        if event:
            await self.api_client.send_call_transferred(event)
            
            # Send to HubSpot
            if self.hubspot_enabled and self.hubspot:
                await self.hubspot.on_call_transferred(
                    call_id,
                    event.caller_number,
                    agent_number,
                    event.timestamp
                )
    
    async def on_status_changed(self, call_id: str, new_status: CallStatus):
        """Handle status change event"""
        event = self.collector.update_call_status(call_id, new_status)
        if event:
            await self.api_client.send_status_changed(event)
    
    def set_transcript_file(self, call_id: str, transcript_file: str):
        """Associate transcript file with call"""
        self.collector.set_transcript_file(call_id, transcript_file)
    
    async def send_transcript(self, call_id: str, transcript_file: str):
        """Parse and send transcript data"""
        try:
            # Read transcript file
            with open(transcript_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Parse transcript
            lines = content.split("\n")
            messages = []
            call_info = {}
            
            for line in lines:
                if line.startswith("CallSID:"):
                    call_info["call_id"] = line.split(":", 1)[1].strip()
                elif line.startswith("Customer:"):
                    call_info["caller_number"] = line.split(":", 1)[1].strip()
                elif line.startswith("Started:"):
                    call_info["start_time"] = line.split(":", 1)[1].strip()
                elif line.startswith("Ended:"):
                    call_info["end_time"] = line.split(":", 1)[1].strip()
                elif line.startswith("[") and ("] USER:" in line or "] ASSISTANT:" in line):
                    # Parse message line
                    parts = line.split("] ", 1)
                    if len(parts) == 2:
                        timestamp = parts[0][1:]  # Remove leading [
                        rest = parts[1]
                        if ": " in rest:
                            role, content = rest.split(": ", 1)
                            messages.append(TranscriptMessage(
                                timestamp=timestamp,
                                role=role,
                                content=content
                            ))
            
            # Calculate duration
            if "start_time" in call_info and "end_time" in call_info:
                start = datetime.fromisoformat(call_info["start_time"])
                end = datetime.fromisoformat(call_info["end_time"])
                duration = int((end - start).total_seconds())
            else:
                duration = 0
            
            # Create transcript data
            transcript_data = TranscriptData(
                call_id=call_info.get("call_id", call_id),
                caller_number=call_info.get("caller_number", "unknown"),
                start_time=call_info.get("start_time", ""),
                end_time=call_info.get("end_time", ""),
                duration=duration,
                transcript_file_path=transcript_file,
                messages=messages
            )
            
            # Send to API
            await self.api_client.send_transcript(transcript_data)
            
            # Send to HubSpot
            if self.hubspot_enabled and self.hubspot:
                await self.hubspot.on_transcript_ready(
                    call_id,
                    transcript_data.caller_number,
                    asdict(transcript_data)
                )
            
            logger.info(f"✓ Transcript sent for call {call_id}")
            
        except Exception as e:
            logger.error(f"❌ Error sending transcript for {call_id}: {e}")
    
    async def send_live_calls_update(self):
        """Send current live calls snapshot"""
        live_calls = self.collector.get_live_calls()
        await self.api_client.send_live_calls(live_calls)
        
        # Export dashboard data
        await self._export_dashboard_data()
    
    async def _send_all_data(self):
        """Send live calls, history, and activity to API"""
        try:
            # Get all data
            live_calls = self.collector.get_live_calls()
            call_history = self.collector.get_call_history()
            recent_activity = self.collector.get_recent_activity()
            
            # Send to API
            await self.api_client.send_live_calls(live_calls)
            await self.api_client.send_call_history(call_history)
            await self.api_client.send_recent_activity(recent_activity)
        except Exception as e:
            logger.error(f"✗ Failed to send all data: {e}")
    
    async def _export_dashboard_data(self):
        """Export complete dashboard data to JSON file"""
        if not self.dashboard_export_enabled or not self.dashboard_exporter:
            return
        
        try:
            # Get current data
            metrics_obj = self.collector.get_current_metrics()
            metrics = asdict(metrics_obj)
            live_calls = self.collector.get_live_calls()
            call_history = self.collector.get_call_history()
            recent_activity = self.collector.get_recent_activity()
            
            # Export to JSON
            self.dashboard_exporter.export_dashboard_data(
                metrics=metrics,
                live_calls=live_calls,
                call_history=call_history,
                recent_activity=recent_activity
            )
        except Exception as e:
            logger.error(f"✗ Failed to export dashboard data: {e}")
    
    def get_metrics(self) -> CallMetrics:
        """Get current metrics"""
        return self.collector.get_current_metrics()
    
    def get_live_calls(self) -> List[Dict[str, Any]]:
        """Get live calls list"""
        return self.collector.get_live_calls()


# Global instance
_call_data_manager: Optional[CallDataManager] = None


def get_call_data_manager() -> CallDataManager:
    """Get or create global CallDataManager instance"""
    global _call_data_manager
    if _call_data_manager is None:
        _call_data_manager = CallDataManager()
    return _call_data_manager
