"""
Dashboard Data Exporter

Exports call data in JSON format for frontend dashboard consumption.
Creates a single JSON file with all dashboard metrics, live calls, and call history.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
from loguru import logger


class DashboardDataExporter:
    """
    Exports call data to JSON format for dashboard frontend.
    Creates a structured JSON file with all necessary dashboard data.
    """
    
    def __init__(self, output_dir: str = "dashboard_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.current_file = os.path.join(output_dir, "dashboard_data.json")
        logger.info(f"✓ Dashboard Data Exporter initialized - Output: {self.current_file}")
    
    def export_dashboard_data(
        self,
        metrics: Dict[str, Any],
        live_calls: List[Dict[str, Any]],
        call_history: List[Dict[str, Any]],
        recent_activity: List[Dict[str, Any]]
    ):
        """
        Export complete dashboard data to JSON file.
        
        Args:
            metrics: Dashboard metrics (total calls, duration, success rate, etc.)
            live_calls: Currently active calls
            call_history: Historical call records
            recent_activity: Recent activity feed
        """
        
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            
            # Dashboard Metrics
            "metrics": {
                "total_calls_today": metrics.get("total_calls", 0),
                "active_calls_count": len(live_calls),  # Number of calls happening RIGHT NOW
                "total_call_time_seconds": metrics.get("total_duration", 0),
                "total_call_time_formatted": self._format_duration(metrics.get("total_duration", 0)),
                "successful_calls": metrics.get("successful_calls", 0),
                "abandoned_calls": metrics.get("abandoned_calls", 0),
                "success_rate_percentage": self._calculate_success_rate(
                    metrics.get("successful_calls", 0),
                    metrics.get("total_calls", 0)
                ),
                "abandoned_rate_percentage": self._calculate_abandoned_rate(
                    metrics.get("abandoned_calls", 0),
                    metrics.get("total_calls", 0)
                ),
                "average_call_duration_seconds": metrics.get("average_duration", 0),
                "average_call_duration_formatted": self._format_duration(metrics.get("average_duration", 0))
            },
            
            # Live Call Monitor
            "live_calls": [
                {
                    "call_id": call.get("call_id"),
                    "caller_number": call.get("caller_number"),
                    "status": call.get("status"),
                    "start_time": call.get("start_time"),
                    "duration_seconds": call.get("duration", 0),
                    "duration_formatted": self._format_duration(call.get("duration", 0)),
                    "stream_id": call.get("stream_id")
                }
                for call in live_calls
            ],
            
            # Call History
            "call_history": [
                {
                    "call_id": call.get("call_id"),
                    "caller_number": call.get("caller_number"),
                    "start_time": call.get("start_time"),
                    "end_time": call.get("end_time"),
                    "duration_seconds": call.get("duration", 0),
                    "duration_formatted": self._format_duration(call.get("duration", 0)),
                    "status": call.get("status"),
                    "transcript_file": call.get("transcript_file"),
                    "transferred_to": call.get("transferred_to")
                }
                for call in call_history
            ],
            
            # Recent Activity Feed
            "recent_activity": [
                {
                    "timestamp": activity.get("timestamp"),
                    "event_type": activity.get("event_type"),
                    "description": activity.get("description"),
                    "call_id": activity.get("call_id"),
                    "details": activity.get("details", {})
                }
                for activity in recent_activity
            ]
        }
        
        # Write to JSON file
        try:
            with open(self.current_file, "w", encoding="utf-8") as f:
                json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Dashboard data exported to {self.current_file}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to export dashboard data: {e}")
            return False
    
    def export_call_event(self, event_type: str, call_data: Dict[str, Any]):
        """
        Export individual call event to separate JSON file.
        Useful for real-time updates.
        
        Args:
            event_type: Type of event (call_started, call_ended, etc.)
            call_data: Call event data
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        event_file = os.path.join(self.output_dir, f"{event_type}_{timestamp}.json")
        
        event_data = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": call_data
        }
        
        try:
            with open(event_file, "w", encoding="utf-8") as f:
                json.dump(event_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"✓ Event exported: {event_file}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to export event: {e}")
            return False
    
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
    
    def _calculate_success_rate(self, successful: int, total: int) -> float:
        """Calculate success rate percentage"""
        if total == 0:
            return 0.0
        return round((successful / total) * 100, 2)
    
    def _calculate_abandoned_rate(self, abandoned: int, total: int) -> float:
        """Calculate abandoned rate percentage"""
        if total == 0:
            return 0.0
        return round((abandoned / total) * 100, 2)


# Global instance
_dashboard_exporter = None


def get_dashboard_exporter() -> DashboardDataExporter:
    """Get or create global DashboardDataExporter instance"""
    global _dashboard_exporter
    if _dashboard_exporter is None:
        _dashboard_exporter = DashboardDataExporter()
    return _dashboard_exporter
