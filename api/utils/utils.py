# api/utils/utils.py
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def normalize_timestamp(timestamp_value):
    """
    Handle different timestamp formats and convert to ISO format string with Z.
    """
    if timestamp_value is None:
        return datetime.now(timezone.utc).isoformat() + 'Z'
        
    if isinstance(timestamp_value, (int, float)):
        # If timestamp is numeric (Unix timestamp), convert to ISO string
        dt = datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
        return dt.isoformat() + 'Z'
    elif isinstance(timestamp_value, str):
        # Fix double timezone indicator: either remove Z or +00:00
        if timestamp_value.endswith('Z') and '+00:00' in timestamp_value:
            timestamp_value = timestamp_value.replace('+00:00Z', 'Z')
            
        # Try parsing directly
        try:
            # Handle Z suffix by replacing with +00:00 for fromisoformat
            if timestamp_value.endswith('Z'):
                dt = datetime.fromisoformat(timestamp_value[:-1] + '+00:00')
            else:
                dt = datetime.fromisoformat(timestamp_value)
            
            # Ensure timezone is UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
                
            return dt.isoformat() + 'Z'
        except ValueError:
            # Last resort fallback - try parsing with more lenient approach
            logger.warning(f"Failed to parse timestamp with standard methods: {timestamp_value}")
            return datetime.now(timezone.utc).isoformat() + 'Z'
    elif isinstance(timestamp_value, datetime):
        # Already a datetime, ensure timezone and convert to ISO
        if timestamp_value.tzinfo is None:
            timestamp_value = timestamp_value.replace(tzinfo=timezone.utc)
        return timestamp_value.isoformat() + 'Z'
    else:
        # Invalid format, log and return current time
        logger.error(f"Unsupported timestamp format: {type(timestamp_value)}")
        return datetime.now(timezone.utc).isoformat() + 'Z'