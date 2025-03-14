# api/utils/utils.py
import logging
logging.basicConfig(level=logging.INFO)
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def normalize_timestamp(timestamp_value):
    """Handle different timestamp formats and convert to ISO format string with Z."""
    if timestamp_value is None:
        return datetime.now(timezone.utc).isoformat() + 'Z'
        
    if isinstance(timestamp_value, (int, float)):
        # If timestamp is numeric (Unix timestamp), convert to ISO string
        dt = datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
        return dt.isoformat() + 'Z'
    elif isinstance(timestamp_value, str):
        # If timestamp is string, normalize and convert
        if 'Z' in timestamp_value or '+00:00' in timestamp_value:
            # Already ISO format, just normalize
            timestamp_value = timestamp_value.replace('Z', '').replace('+00:00', '')
            dt = datetime.fromisoformat(timestamp_value).replace(tzinfo=timezone.utc)
        else:
            # Try to parse as ISO
            try:
                dt = datetime.fromisoformat(timestamp_value).replace(tzinfo=timezone.utc)
            except ValueError:
                # Default to current time if parsing fails
                dt = datetime.now(timezone.utc)
        return dt.isoformat() + 'Z'
    elif isinstance(timestamp_value, datetime):
        # Already a datetime, ensure timezone and convert to ISO
        if timestamp_value.tzinfo is None:
            timestamp_value = timestamp_value.replace(tzinfo=timezone.utc)
        return timestamp_value.isoformat() + 'Z'
    else:
        # Invalid format, return current time
        logger.warning(f"Unsupported timestamp format: {type(timestamp_value)}, using current time")
        return datetime.now(timezone.utc).isoformat() + 'Z'