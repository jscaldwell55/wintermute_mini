# api/utils/utils.py
import logging
logging.basicConfig(level=logging.INFO)
from datetime import datetime, timezone

def normalize_timestamp(timestamp_value):
    """Handle different timestamp formats and convert to datetime object."""
    if isinstance(timestamp_value, (int, float)):
        # If timestamp is numeric (Unix timestamp), convert to datetime
        return datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
    elif isinstance(timestamp_value, str):
        # If timestamp is string, normalize and convert
        timestamp_value = timestamp_value.replace('Z', '').replace('+00:00', '')
        return datetime.fromisoformat(timestamp_value + '+00:00')
    elif isinstance(timestamp_value, datetime):
        # Already a datetime, just return it
        return timestamp_value
    else:
        # Invalid format
        raise ValueError(f"Unsupported timestamp format: {type(timestamp_value)}")