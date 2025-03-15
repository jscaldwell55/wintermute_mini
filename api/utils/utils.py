# api/utils/utils.py
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def normalize_timestamp(timestamp_value):
    """
    Handle different timestamp formats and convert to ISO format string with Z.
    Raises ValueError or TypeError for unparseable timestamps instead of falling back to current time.
    """
    if timestamp_value is None:
        return datetime.now(timezone.utc).isoformat() + 'Z'
        
    if isinstance(timestamp_value, (int, float)):
        # If timestamp is numeric (Unix timestamp), convert to ISO string
        dt = datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
        return dt.isoformat() + 'Z'
    elif isinstance(timestamp_value, str):
        # If timestamp is string, normalize and convert
        try:
            # Try parsing directly, treating 'Z' as '+00:00'
            dt = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
            return dt.isoformat() + 'Z'
        except ValueError as e_iso_parse:
            # If direct ISO parsing fails, try removing 'Z' and '+00:00' as fallback (less preferred)
            try:
                timestamp_value_no_tz = timestamp_value.replace('Z', '').replace('+00:00', '')
                dt = datetime.fromisoformat(timestamp_value_no_tz).replace(tzinfo=timezone.utc)
                logger.warning(f"Successfully parsed timestamp after removing timezone info (fallback): {timestamp_value}. Original error: {e_iso_parse}")
                return dt.isoformat() + 'Z'
            except ValueError as e_fallback_parse:
                # If even fallback parsing fails, raise the ValueError to be handled upstream
                logger.error(f"Failed to parse timestamp string even with fallback: {timestamp_value}. Original ISO parse error: {e_iso_parse}, Fallback parse error: {e_fallback_parse}")
                raise ValueError(f"Invalid timestamp format: {timestamp_value}") from e_iso_parse # Re-raise ValueError

    elif isinstance(timestamp_value, datetime):
        # Already a datetime, ensure timezone and convert to ISO
        if timestamp_value.tzinfo is None:
            timestamp_value = timestamp_value.replace(tzinfo=timezone.utc)
        return timestamp_value.isoformat() + 'Z'
    else:
        # Invalid format, raise TypeError to be handled upstream
        logger.error(f"Unsupported timestamp format: {type(timestamp_value)}")
        raise TypeError(f"Unsupported timestamp format: {type(timestamp_value)}")