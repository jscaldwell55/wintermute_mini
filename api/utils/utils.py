# api/utils/utils.py
import logging
from datetime import datetime, timezone
from dateutil import parser # Import dateutil parser for more flexible parsing

logger = logging.getLogger(__name__)

def normalize_timestamp(timestamp_value):
    """
    Handle different timestamp formats and convert to ISO format string with Z,
    using dateutil.parser for more robust parsing.
    """
    if timestamp_value is None:
        return datetime.now(timezone.utc).isoformat() + 'Z'

    if isinstance(timestamp_value, (int, float)):
        # If timestamp is numeric (Unix timestamp), convert to ISO string
        dt = datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
        return dt.isoformat() + 'Z'
    elif isinstance(timestamp_value, str):
        original_timestamp_str = timestamp_value # Keep original string for logging

        # Fix double timezone indicator: either remove Z or +00:00 (before dateutil parser)
        if timestamp_value.endswith('Z') and '+00:00' in timestamp_value:
            timestamp_value = timestamp_value.replace('+00:00Z', 'Z')

        try:
            # Use dateutil.parser for more flexible parsing
            dt = parser.isoparse(timestamp_value)

            # Ensure timezone is UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            return dt.isoformat() + 'Z'
        except ValueError as e_iso:
            try:
                # Fallback to dateutil's more generic parser if ISO parsing fails
                dt = parser.parse(original_timestamp_str) # Parse original string
                # Ensure timezone is UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat() + 'Z'

            except ValueError as e_generic:
                # Last resort fallback - parsing failed with all methods
                logger.warning(f"Failed to parse timestamp '{original_timestamp_str}' with ISO ({e_iso}) and generic parser ({e_generic}). Using current time.")
                return datetime.now(timezone.utc).isoformat() + 'Z'

    elif isinstance(timestamp_value, datetime):
        # Already a datetime, ensure timezone and convert to ISO
        if timestamp_value.tzinfo is None:
            timestamp_value = timestamp_value.replace(tzinfo=timezone.utc)
        return timestamp_value.isoformat() + 'Z'
    else:
        # Invalid format, log and return current time
        logger.error(f"Unsupported timestamp format: {type(timestamp_value)}. Using current time.")
        return datetime.now(timezone.utc).isoformat() + 'Z'