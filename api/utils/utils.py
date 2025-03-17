import logging
from datetime import datetime, timezone
from dateutil import parser  # Import dateutil parser for more flexible parsing

logger = logging.getLogger(__name__)

def normalize_timestamp(timestamp_value):
    """
    Normalize timestamps to a consistent format, handling various input types.
    Returns a datetime object with timezone info.
    """
    if timestamp_value is None:
        return datetime.now(timezone.utc)

    if isinstance(timestamp_value, (int, float)):
        # If timestamp is numeric (Unix timestamp), convert to datetime
        return datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
    
    elif isinstance(timestamp_value, str):
        # Fix the most common issue: having both +00:00 and Z
        if '+00:00Z' in timestamp_value:
            timestamp_value = timestamp_value.replace('+00:00Z', '+00:00')
        elif timestamp_value.endswith('Z') and '+' in timestamp_value:
            timestamp_value = timestamp_value[:-1]  # Remove the Z
        
        try:
            # Use dateutil.parser for more robust parsing
            dt = parser.parse(timestamp_value)
            
            # Ensure timezone is UTC if not specified
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
                
            return dt
        except Exception as e:
            logger.warning(f"Failed to parse timestamp '{timestamp_value}': {e}. Using current time.")
            return datetime.now(timezone.utc)
    
    elif isinstance(timestamp_value, datetime):
        # Already a datetime, just ensure it has timezone
        if timestamp_value.tzinfo is None:
            return timestamp_value.replace(tzinfo=timezone.utc)
        return timestamp_value
    
    else:
        # Invalid format, log and return current time
        logger.warning(f"Unsupported timestamp format: {type(timestamp_value)}. Using current time.")
        return datetime.now(timezone.utc)