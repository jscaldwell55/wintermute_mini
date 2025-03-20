import logging
from datetime import datetime, timezone
from dateutil import parser  # Make sure python-dateutil is installed

logger = logging.getLogger(__name__)

def normalize_timestamp(timestamp_value):
    """
    Normalize timestamps to a consistent format, handling various input types.
    Returns a datetime object with timezone info.
    
    Args:
        timestamp_value: A timestamp in various formats (string, int, datetime)
        
    Returns:
        datetime: A timezone-aware datetime object
    """
    if timestamp_value is None:
        logger.debug("No timestamp provided, using current time")
        return datetime.now(timezone.utc)

    if isinstance(timestamp_value, (int, float)):
        # Unix timestamp (seconds since epoch)
        try:
            dt = datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
            logger.debug(f"Converted Unix timestamp {timestamp_value} to {dt.isoformat()}")
            return dt
        except (ValueError, OverflowError) as e:
            # Handle Unix millisecond timestamps
            if timestamp_value > 1e10:  # Likely milliseconds
                try:
                    dt = datetime.fromtimestamp(timestamp_value / 1000, tz=timezone.utc)
                    logger.debug(f"Converted Unix millisecond timestamp {timestamp_value} to {dt.isoformat()}")
                    return dt
                except Exception as e2:
                    logger.warning(f"Failed to parse Unix timestamp {timestamp_value}: {e2}")
            else:
                logger.warning(f"Failed to parse Unix timestamp {timestamp_value}: {e}")
            
            return datetime.now(timezone.utc)

    elif isinstance(timestamp_value, str):
        # String timestamp - handle common issues
        cleaned_value = timestamp_value.strip()
        
        # Fix common issues
        if '+00:00Z' in cleaned_value:
            cleaned_value = cleaned_value.replace('+00:00Z', '+00:00')
        elif cleaned_value.endswith('Z') and ('+' in cleaned_value or '-' in cleaned_value[1:]):
            # If has both Z and timezone offset, remove Z
            cleaned_value = cleaned_value[:-1]
        elif cleaned_value.endswith('Z'):
            # Convert Z notation to explicit UTC
            cleaned_value = cleaned_value[:-1] + '+00:00'
            
        try:
            # Use dateutil parser for flexible string parsing
            dt = parser.parse(cleaned_value)
            
            # Ensure timezone awareness
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
                logger.debug(f"Added UTC timezone to naive datetime: {dt.isoformat()}")
            
            logger.debug(f"Parsed string timestamp '{timestamp_value}' to {dt.isoformat()}")
            return dt
            
        except Exception as e:
            logger.warning(f"Failed to parse string timestamp '{timestamp_value}': {e}")
            return datetime.now(timezone.utc)

    elif isinstance(timestamp_value, datetime):
        # Already a datetime, just ensure it has timezone
        if timestamp_value.tzinfo is None:
            dt = timestamp_value.replace(tzinfo=timezone.utc)
            logger.debug(f"Added UTC timezone to naive datetime")
            return dt
        
        return timestamp_value

    else:
        # Invalid format
        logger.warning(f"Unsupported timestamp format: {type(timestamp_value)}")
        return datetime.now(timezone.utc)