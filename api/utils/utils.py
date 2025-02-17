# api/utils/utils.py
import logging
logging.basicConfig(level=logging.INFO)
from datetime import datetime

def normalize_timestamp(timestamp: str) -> str:
    """
    Normalizes a timestamp string to a consistent format suitable for
    datetime.fromisoformat().  Handles variations with 'Z', '+00:00',
    and the problematic '+00:00+00:00'.

    Args:
        timestamp: The timestamp string.

    Returns:
        A normalized timestamp string with '+00:00' offset.
    """
    # Remove any existing timezone info ('Z' or '+00:00')
    timestamp = timestamp.replace('Z', '').replace('+00:00', '')
    # Add standard UTC offset
    return timestamp + '+00:00'