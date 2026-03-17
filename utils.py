"""
utils.py
--------
Shared utility functions for Retrievr.
"""

import uuid
from datetime import datetime, timezone


def generate_image_id() -> str:
    """
    Generate a unique identifier for an image.

    Returns:
        A UUID4 string (e.g. '3f2504e0-4f89-11d3-9a0c-0305e82c3301').
    """
    return str(uuid.uuid4())


def get_timestamp() -> str:
    """
    Return the current UTC time as an ISO-8601 string.

    Returns:
        Timestamp string (e.g. '2025-01-15T10:30:00+00:00').
    """
    return datetime.now(tz=timezone.utc).isoformat()
