"""Compatibility wrapper for the old reserving module path.

New model work should go under reserve_agent.models.
"""

from reserve_agent.models.reserving import *  # noqa: F401,F403
