"""Compatibility wrapper for the old data_processing module path.

New data work should go under reserve_agent.data.
"""

from reserve_agent.data.loader import *  # noqa: F401,F403
