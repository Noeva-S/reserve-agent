"""Compatibility wrapper for the old explanation module path.

New Agent explanation work should go under reserve_agent.agent.
"""

from reserve_agent.agent.explanation import *  # noqa: F401,F403
