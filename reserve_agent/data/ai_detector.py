"""Compatibility alias for the AI-assisted Excel detector.

The project originally used ``api_detector.py``.  The final division of work
calls this module ``ai_detector.py``, so this file re-exports the implementation
without breaking existing imports.
"""

from reserve_agent.data.api_detector import *  # noqa: F401,F403
