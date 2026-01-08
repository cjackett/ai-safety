"""
Vulture whitelist for ai-safety.

This file contains patterns that should not be flagged as dead code by vulture.
Vulture whitelist files must be valid Python code that references the items
you want to whitelist.

See: https://github.com/jendrikseipp/vulture#whitelisting-false-positives
"""

import logging

# Common patterns that may trigger false positives
logger = logging.getLogger(__name__)
logger.debug  # noqa: B018
logger.info  # noqa: B018
logger.warning  # noqa: B018
logger.error  # noqa: B018

# Common attribute patterns
__version__ = None
__all__: list[str] = []
