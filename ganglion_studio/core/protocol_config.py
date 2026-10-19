"""Remember the last-used parameters for each protocol.

So the researcher does not have to re-type "5 blinks, 3 sets, 10 s between sets"
every session, the Protocol Helper saves the parameters it was last run with. Like
:mod:`ganglion_studio.core.saved_devices`, this is a tiny JSON file in the user's
home config dir (not the project), keyed by protocol id
(``{"blinks": {...}, "eyes": {...}, ...}``).
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

# Module-level so tests can point it at a temp file via monkeypatch.
PATH = os.path.expanduser(os.path.join("~", ".ganglion_studio", "protocols.json"))


def load() -> dict:
    """Return the saved {protocol_key: params} map (empty on missing/unreadable)."""
    try:
        with open(PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: params for key, params in data.items() if isinstance(params, dict)}


def save(key: str, params: dict) -> dict:
    """Remember ``params`` when writable; return the updated map in either case.

    Preferences are best effort: an unavailable config directory must not stop
    the experiment from starting.
    """
    data = load()
    data[key] = dict(params)
    try:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        with open(PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except OSError as exc:
        logger.warning("Could not save protocol preferences: %s", exc)
    return data
