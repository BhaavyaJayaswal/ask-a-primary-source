import json
from difflib import SequenceMatcher
from pathlib import Path

from mcp.server import MCPServer

TIMELINE_PATH = Path(__file__).parent / "timeline.json"
TIMELINE = json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))

MATCH_THRESHOLD = 0.5

mcp = MCPServer("timeline-server")


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


@mcp.tool()
def lookup_year(event: str) -> str:
    """Look up the year a historical event happened, from a fixed timeline
    of events across ancient, medieval, and early modern history."""
    best_event = max(TIMELINE, key=lambda known: _similarity(event, known))
    if _similarity(event, best_event) >= MATCH_THRESHOLD:
        return f"{best_event}: {TIMELINE[best_event]} CE"
    available = ", ".join(TIMELINE.keys())
    return f"No close match for '{event}' in the timeline. Known events: {available}"


if __name__ == "__main__":
    mcp.run()
