import json
from typing import Any, Dict, List

def load_json_array(path: str) -> List[Dict[str, Any]]:
    """
    Loads a JSON file that contains an array of objects:
    [
      {...},
      {...}
    ]
    Each object is treated as one chunk.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data)}")
    return data

def event_to_text(e: Dict[str, Any]) -> str:
    # Make a strong searchable text for events
    # Keep it stable and consistent
    fields = [
        ("Id", e.get("Id")),
        ("event_type", e.get("event_type")),
        ("event_name", e.get("event_name")),
        ("event_desc", e.get("event_desc")),
        ("event_dt", e.get("event_dt")),
        ("event_ts", e.get("event_ts")),
        ("event_intent", e.get("event_intent")),
        ("event_sub_intent", e.get("event_sub_intent")),
    ]
    parts = [f"{k}: {v}" for k, v in fields if v is not None]
    return " | ".join(parts)

def transcript_to_text(t: Dict[str, Any]) -> str:
    fields = [
        ("id", t.get("id")),
        ("transcript_date", t.get("transcript_date")),
        ("transcript", t.get("transcript")),
        ("transcript_timestamp", t.get("transcript_timestamp")),
        ("intent", t.get("intent")),
        ("agent_id", t.get("agent_id")),
    ]
    parts = [f"{k}: {v}" for k, v in fields if v is not None]
    return " | ".join(parts)
