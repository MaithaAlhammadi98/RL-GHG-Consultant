from __future__ import annotations
from typing import Dict, Any
import json, re
from datetime import datetime

TOPIC_BUCKETS = {
    "legal":     [r"\blaw|legal|sue|regulat|policy|compliance|malpractice\b"],
    "fin":       [r"\bbudget|cost|price|fund|profit|finance|investment\b"],
    "ghg":       [r"\bghg|emission|carbon|co2|footprint|offset|net\s*zero|scope\s*[123]\b"],
    "other":     [r".*"],
}

def _bucket(text: str, buckets: Dict[str, list[str]]) -> str:
    t = (text or "").lower()
    for name, patterns in buckets.items():
        for p in patterns:
            if re.search(p, t):
                return name
    return "other"

def encode_state(prompt: str, company_info: Dict[str, Any] | None) -> Dict[str, Any]:
    topic  = _bucket(prompt, TOPIC_BUCKETS)
    length = "short" if len(prompt) < 80 else "medium" if len(prompt) < 200 else "long"
    sector = (company_info or {}).get("sector", "unknown").lower()
    size   = (company_info or {}).get("size", "unknown").lower()
    month  = datetime.utcnow().strftime("%Y-%m")
    return {"topic": topic, "len": length, "sector": sector, "size": size, "month": month}

def state_key(state: Dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True)
