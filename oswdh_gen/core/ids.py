import hashlib

def make_id(prefix: str, *parts: str) -> str:
    raw = "|".join([prefix, *map(str, parts)]).encode("utf-8")
    h = hashlib.sha256(raw).hexdigest()[:16]
    return f"{prefix}_{h}"
