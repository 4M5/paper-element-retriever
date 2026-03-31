"""utils/session_manager.py"""
import uuid, time, threading, os
from typing import Dict, Optional
from ingestion.vector_store import delete_session
from dotenv import load_dotenv
load_dotenv()

TIMEOUT  = int(os.getenv("SESSION_TIMEOUT_MINUTES", 60)) * 60
_sessions: Dict[str, Dict] = {}
_lock = threading.Lock()

def create_session() -> str:
    sid = str(uuid.uuid4())[:8]
    with _lock:
        _sessions[sid] = {"created_at": time.time(), "last_active": time.time(), "papers": 0}
    return sid

def touch(sid: str):
    with _lock:
        if sid in _sessions: _sessions[sid]["last_active"] = time.time()

def add_paper(sid: str):
    with _lock:
        if sid in _sessions: _sessions[sid]["papers"] += 1

def get_info(sid: str) -> Optional[Dict]:
    with _lock: return _sessions.get(sid)

def end_session(sid: str):
    delete_session(sid)
    with _lock: _sessions.pop(sid, None)

def _cleanup():
    while True:
        time.sleep(300)
        now = time.time()
        with _lock:
            expired = [s for s, i in _sessions.items() if now - i["last_active"] > TIMEOUT]
        for s in expired: end_session(s)

threading.Thread(target=_cleanup, daemon=True).start()
