# server.py
from typing import Any, Dict, List, Tuple
from contextlib import asynccontextmanager, suppress
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from threading import RLock
from pathlib import Path
import asyncio, json, os, sqlite3, sys

# ---------------------- Config ----------------------
DB_PATH = os.getenv("STORE_DB", "kv_store.db")
FLUSH_INTERVAL = int(os.getenv("FLUSH_INTERVAL", "10"))  # flush to SQLite every N seconds
BOOTSTRAP_JSON = os.getenv("INITIAL_JSON")               # optional: preload from JSON if DB is empty

# ---------------------- In-memory state ----------------------
STORE: Dict[str, List[Any]] = {}          # { key: [item, ...] }
PERSISTED_LEN: Dict[str, int] = {}        # per-key number of items already persisted to SQLite
LOCK = RLock()                            # protects STORE and PERSISTED_LEN

# ---------------------- DB state ----------------------
CONN: sqlite3.Connection | None = None
DB_LOCK = RLock()                         # protects the SQLite connection

def _db_connect(path: str) -> sqlite3.Connection:
    """Open SQLite connection and ensure schema exists."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kv_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            k  TEXT NOT NULL,
            item_json TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_kv_items_k ON kv_items(k)")
    return conn

def _normalize_to_items(v: Any) -> List[Any]:
    """
    Normalize input value into a list of items:
    - If v is a list whose first element is a list/tuple -> treat as multiple items.
    - Otherwise -> treat v as a single item.
    """
    if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
        return list(v)
    return [v]

def _load_store_from_db(conn: sqlite3.Connection) -> Tuple[Dict[str, List[Any]], Dict[str, int]]:
    """Replay all rows from SQLite into memory; return (STORE, PERSISTED_LEN)."""
    cur = conn.cursor()
    cur.execute("SELECT k, item_json FROM kv_items ORDER BY id ASC")
    store: Dict[str, List[Any]] = {}
    for k, item_json in cur.fetchall():
        store.setdefault(k, []).append(json.loads(item_json))
    persisted = {k: len(v) for k, v in store.items()}
    return store, persisted

def _flush_once() -> Dict[str, int]:
    """
    Persist in-memory deltas to SQLite in one transaction.
    Returns: dict of {key: inserted_count} for this flush.
    """
    global CONN
    assert CONN is not None, "DB not initialized"

    # 1) Take an in-memory snapshot under lock
    with LOCK:
        snapshot_store = {k: list(v) for k, v in STORE.items()}
        start_counts = dict(PERSISTED_LEN)

    # 2) Compute deltas
    to_insert: List[Tuple[str, str]] = []
    inserted_per_key: Dict[str, int] = {}
    for k, items in snapshot_store.items():
        start = start_counts.get(k, 0)
        if start < len(items):
            new_items = items[start:]
            to_insert.extend((k, json.dumps(it)) for it in new_items)
            inserted_per_key[k] = len(new_items)

    if not to_insert:
        return {}

    # 3) Write all deltas in a single transaction
    with DB_LOCK:
        with CONN:
            CONN.executemany("INSERT INTO kv_items (k, item_json) VALUES (?, ?)", to_insert)

    # 4) Update persisted counters based on the snapshot start
    with LOCK:
        for k, n in inserted_per_key.items():
            PERSISTED_LEN[k] = start_counts.get(k, 0) + n

    return inserted_per_key

async def _flush_loop():
    """Background task: periodically flush in-memory changes to SQLite."""
    while True:
        await asyncio.sleep(FLUSH_INTERVAL)
        try:
            inserted = _flush_once()
            if inserted:
                total = sum(inserted.values())
                print(f"[flush] persisted {total} items across {len(inserted)} keys")
        except Exception as e:
            # Never crash the task; log and continue.
            print(f"[flush] error: {e}", file=sys.stderr)

def _bootstrap_from_json_if_needed():
    """
    If DB is empty and BOOTSTRAP_JSON is provided, preload JSON into memory.
    Persisting to DB will be handled by the periodic flush.
    """
    global STORE, PERSISTED_LEN, CONN
    if not BOOTSTRAP_JSON or not os.path.exists(BOOTSTRAP_JSON):
        return
    cur = CONN.cursor()
    cur.execute("SELECT 1 FROM kv_items LIMIT 1")
    if cur.fetchone():
        return  # DB is not empty; DB is the source of truth.

    try:
        with open(BOOTSTRAP_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"[bootstrap] {BOOTSTRAP_JSON} is not a JSON object, skip.")
            return
        with LOCK:
            for k, v in data.items():
                sk = str(k)
                STORE.setdefault(sk, []).extend(_normalize_to_items(v))
                PERSISTED_LEN.setdefault(sk, 0)
        print(f"[bootstrap] preloaded {len(STORE)} keys from {BOOTSTRAP_JSON}")
    except Exception as e:
        print(f"[bootstrap] error: {e}", file=sys.stderr)

# ---------------------- Lifespan (startup/shutdown) ----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Set up DB and background flush on startup; flush and cleanup on shutdown."""
    global CONN, STORE, PERSISTED_LEN
    db_exists = Path(DB_PATH).exists()
    CONN = _db_connect(DB_PATH)
    print(f"[db] {'found' if db_exists else 'created'} SQLite at {DB_PATH}")

    STORE, PERSISTED_LEN = _load_store_from_db(CONN)
    print(f"[db] loaded {sum(len(v) for v in STORE.values())} items across {len(STORE)} keys")

    _bootstrap_from_json_if_needed()

    # Start the periodic flush task
    app.state.flush_task = asyncio.create_task(_flush_loop())

    # ---- running phase ----
    try:
        yield
    finally:
        # ---- shutdown phase ----
        try:
            inserted = _flush_once()
            if inserted:
                total = sum(inserted.values())
                print(f"[shutdown] persisted {total} items before exit")
        finally:
            flush_task = getattr(app.state, "flush_task", None)
            if flush_task:
                flush_task.cancel()
                with suppress(asyncio.CancelledError):
                    await flush_task
            if CONN is not None:
                CONN.close()

# Initialize FastAPI with lifespan (no deprecated on_event)
app = FastAPI(title="KV Append Store (SQLite-backed)", version="1.2.0", lifespan=lifespan)

# ---------------------- API ----------------------
@app.post("/append")
def append(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    """
    Append values to keys:
    - If value is list of lists/tuples: append multiple items.
    - Otherwise: append a single item.
    """
    updated = []
    with LOCK:
        for k, v in payload.items():
            sk = str(k)
            items = _normalize_to_items(v)
            STORE.setdefault(sk, []).extend(items)
            PERSISTED_LEN.setdefault(sk, 0)
            updated.append(sk)
    return JSONResponse({
        "status": "ok",
        "updated_keys": sorted(set(updated)),
        "total_keys": len(STORE)
    })

@app.get("/dump")
def dump() -> JSONResponse:
    """Return the full in-memory snapshot."""
    with LOCK:
        snapshot = {k: list(v) for k, v in STORE.items()}
    return JSONResponse(snapshot)

@app.post("/save")
def save_now() -> JSONResponse:
    """Force an immediate flush to SQLite."""
    inserted = _flush_once()
    return JSONResponse({"status": "ok", "inserted_per_key": inserted, "keys": len(STORE)})

@app.get("/healthz")
def healthz():
    """Basic health/metrics endpoint."""
    with LOCK:
        total_items = sum(len(v) for v in STORE.values())
    return {"ok": True, "keys": len(STORE), "items": total_items, "db": DB_PATH, "interval": FLUSH_INTERVAL}

@app.post("/clear")
def clear_all() -> JSONResponse:
    """
    DANGER: Clear EVERYTHING without auth.
    - Clears in-memory STORE and persisted counters.
    - Deletes all rows from SQLite and VACUUMs the file.
    """
    global CONN
    # 1) Clear memory first (same lock order as flush: acquire LOCK before DB_LOCK)
    with LOCK:
        cleared_keys = len(STORE)
        cleared_items = sum(len(v) for v in STORE.values())
        STORE.clear()
        PERSISTED_LEN.clear()

    # 2) Clear SQLite (single transaction), then VACUUM
    with DB_LOCK:
        with CONN:  # transaction
            CONN.execute("DELETE FROM kv_items")
            # Optional: reset AUTOINCREMENT counter
            try:
                CONN.execute("DELETE FROM sqlite_sequence WHERE name='kv_items'")
            except sqlite3.OperationalError:
                pass
        # VACUUM must not run inside a transaction
        try:
            CONN.execute("VACUUM")
        except Exception:
            pass

    return JSONResponse({
        "status": "ok",
        "cleared_keys": cleared_keys,
        "cleared_items": cleared_items
    })

# ---------------------- Entrypoint ----------------------
# Enables `python server.py` as well as `uvicorn server:app`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))