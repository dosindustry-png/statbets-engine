import os
import json
import uuid
import sqlite3
import statistics
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, HTTPException, Query

app = FastAPI()
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# ======================
# ENV / CONFIG
# ======================

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")  # set this in Render env vars

REGIONS = "au"                  # AU ONLY
ODDS_FORMAT = "decimal"
MAX_PICKS_PER_SPORT = 5
TOP_TIP_THRESHOLD = 0.85        # 85%

# Odds API sport keys
SPORT_KEYS = {
    "NBA": ["basketball_nba"],
    "NHL": ["icehockey_nhl"],
    "NFL": ["americanfootball_nfl"],
    "NRL": ["rugbyleague_nrl"],
    "MLB": ["baseball_mlb"],
    "EPL": ["soccer_epl"],
    # Tennis keys are tournament-specific. Replace these when tournament changes.
    "TENNIS": [
        "tennis_atp_aus_open_singles",
        "tennis_wta_aus_open_singles",
    ],
    "UFC": ["mma_mixed_martial_arts_ufc"],
}

DB_PATH = os.getenv("DB_PATH", "statbets.sqlite3")


# ======================
# DB HELPERS
# ======================

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_runs (
            run_date TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS picks (
            id TEXT PRIMARY KEY,
            pick_date TEXT NOT NULL,
            sport TEXT NOT NULL,
            rank INTEGER NOT NULL,
            top_tip INTEGER NOT NULL,
            model_probability REAL NOT NULL,
            implied_probability REAL NOT NULL,
            avg_odds REAL NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_picks_date_sport ON picks(pick_date, sport)")
    conn.commit()
    conn.close()

@app.on_event("startup")
def _startup():
    init_db()


# ======================
# CORE HELPERS
# ======================

def utc_today() -> date:
    return datetime.now(timezone.utc).date()

def avg_decimal_odds(prices: List[float]) -> Optional[float]:
    valid = [p for p in prices if isinstance(p, (int, float)) and p > 1.0]
    if not valid:
        return None
    return round(statistics.mean(valid), 4)

def implied_prob(decimal_odds: float) -> float:
    return round(1.0 / decimal_odds, 4)

def require_admin(key: str):
    if not ADMIN_KEY:
        raise HTTPException(status_code=500, detail="Missing ADMIN_KEY on server")
    if key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")

def already_ran_today() -> bool:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM daily_runs WHERE run_date = ?", (utc_today().isoformat(),))
    row = cur.fetchone()
    conn.close()
    return row is not None


# ======================
# ODDS FETCH + PICK BUILD
# ======================

def fetch_events_for_sport(sport_key: str, markets: List[str]) -> List[Dict[str, Any]]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": ",".join(markets),
        "oddsFormat": ODDS_FORMAT,
    }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        # Don’t crash the whole run; just return empty for this sport key.
        return []
    data = r.json()
    return data if isinstance(data, list) else []

def build_picks_for_sport(sport: str) -> List[Dict[str, Any]]:
    # For now we generate ML picks (h2h) only, to keep it consistent.
    markets = ["h2h"]
    all_events: List[Dict[str, Any]] = []

    for sk in SPORT_KEYS[sport]:
        all_events.extend(fetch_events_for_sport(sk, markets))

    picks: List[Dict[str, Any]] = []

    for ev in all_events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        bookmakers = ev.get("bookmakers", [])

        if not home or not away:
            continue

        # Collect home-team h2h prices from AU books
        prices: List[float] = []
        for bm in bookmakers:
            for m in bm.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes", []):
                    if o.get("name") == home:
                        price = o.get("price")
                        if isinstance(price, (int, float)):
                            prices.append(float(price))

        avg_odds = avg_decimal_odds(prices)
        if not avg_odds:
            continue

        imp = implied_prob(avg_odds)

        # Placeholder “model probability” until NBA context engine is added.
        # For now, it equals implied probability (market baseline).
        model_p = imp

        confidence_level = (
            "HIGH" if model_p >= 0.75 else
            "MEDIUM" if model_p >= 0.62 else
            "LOW"
        )

        analysis = [
            "AU market consensus (averaged AU bookmakers)",
            f"Books used: {len(prices)}",
            "NBA deep context engine (injuries/rest/travel) will be added next"
        ]

        picks.append({
            "id": str(uuid.uuid4()),
            "sport": sport,
            "league": sport,  # keep simple; your UI expects a league-ish label
            "home_team": home,
            "away_team": away,
            "game_time": commence,
            "game_date": (commence or "")[:10],
            "recommended_bet": f"{home} ML",
            "avg_odds": avg_odds,
            "implied_probability": imp,
            "model_probability": model_p,
            "confidence_score": round(model_p * 100, 2),
            "confidence_level": confidence_level,
            "top_tip": bool(model_p >= TOP_TIP_THRESHOLD),
            "rank": None,  # filled after sorting
            # fields your UI references (safe defaults)
            "edge": 0.0,
            "simulation_results": None,
            "game_status": None,
            "home_live_score": None,
            "away_live_score": None,
            "live_period": None,
            "live_clock": None,
            "followed_by_users": 0,
            "analysis": analysis,
        })

    # Rank by model probability, take top 5
    picks.sort(key=lambda x: x["model_probability"], reverse=True)
    picks = picks[:MAX_PICKS_PER_SPORT]
    for i, p in enumerate(picks, start=1):
        p["rank"] = i
    return picks


# ======================
# STORAGE
# ======================

def store_daily_run(picks_by_sport: Dict[str, List[Dict[str, Any]]]) -> None:
    run_date = utc_today().isoformat()
    now = datetime.now(timezone.utc).isoformat()

    conn = db()
    cur = conn.cursor()

    # mark run
    cur.execute(
        "INSERT OR REPLACE INTO daily_runs(run_date, created_at) VALUES (?, ?)",
        (run_date, now)
    )

    # delete existing picks for date (idempotent)
    cur.execute("DELETE FROM picks WHERE pick_date = ?", (run_date,))

    for sport, picks in picks_by_sport.items():
        for p in picks:
            cur.execute("""
                INSERT INTO picks (
                    id, pick_date, sport, rank, top_tip,
                    model_probability, implied_probability, avg_odds,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p["id"],
                run_date,
                sport,
                int(p["rank"]),
                1 if p["top_tip"] else 0,
                float(p["model_probability"]),
                float(p["implied_probability"]),
                float(p["avg_odds"]),
                json.dumps(p),
                now
            ))

    conn.commit()
    conn.close()


# ======================
# ROUTES
# ======================

@app.get("/")
def root():
    return {"status": "ok", "message": "StatBets engine running", "regions": REGIONS}

@app.get("/sports")
def sports():
    # Your frontend expects {id, name}
    out = []
    for name in SPORT_KEYS.keys():
        out.append({"id": name, "name": name})
    return out

@app.post("/admin/generate")
def admin_generate(key: str = Query(...)):
    if not ODDS_API_KEY:
        raise HTTPException(status_code=500, detail="Missing ODDS_API_KEY")
    require_admin(key)

    # enforce once/day
    if already_ran_today():
        return {"status": "ok", "message": "Already generated today", "date": utc_today().isoformat()}

    picks_by_sport: Dict[str, List[Dict[str, Any]]] = {}
    for sport in SPORT_KEYS.keys():
        picks_by_sport[sport] = build_picks_for_sport(sport)

    store_daily_run(picks_by_sport)

    return {"status": "ok", "message": "Generated daily picks", "date": utc_today().isoformat()}

@app.get("/picks/today")
def picks_today():
    run_date = utc_today().isoformat()

    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT payload_json
        FROM picks
        WHERE pick_date = ?
        ORDER BY sport ASC, rank ASC
    """, (run_date,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return {"date": run_date, "generated": False, "sports": {}}

    sports_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        p = json.loads(r["payload_json"])
        sports_map.setdefault(p["sport"], []).append(p)

    return {"date": run_date, "generated": True, "sports": sports_map}

@app.get("/picks/history")
def picks_history(limit_days: int = 7):
    limit_days = max(1, min(limit_days, 60))

    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT pick_date
        FROM picks
        ORDER BY pick_date DESC
        LIMIT ?
    """, (limit_days,))
    dates = [row["pick_date"] for row in cur.fetchall()]

    history = []
    for d in dates:
        cur.execute("""
            SELECT payload_json
            FROM picks
            WHERE pick_date = ?
            ORDER BY sport ASC, rank ASC
        """, (d,))
        rows = cur.fetchall()
        sports_map: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            p = json.loads(r["payload_json"])
            sports_map.setdefault(p["sport"], []).append(p)
        history.append({"date": d, "sports": sports_map})

    conn.close()
    return {"history": history}

@app.get("/picks/{pick_id}")
def pick_detail(pick_id: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT payload_json FROM picks WHERE id = ?", (pick_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Pick not found")

    return json.loads(row["payload_json"])
from fastapi.middleware.cors import CORSMiddleware

# Allow your website to call the API (change domains later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # quick start; tighten later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Returns the sports list
@app.get("/sports")
def sports():
    return [{"id": k, "name": k} for k in SPORT_KEYS.keys()]

# Returns today's generated picks (this assumes you are storing picks somewhere)
# If you are NOT storing picks yet, tell me and I’ll provide the exact storage block.
@app.get("/picks/today")
def picks_today():
    # If you already have SQLite logic in your code, this should query that.
    # If not, we need to add it.
    raise HTTPException(status_code=501, detail="picks storage not implemented yet")
