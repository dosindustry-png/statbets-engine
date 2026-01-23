import os
import statistics
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, HTTPException

app = FastAPI()

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# Each sport maps to a LIST of Odds API sport keys (Tennis uses two feeds)
SPORT_KEYS = {
    "NBA": ["basketball_nba"],
    "NHL": ["icehockey_nhl"],
    "NFL": ["americanfootball_nfl"],
    "NRL": ["rugbyleague_nrl"],
    "MLB": ["baseball_mlb"],
    "EPL": ["soccer_epl"],
    "TENNIS": ["tennis_atp", "tennis_wta"],
    "UFC": ["mma_mixed_martial_arts"],
}

REGIONS = "au,us,eu"
ODDS_FORMAT = "decimal"


def avg_decimal_odds(odds_list: List[float]) -> Optional[float]:
    odds_list = [o for o in odds_list if isinstance(o, (int, float)) and o > 1.0]
    if not odds_list:
        return None
    return round(statistics.mean(odds_list), 4)


@app.get("/")
def root():
    return {"status": "ok", "message": "StatBets engine is running"}


@app.post("/daily-picks")
def daily_picks(payload: Optional[Dict[str, Any]] = None):
    """
    Fetch odds from The Odds API and return global-averaged odds across AU/US/EU.
    No pick logic yet.
    """
    if not ODDS_API_KEY:
        raise HTTPException(status_code=500, detail="Missing ODDS_API_KEY env var in Render")

    payload = payload or {}
    sport = payload.get("sport", "NBA")
    markets = payload.get("markets", ["h2h", "spreads", "totals"])

    sport_keys = SPORT_KEYS.get(sport)
    if not sport_keys:
        raise HTTPException(status_code=400, detail=f"Unsupported sport '{sport}'")

    # Build params ONCE (you were missing this)
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": ",".join(markets),
        "oddsFormat": ODDS_FORMAT,
    }

    # Fetch events for one or many sport keys (TENNIS uses two)
    events: List[Dict[str, Any]] = []
    for sport_key in sport_keys:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Odds API error {r.status_code}: {r.text}")
        data = r.json()
        if isinstance(data, list):
            events.extend(data)

    output = []
    for ev in events[:15]:  # limit for now
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        bookmakers = ev.get("bookmakers", [])

        # Collect all prices across all bookmakers for each outcome name
        market_prices: Dict[str, Dict[str, List[float]]] = {}

        for bm in bookmakers:
            for m in bm.get("markets", []):
                mkey = m.get("key")
                if mkey not in markets:
                    continue
                market_prices.setdefault(mkey, {})
                for out in m.get("outcomes", []):
                    name = out.get("name")
                    price = out.get("price")
                    if name and isinstance(price, (int, float)):
                        market_prices[mkey].setdefault(name, []).append(float(price))

        averaged = []
        for mkey, outcomes in market_prices.items():
            for name, prices in outcomes.items():
                avg = avg_decimal_odds(prices)
                if avg is None:
                    continue
                averaged.append(
                    {
                        "market": mkey,
                        "selection": name,
                        "avg_odds": avg,
                        "books_used": len(prices),
                    }
                )

        output.append(
            {
                "match": f"{away} vs {home}",
                "commence_time": commence,
                "averages": averaged,
            }
        )

    return {
        "sport": sport,
        "regions": ["au", "us", "eu"],
        "note": "Odds averaging only (no picks yet).",
        "events_returned": len(output),
        "events": output,
    }
