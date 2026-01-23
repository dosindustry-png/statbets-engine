import os
import statistics
import requests
from fastapi import FastAPI, HTTPException

app = FastAPI()

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

SPORT_KEYS = {
    "NBA": "basketball_nba",
}

REGIONS = "au,us,eu"
ODDS_FORMAT = "decimal"


def avg_decimal_odds(odds_list):
    odds_list = [o for o in odds_list if isinstance(o, (int, float)) and o > 1.0]
    if not odds_list:
        return None
    return round(statistics.mean(odds_list), 4)


@app.get("/")
def root():
    return {"status": "ok", "message": "StatBets engine is running"}


@app.post("/daily-picks")
def daily_picks(payload: dict | None = None):
    if not ODDS_API_KEY:
        raise HTTPException(status_code=500, detail="Missing ODDS_API_KEY env var in Render")

    payload = payload or {}
    sport = payload.get("sport", "NBA")
    markets = payload.get("markets", ["h2h"])  # start with moneyline only

    if sport not in SPORT_KEYS:
        raise HTTPException(status_code=400, detail=f"Unsupported sport '{sport}' (try NBA)")

    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEYS[sport]}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": ",".join(markets),
        "oddsFormat": ODDS_FORMAT,
    }

    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Odds API error {r.status_code}: {r.text}")

    events = r.json()

    output = []
    for ev in events[:15]:  # limit for now
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        bookmakers = ev.get("bookmakers", [])

        # Collect all prices across all bookmakers for each outcome name
        market_prices = {}
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
                        market_prices[mkey].setdefault(name, []).append(price)

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
