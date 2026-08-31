from fastapi import FastAPI
import requests
import os

app = FastAPI()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}
BASE = "https://api-football-v1.p.rapidapi.com/v3"

@app.get("/")
async def root():
    return {"status": "API-Football test server"}

@app.get("/test")
async def test():
    """Test if API-Football via RapidAPI works with your key."""
    if not RAPIDAPI_KEY:
        return {"error": "RAPIDAPI_KEY not set in environment variables"}

    results = {}

    # Test 1: Get leagues
    try:
        r = requests.get(f"{BASE}/leagues", headers=HEADERS, timeout=15)
        data = r.json()
        leagues = data.get("response", [])
        results["leagues"] = {
            "status": r.status_code,
            "count": len(leagues),
            "sample": [l.get("league", {}).get("name") for l in leagues[:3]]
        }
    except Exception as e:
        results["leagues"] = {"error": str(e)}

    # Test 2: Get today's fixtures (EPL = id 39)
    from datetime import datetime
    from zoneinfo import ZoneInfo
    SAST = ZoneInfo("Africa/Johannesburg")
    today = datetime.now(SAST).strftime("%Y-%m-%d")

    try:
        r = requests.get(f"{BASE}/fixtures", headers=HEADERS,
                        params={"league": 39, "season": 2025, "date": today}, timeout=15)
        data = r.json()
        fixtures = data.get("response", [])
        results["fixtures"] = {
            "status": r.status_code,
            "date": today,
            "count": len(fixtures),
            "sample": [
                {
                    "home": f.get("teams", {}).get("home", {}).get("name"),
                    "away": f.get("teams", {}).get("away", {}).get("name"),
                    "id": f.get("fixture", {}).get("id")
                }
                for f in fixtures[:2]
            ]
        }
    except Exception as e:
        results["fixtures"] = {"error": str(e)}

    # Test 3: Get odds for first fixture
    if fixtures:
        fid = fixtures[0].get("fixture", {}).get("id")
        try:
            r = requests.get(f"{BASE}/odds", headers=HEADERS,
                            params={"fixture": fid}, timeout=15)
            data = r.json()
            odds = data.get("response", [])
            results["odds"] = {
                "status": r.status_code,
                "fixture_id": fid,
                "bookmakers_count": len(odds),
                "sample_bookmaker": odds[0].get("bookmaker", {}).get("name") if odds else None,
                "sample_bets": [b.get("name") for b in odds[0].get("bets", [])[:3]] if odds else []
            }
        except Exception as e:
            results["odds"] = {"error": str(e)}
    else:
        results["odds"] = {"message": "No fixtures to fetch odds for"}

    # Test 4: Get team stats
    if fixtures:
        home_id = fixtures[0].get("teams", {}).get("home", {}).get("id")
        try:
            r = requests.get(f"{BASE}/teams/statistics", headers=HEADERS,
                            params={"league": 39, "season": 2025, "team": home_id}, timeout=15)
            data = r.json()
            stats = data.get("response", {})
            results["stats"] = {
                "status": r.status_code,
                "team_id": home_id,
                "has_data": bool(stats),
                "goals_for": stats.get("goals", {}).get("for", {}).get("total", {}).get("total"),
                "goals_against": stats.get("goals", {}).get("against", {}).get("total", {}).get("total"),
                "form": stats.get("form"),
                "played": stats.get("fixtures", {}).get("played", {}).get("total")
            }
        except Exception as e:
            results["stats"] = {"error": str(e)}
    else:
        results["stats"] = {"message": "No fixtures to fetch stats for"}

    return results
