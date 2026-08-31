from fastapi import FastAPI
from pydantic import BaseModel
import requests
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from itertools import combinations
from math import prod
import os
import json

app = FastAPI()

# ─── CONFIG ─────────────────────────────────────────
BIG_BALLS_KEY = os.getenv("BIG_BALLS_KEY")
SHARPAPI_KEY = os.getenv("SHARPAPI_KEY")

if not BIG_BALLS_KEY or not SHARPAPI_KEY:
    raise RuntimeError("API keys missing. Set BIG_BALLS_KEY and SHARPAPI_KEY environment variables.")

SAST = ZoneInfo("Africa/Johannesburg")

HEADERS_BB = {"Authorization": f"Bearer {BIG_BALLS_KEY}"}
HEADERS_SHARP = {"X-API-Key": SHARPAPI_KEY}

BASE_BB = "https://api.bigballsdata.com/v1"
BASE_SHARP = "https://api.sharpapi.io/api/v1"

# ─── HELPER: Safe JSON parsing ────────────────────────
def safe_json(resp):
    """Parse response safely. Returns (data, error)."""
    try:
        data = resp.json()
        if isinstance(data, dict):
            return data, None
        elif isinstance(data, list):
            return {"data": data}, None
        else:
            return None, f"Unexpected response type: {type(data).__name__}: {str(data)[:200]}"
    except Exception as e:
        return None, f"JSON parse error: {str(e)}. Raw text: {resp.text[:500]}"

# ─── DEBUG ENDPOINT ─────────────────────────────────
@app.get("/debug")
async def debug():
    """Show raw API responses for debugging."""
    results = {}
    
    # Test Big Balls
    today = datetime.now(SAST).strftime("%Y-%m-%d")
    try:
        url = f"{BASE_BB}/matches"
        params = {"sport": "football", "date": today, "per_page": 5}
        resp = requests.get(url, headers=HEADERS_BB, params=params, timeout=30)
        data, err = safe_json(resp)
        results["bigballs"] = {
            "url": url,
            "status": resp.status_code,
            "error": err,
            "data_preview": str(data)[:1000] if data else None
        }
    except Exception as e:
        results["bigballs"] = {"error": str(e)}
    
    # Test SharpAPI
    try:
        url = f"{BASE_SHARP}/odds"
        params = {"league": "EPL", "market_type": "moneyline", "per_page": 3}
        resp = requests.get(url, headers=HEADERS_SHARP, params=params, timeout=15)
        data, err = safe_json(resp)
        results["sharpapi"] = {
            "url": url,
            "status": resp.status_code,
            "error": err,
            "data_preview": str(data)[:1000] if data else None
        }
    except Exception as e:
        results["sharpapi"] = {"error": str(e)}
    
    return results

# ─── MODELS ─────────────────────────────────────────
class AnalyzeResponse(BaseModel):
    status: str
    timestamp: str
    matches_scanned: int
    markets_scanned: int
    accumulators: list

# ─── MONTE CARLO ENGINE ─────────────────────────────
class MonteCarloEngine:
    def __init__(self, simulations: int = 20000):
        self.simulations = simulations
        self.home_adv = 1.35

    def fetch_fixtures(self):
        today = datetime.now(SAST).strftime("%Y-%m-%d")
        url = f"{BASE_BB}/matches"
        params = {
            "sport": "football",
            "date": today,
            "include": "stats,elo,lineups,injuries",
            "per_page": 100
        }
        resp = requests.get(url, headers=HEADERS_BB, params=params, timeout=30)
        resp.raise_for_status()
        data, err = safe_json(resp)
        if err:
            raise RuntimeError(f"Big Balls API error: {err}")
        return data.get("data", [])

    def fetch_odds(self, leagues: list):
        all_odds = []
        for league in leagues:
            url = f"{BASE_SHARP}/odds"
            params = {
                "league": league,
                "market_type": "moneyline,totals,btts,spreads",
                "per_page": 100
            }
            resp = requests.get(url, headers=HEADERS_SHARP, params=params, timeout=15)
            if resp.status_code == 200:
                data, err = safe_json(resp)
                if err:
                    continue
                all_odds.extend(data.get("data", []))
        return all_odds

    def match_odds_to_fixture(self, fixture: dict, all_odds: list):
        home = ""
        away = ""
        
        # Try multiple possible structures
        if isinstance(fixture, dict):
            home = fixture.get("home", {}).get("name", "") if isinstance(fixture.get("home"), dict) else fixture.get("home_team", {}).get("name", "") if isinstance(fixture.get("home_team"), dict) else str(fixture.get("home", ""))
            away = fixture.get("away", {}).get("name", "") if isinstance(fixture.get("away"), dict) else fixture.get("away_team", {}).get("name", "") if isinstance(fixture.get("away_team"), dict) else str(fixture.get("away", ""))
        
        if not home or not away:
            return []
        
        match_odds = []
        for odd in all_odds:
            if not isinstance(odd, dict):
                continue
            odd_home = odd.get("home_team", "")
            odd_away = odd.get("away_team", "")
            if (home.lower() in str(odd_home).lower() or str(odd_home).lower() in home.lower()) and \
               (away.lower() in str(odd_away).lower() or str(odd_away).lower() in away.lower()):
                match_odds.append(odd)
        return match_odds

    def calc_lambda(self, match: dict):
        if not isinstance(match, dict):
            return 1.4, 1.2
            
        league_avg = 2.65
        
        home_team = match.get("home_team", {}) if isinstance(match.get("home_team"), dict) else {}
        away_team = match.get("away_team", {}) if isinstance(match.get("away_team"), dict) else {}
        
        hs = home_team.get("recent_stats", {}) if isinstance(home_team.get("recent_stats"), dict) else {}
        as_ = away_team.get("recent_stats", {}) if isinstance(away_team.get("recent_stats"), dict) else {}
        
        if not hs or not as_:
            return 1.4, 1.2
        
        home_att = hs.get("goals_scored_pg", 1.4) / (league_avg / 2)
        home_def = hs.get("goals_conceded_pg", 1.1) / (league_avg / 2)
        away_att = as_.get("goals_scored_pg", 1.2) / (league_avg / 2)
        away_def = as_.get("goals_conceded_pg", 1.3) / (league_avg / 2)
        
        elo = match.get("predictions", {}).get("elo", {}) if isinstance(match.get("predictions", {}), dict) else {}
        elo_diff = (elo.get("home_elo", 1500) - elo.get("away_elo", 1500)) / 400
        elo_mult = 10 ** elo_diff
        
        base_h = (league_avg / 2) * home_att * away_def
        base_a = (league_avg / 2) * away_att * home_def
        
        lambda_h = base_h * self.home_adv * elo_mult
        lambda_a = base_a / elo_mult
        
        injuries = match.get("injuries", []) if isinstance(match.get("injuries"), list) else []
        home_inj = len([i for i in injuries if isinstance(i, dict) and i.get("team") == "home" and i.get("impact") == "high"])
        away_inj = len([i for i in injuries if isinstance(i, dict) and i.get("team") == "away" and i.get("impact") == "high"])
        lambda_h *= (1 - 0.08 * home_inj)
        lambda_a *= (1 - 0.08 * away_inj)
        
        return max(lambda_h, 0.3), max(lambda_a, 0.3)

    def simulate(self, lambda_h: float, lambda_a: float):
        np.random.seed(42)
        hg = np.random.poisson(lambda_h, self.simulations)
        ag = np.random.poisson(lambda_a, self.simulations)
        tg = hg + ag
        
        return {
            "home_win": float(np.mean(hg > ag)),
            "draw": float(np.mean(hg == ag)),
            "away_win": float(np.mean(hg < ag)),
            "over_1.5": float(np.mean(tg > 1.5)),
            "over_2.5": float(np.mean(tg > 2.5)),
            "over_3.5": float(np.mean(tg > 3.5)),
            "btts_yes": float(np.mean((hg > 0) & (ag > 0))),
            "btts_no": float(np.mean((hg == 0) | (ag == 0))),
            "home_cs": float(np.mean(ag == 0)),
            "away_cs": float(np.mean(hg == 0)),
            "home_or_draw": float(np.mean(hg >= ag)),
            "ah_home_minus1":
