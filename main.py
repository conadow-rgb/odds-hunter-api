from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from itertools import combinations
from math import prod
import os

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
        return resp.json().get("data", [])
    
    def fetch_odds(self, leagues: list):
        all_odds = []
        for league in leagues:
            url = f"{BASE_SHARP}/odds"
            params = {
                "league": league,
                "market_type": "moneyline,totals,btts,spreads"
            }
            resp = requests.get(url, headers=HEADERS_SHARP, params=params, timeout=15)
            if resp.status_code == 200:
                all_odds.extend(resp.json().get("data", []))
        return all_odds
    
    def match_odds_to_fixture(self, fixture: dict, all_odds: list):
        home = fixture.get("home", {}).get("name", "")
        if not home:
            home = fixture.get("home_team", {}).get("name", "")
        away = fixture.get("away", {}).get("name", "")
        if not away:
            away = fixture.get("away_team", {}).get("name", "")
        
        match_odds = []
        for odd in all_odds:
            odd_home = odd.get("home_team", "")
            odd_away = odd.get("away_team", "")
            if (home.lower() in odd_home.lower() or odd_home.lower() in home.lower()) and \
               (away.lower() in odd_away.lower() or odd_away.lower() in away.lower()):
                match_odds.append(odd)
        return match_odds
    
    def calc_lambda(self, match: dict):
        league_avg = 2.65
        
        hs = match.get("home_team", {}).get("recent_stats", {}) if "home_team" in match else {}
        as_ = match.get("away_team", {}).get("recent_stats", {}) if "away_team" in match else {}
        
        if not hs or not as_:
            return 1.4, 1.2
        
        home_att = hs.get("goals_scored_pg", 1.4) / (league_avg / 2)
        home_def = hs.get("goals_conceded_pg", 1.1) / (league_avg / 2)
        away_att = as_.get("goals_scored_pg", 1.2) / (league_avg / 2)
        away_def = as_.get("goals_conceded_pg", 1.3) / (league_avg / 2)
        
        elo = match.get("predictions", {}).get("elo", {})
        elo_diff = (elo.get("home_elo", 1500) - elo.get("away_elo", 1500)) / 400
        elo_mult = 10 ** elo_diff
        
        base_h = (league_avg / 2) * home_att * away_def
        base_a = (league_avg / 2) * away_att * home_def
        
        lambda_h = base_h * self.home_adv * elo_mult
        lambda_a = base_a / elo_mult
        
        home_inj = len([i for i in match.get("injuries", []) if i.get("team") == "home" and i.get("impact") == "high"])
        away_inj = len([i for i in match.get("injuries", []) if i.get("team") == "away" and i.get("impact") == "high"])
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
            "ah_home_minus1": float(np.mean((hg - ag) >= 1)),
        }
    
    def scan_markets(self, match: dict, sim: dict, match_odds: list):
        picks = []
        mid = match.get("id", "unknown")
        home = match.get("home_team", {}).get("name", "Home")
        away = match.get("away_team", {}).get("name", "Away")
        league = match.get("league", {}).get("name", "Unknown")
        
        markets = [
            ("1X2", "Home Win", "home_win", "moneyline", "home"),
            ("1X2", "Draw", "draw", "moneyline", "draw"),
            ("1X2", "Away Win", "away_win", "moneyline", "away"),
            ("O/U 2.5", "Over 2.5", "over_2.5", "totals", "over"),
            ("O/U 2.5", "Under 2.5", "over_2.5", "totals", "under"),
            ("O/U 3.5", "Over 3.5", "over_3.5", "totals", "over"),
            ("BTTS", "Yes", "btts_yes", "btts", "yes"),
            ("BTTS", "No", "btts_no", "btts", "no"),
            ("AH -1", "Home -1", "ah_home_minus1", "spreads", "home"),
        ]
        
        for mkt_name, sel, sim_key, mkt_type, selection_key in markets:
            prob = sim.get(sim_key, 0)
            if prob < 0.15:
                continue
            
            best_odd = None
            book = "N/A"
            
            for odd in match_odds:
                if odd.get("market_type") != mkt_type:
                    continue
                if selection_key.lower() in odd.get("selection", "").lower():
                    price = odd.get("odds_decimal", 0)
                    if price > 1.1 and (best_odd is None or price > best_odd):
                        best_odd = price
                        book = odd.get("sportsbook", "unknown")
            
            if not best_odd:
                continue
            
            implied = 1 / best_odd
            if "Under" in sel:
                prob = 1 - prob
            
            edge = prob - implied
            if edge >= 0.03 and prob >= 0.50:
                conf = "A+" if edge > 0.10 else "A" if edge > 0.07 else "B"
                exp = f"Monte Carlo ({self.simulations:,} runs): {sel} = {prob*100:.1f}%. {book} @ {best_odd} (implied {implied*100:.1f}%). Edge: {edge*100:.1f}%."
                
                picks.append({
                    "match_id": mid,
                    "fixture": f"{home} vs {away}",
                    "league": league,
                    "market": mkt_name,
                    "selection": sel,
                    "odds": best_odd,
                    "bookmaker": book,
                    "sim_probability": round(prob, 3),
                    "implied_probability": round(implied, 3),
                    "edge": round(edge, 3),
                    "confidence": conf,
                    "explanation": exp
                })
        
        picks.sort(key=lambda x: x["edge"], reverse=True)
        return picks
    
    def build_accas(self, picks: list, target: float = 10.0, max_legs: int = 6):
        if len(picks) < 2:
            return []
        
        results = []
        for legs in range(2, max_legs + 1):
            for combo in combinations(picks, legs):
                mids = [p["match_id"] for p in combo]
                if len(set(mids)) != legs:
                    continue
                
                combined = prod(p["odds"] for p in combo)
                if combined >= target:
                    total_edge = sum(p["edge"] for p in combo)
                    avg_conf = sum(p["sim_probability"] for p in combo) / legs
                    results.append({
                        "legs": list(combo),
                        "combined_odds": round(combined, 2),
                        "total_edge": round(total_edge, 3),
                        "avg_confidence": round(avg_conf, 3),
                        "leg_count": legs,
                        "rating": "A+" if total_edge > 0.25 else "A" if total_edge > 0.15 else "B"
                    })
            
            if results:
                results.sort(key=lambda x: (-x["total_edge"], abs(x["combined_odds"] - target)))
                return results[:5]
        return results
    
    def run(self):
        fixtures = self.fetch_fixtures()
        if not fixtures:
            return {"status": "no_matches", "message": "No matches today."}
        
        leagues = list(set(f.get("league", {}).get("name", "EPL") for f in fixtures))
        all_odds = self.fetch_odds(leagues)
        
        all_picks = []
        for match in fixtures:
            match_odds = self.match_odds_to_fixture(match, all_odds)
            lh, la = self.calc_lambda(match)
            sim = self.simulate(lh, la)
            picks = self.scan_markets(match, sim, match_odds)
            all_picks.extend(picks)
        
        if len(all_picks) < 2:
            return {"status": "no_value", "message": "No value markets found today."}
        
        accas = self.build_accas(all_picks)
        
        return {
            "status": "success",
            "timestamp": datetime.now(SAST).isoformat(),
            "timezone": "SAST",
            "simulations": self.simulations,
            "matches_scanned": len(fixtures),
            "markets_scanned": len(all_picks),
            "accumulators": accas
        }

# ─── FASTAPI ENDPOINTS ──────────────────────────────
@app.get("/")
async def root():
    return {"status": "ODDS HUNTER API is live", "endpoints": ["/health", "/analyze"]}

@app.get("/health")
async def health():
    return {"status": "alive", "time": datetime.now(SAST).isoformat()}

@app.get("/analyze")
@app.post("/analyze")
async def analyze():
    try:
        engine = MonteCarloEngine(simulations=20000)
        result = engine.run()
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}
