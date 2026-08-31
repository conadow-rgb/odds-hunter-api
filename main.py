from fastapi import FastAPI
import requests
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from itertools import combinations
from math import prod
import os
import traceback

app = FastAPI()

# ─── CONFIG ─────────────────────────────────────────
SHARPAPI_KEY = os.getenv("SHARPAPI_KEY")

if not SHARPAPI_KEY:
    raise RuntimeError("SHARPAPI_KEY missing. Set it in Render Environment Variables.")

SAST = ZoneInfo("Africa/Johannesburg")
HEADERS_SHARP = {"X-API-Key": SHARPAPI_KEY}
BASE_SHARP = "https://api.sharpapi.io/api/v1"

# ─── LEAGUE PARAMETERS ──────────────────────────────
LEAGUE_PARAMS = {
    "premier_league": {"avg_goals": 2.7, "home_adv": 1.35},
    "la_liga": {"avg_goals": 2.5, "home_adv": 1.40},
    "serie_a": {"avg_goals": 2.4, "home_adv": 1.30},
    "bundesliga": {"avg_goals": 3.0, "home_adv": 1.35},
    "ligue_1": {"avg_goals": 2.3, "home_adv": 1.30},
    "mls": {"avg_goals": 2.9, "home_adv": 1.45},
    "champions_league": {"avg_goals": 2.8, "home_adv": 1.25},
    "default": {"avg_goals": 2.5, "home_adv": 1.35},
}

def get_league_params(league_name):
    name = str(league_name).lower()
    for key, params in LEAGUE_PARAMS.items():
        if key in name:
            return params
    return LEAGUE_PARAMS["default"]

# ─── ENDPOINTS ──────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ODDS HUNTER API is live", "endpoints": ["/health", "/analyze", "/diagnostic"]}

@app.get("/health")
async def health():
    return {"status": "alive", "time": datetime.now(SAST).isoformat()}

@app.get("/diagnostic")
async def diagnostic():
    today = datetime.now(SAST).strftime("%Y-%m-%d")
    try:
        r = requests.get(f"{BASE_SHARP}/odds", headers=HEADERS_SHARP,
                        params={"sport": "soccer", "date": today, "per_page": 100}, timeout=15)
        data = r.json()
        odds = data.get("data", []) if isinstance(data, dict) else data if isinstance(data, list) else []

        matches = {}
        for odd in odds:
            if not isinstance(odd, dict):
                continue
            home = str(odd.get("home_team", "")).strip()
            away = str(odd.get("away_team", "")).strip()
            if not home or not away:
                continue
            key = f"{home} vs {away}"
            if key not in matches:
                matches[key] = {"league": odd.get("league"), "markets": set()}
            matches[key]["markets"].add(odd.get("market_type"))

        for k in matches:
            matches[k]["markets"] = list(matches[k]["markets"])

        return {
            "date": today,
            "total_odds": len(odds),
            "unique_matches": len(matches),
            "sample_matches": dict(list(matches.items())[:10])
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/analyze")
@app.post("/analyze")
async def analyze():
    try:
        engine = OddsHunterEngine(simulations=20000)
        return engine.run()
    except Exception as e:
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}

# ─── ENGINE ───────────────────────────────────────────
class OddsHunterEngine:
    def __init__(self, simulations: int = 20000):
        self.simulations = simulations

    def fetch_sharpapi_odds(self):
        today = datetime.now(SAST).strftime("%Y-%m-%d")
        all_odds = []
        for page in range(1, 6):
            url = f"{BASE_SHARP}/odds"
            params = {"sport": "soccer", "date": today, "per_page": 100, "page": page}
            r = requests.get(url, headers=HEADERS_SHARP, params=params, timeout=15)
            if r.status_code != 200:
                break
            data = r.json()
            odds_page = data.get("data", []) if isinstance(data, dict) else data if isinstance(data, list) else []
            if not odds_page:
                break
            all_odds.extend(odds_page)

        matches = {}
        for odd in all_odds:
            if not isinstance(odd, dict):
                continue
            home = str(odd.get("home_team", "")).strip()
            away = str(odd.get("away_team", "")).strip()
            if not home or not away:
                continue
            key = f"{home} vs {away}"
            if key not in matches:
                league = str(odd.get("league", "default"))
                params = get_league_params(league)
                matches[key] = {
                    "home": home, "away": away, "league": league,
                    "avg_goals": params["avg_goals"], "home_adv": params["home_adv"],
                    "odds": [], "kickoff": odd.get("event_start_time")
                }
            matches[key]["odds"].append(odd)

        return list(matches.values())

    def simulate(self, avg_goals, home_adv):
        lambda_h = (avg_goals / 2) * home_adv
        lambda_a = avg_goals / 2
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

    def get_decimal_odds(self, odd):
        dec = odd.get("odds_decimal")
        if dec and float(dec) > 1.0:
            return float(dec)
        american = odd.get("odds_american")
        if american:
            am = int(american)
            if am > 0:
                return round((am / 100) + 1, 2)
            else:
                return round((100 / abs(am)) + 1, 2)
        return 0

    def scan_markets(self, match, sim):
        picks = []
        home = match["home"]
        away = match["away"]
        league = match["league"]

        # Map SharpAPI market_type names to our internal format
        # SharpAPI uses: moneyline, double_chance, both_teams_to_score, correct_score, totals, spreads
        market_map = {
            "moneyline": [
                ("1X2", "Home Win", "home_win", ["home"]),
                ("1X2", "Draw", "draw", ["draw"]),
                ("1X2", "Away Win", "away_win", ["away"]),
            ],
            "totals": [
                ("O/U 2.5", "Over 2.5", "over_2.5", ["over", "o"]),
                ("O/U 2.5", "Under 2.5", "over_2.5", ["under", "u"]),
            ],
            "both_teams_to_score": [
                ("BTTS", "Yes", "btts_yes", ["yes", "y"]),
                ("BTTS", "No", "btts_no", ["no", "n"]),
            ],
            "double_chance": [
                ("DC", "Home/Draw", "home_or_draw", ["home/draw", "1x"]),
                ("DC", "Away/Draw", "away_or_draw", ["away/draw", "x2"]),
            ],
            "spreads": [
                ("AH -1", "Home -1", "ah_home_minus1", ["home"]),
            ],
        }

        for odd in match["odds"]:
            if not isinstance(odd, dict):
                continue
            mkt_type = str(odd.get("market_type", "")).lower()
            if mkt_type not in market_map:
                continue

            best_odd = self.get_decimal_odds(odd)
            if not best_odd or best_odd < 1.1:
                continue

            selection = str(odd.get("selection", "")).lower()
            sel_type = str(odd.get("selection_type", "")).lower()
            book = odd.get("sportsbook", "unknown")

            for mkt_name, sel_label, sim_key, keywords in market_map[mkt_type]:
                matched = False
                for kw in keywords:
                    if kw in selection or kw in sel_type:
                        matched = True
                        break
                if not matched:
                    continue

                prob = sim.get(sim_key, 0)
                if prob < 0.10:  # Lowered from 0.15
                    continue

                implied = 1 / best_odd
                if "Under" in sel_label or "No" in sel_label:
                    prob = 1 - prob

                edge = prob - implied
                if edge >= 0.02 and prob >= 0.40:  # Lowered thresholds for obscure leagues
                    conf = "A+" if edge > 0.10 else "A" if edge > 0.06 else "B"
                    exp = f"Monte Carlo ({self.simulations:,} runs): {sel_label} = {prob*100:.1f}%. {book} @ {best_odd} (implied {implied*100:.1f}%). Edge: {edge*100:.1f}%."
                    picks.append({
                        "match_id": f"{home}-{away}",
                        "fixture": f"{home} vs {away}",
                        "league": league,
                        "market": mkt_name,
                        "selection": sel_label,
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

    def build_accas(self, picks, target=10.0, max_legs=6):
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
        matches = self.fetch_sharpapi_odds()
        if not matches:
            return {"status": "no_matches", "message": "No matches with odds found today."}

        all_picks = []
        for match in matches:
            sim = self.simulate(match["avg_goals"], match["home_adv"])
            picks = self.scan_markets(match, sim)
            all_picks.extend(picks)

        if len(all_picks) < 2:
            return {"status": "no_value", "message": "No value markets found today. Bookmakers may be too efficient, or markets are limited."}

        accas = self.build_accas(all_picks)

        return {
            "status": "success",
            "timestamp": datetime.now(SAST).isoformat(),
            "timezone": "SAST",
            "simulations": self.simulations,
            "matches_scanned": len(matches),
            "markets_scanned": len(all_picks),
            "accumulators": accas
        }
