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
BIG_BALLS_KEY = os.getenv("BIG_BALLS_KEY")
SHARPAPI_KEY = os.getenv("SHARPAPI_KEY")

if not BIG_BALLS_KEY or not SHARPAPI_KEY:
    raise RuntimeError("API keys missing. Set BIG_BALLS_KEY and SHARPAPI_KEY environment variables.")

SAST = ZoneInfo("Africa/Johannesburg")
HEADERS_BB = {"Authorization": f"Bearer {BIG_BALLS_KEY}"}
HEADERS_SHARP = {"X-API-Key": SHARPAPI_KEY}
BASE_BB = "https://api.bigballsdata.com/v1"
BASE_SHARP = "https://api.sharpapi.io/api/v1"

# ─── ENDPOINTS ──────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ODDS HUNTER API is live", "endpoints": ["/health", "/analyze", "/diagnostic", "/debug"]}

@app.get("/health")
async def health():
    return {"status": "alive", "time": datetime.now(SAST).isoformat()}

@app.get("/debug")
async def debug():
    out = {}
    today = datetime.now(SAST).strftime("%Y-%m-%d")
    try:
        r = requests.get(f"{BASE_BB}/matches", headers=HEADERS_BB,
                         params={"sport": "football", "date": today, "per_page": 3}, timeout=30)
        out["bigballs"] = {"status": r.status_code, "preview": str(r.text)[:800]}
    except Exception as e:
        out["bigballs"] = {"error": str(e)}
    try:
        r = requests.get(f"{BASE_SHARP}/odds", headers=HEADERS_SHARP,
                         params={"sport": "soccer", "date": today, "per_page": 10}, timeout=15)
        out["sharpapi"] = {"status": r.status_code, "preview": str(r.text)[:800]}
    except Exception as e:
        out["sharpapi"] = {"error": str(e)}
    return out

@app.get("/diagnostic")
async def diagnostic():
    """Shows exactly what both APIs return — fixtures, odds, and matching."""
    today = datetime.now(SAST).strftime("%Y-%m-%d")
    diag = {"date": today, "fixtures": [], "odds_summary": {}, "matches_with_odds": 0, "value_picks": []}

    # ── Fetch fixtures from Big Balls ──
    try:
        r = requests.get(f"{BASE_BB}/matches", headers=HEADERS_BB,
                         params={"sport": "football", "date": today, "include": "stats,elo,lineups,injuries", "per_page": 100}, timeout=30)
        fixtures = r.json().get("data", []) if isinstance(r.json(), dict) else r.json() if isinstance(r.json(), list) else []
    except Exception as e:
        return {"error": f"Big Balls failed: {str(e)}"}

    diag["fixture_count"] = len(fixtures)

    # ── Fetch ALL odds from SharpAPI (multiple pages if needed) ──
    all_odds = []
    try:
        for page in range(1, 4):  # Try up to 3 pages
            r = requests.get(f"{BASE_SHARP}/odds", headers=HEADERS_SHARP,
                             params={"sport": "soccer", "date": today, "per_page": 100, "page": page}, timeout=15)
            if r.status_code == 200:
                data = r.json()
                odds_page = data.get("data", []) if isinstance(data, dict) else data if isinstance(data, list) else []
                if not odds_page:
                    break
                all_odds.extend(odds_page)
            else:
                break
    except Exception as e:
        return {"error": f"SharpAPI failed: {str(e)}"}

    diag["odds_summary"] = {
        "total_odds_fetched": len(all_odds),
        "first_10_odds_teams": []
    }

    # Show team names from first 10 odds
    for odd in all_odds[:10]:
        if isinstance(odd, dict):
            diag["odds_summary"]["first_10_odds_teams"].append({
                "home": odd.get("home_team"),
                "away": odd.get("away_team"),
                "market": odd.get("market_type"),
                "selection": odd.get("selection"),
                "odds": odd.get("odds_decimal"),
                "book": odd.get("sportsbook")
            })

    # ── Match odds to fixtures ──
    for match in fixtures[:10]:
        if not isinstance(match, dict):
            continue
        home = match.get("home", {}).get("name", "") if isinstance(match.get("home"), dict) else ""
        away = match.get("away", {}).get("name", "") if isinstance(match.get("away"), dict) else ""
        league = match.get("league", "Unknown") if isinstance(match, dict) else "Unknown"

        # Find matching odds
        match_odds = []
        for odd in all_odds:
            if not isinstance(odd, dict):
                continue
            oh = str(odd.get("home_team", "")).lower()
            oa = str(odd.get("away_team", "")).lower()
            if home and away and (home.lower() in oh or oh in home.lower()) and (away.lower() in oa or oa in away.lower()):
                match_odds.append({
                    "book": odd.get("sportsbook"),
                    "market": odd.get("market_type"),
                    "selection": odd.get("selection"),
                    "selection_type": odd.get("selection_type"),
                    "odds": odd.get("odds_decimal")
                })

        # Get stats availability
        h_stats = match.get("home", {}).get("recent_stats") if isinstance(match.get("home"), dict) else None
        a_stats = match.get("away", {}).get("recent_stats") if isinstance(match.get("away"), dict) else None
        elo = match.get("predictions", {}).get("elo") if isinstance(match.get("predictions"), dict) else None

        diag["fixtures"].append({
            "fixture": f"{home} vs {away}",
            "league": league,
            "has_home_stats": bool(h_stats),
            "has_away_stats": bool(a_stats),
            "has_elo": bool(elo),
            "elo_home": elo.get("home_elo") if isinstance(elo, dict) else None,
            "elo_away": elo.get("away_elo") if isinstance(elo, dict) else None,
            "odds_found": len(match_odds),
            "sample_odds": match_odds[:5]
        })

        if match_odds:
            diag["matches_with_odds"] += 1

    return diag

@app.get("/analyze")
@app.post("/analyze")
async def analyze():
    try:
        engine = MonteCarloEngine(simulations=20000)
        return engine.run()
    except Exception as e:
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}

# ─── MONTE CARLO ENGINE ─────────────────────────────
class MonteCarloEngine:
    def __init__(self, simulations: int = 20000):
        self.simulations = simulations
        self.home_adv = 1.35

    def fetch_fixtures(self):
        today = datetime.now(SAST).strftime("%Y-%m-%d")
        url = f"{BASE_BB}/matches"
        params = {"sport": "football", "date": today, "include": "stats,elo,lineups,injuries", "per_page": 100}
        r = requests.get(url, headers=HEADERS_BB, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            return data.get("data", [])
        if isinstance(data, list):
            return data
        return []

    def fetch_all_odds(self):
        today = datetime.now(SAST).strftime("%Y-%m-%d")
        all_odds = []
        for page in range(1, 4):
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
        return all_odds

    def match_odds_to_fixture(self, fixture, all_odds):
        home = ""
        away = ""
        if isinstance(fixture, dict):
            h = fixture.get("home")
            if isinstance(h, dict):
                home = h.get("name", "")
            a = fixture.get("away")
            if isinstance(a, dict):
                away = a.get("name", "")
        if not home or not away:
            return []
        match_odds = []
        for odd in all_odds:
            if not isinstance(odd, dict):
                continue
            oh = str(odd.get("home_team", "")).lower()
            oa = str(odd.get("away_team", "")).lower()
            if (home.lower() in oh or oh in home.lower()) and (away.lower() in oa or oa in away.lower()):
                match_odds.append(odd)
        return match_odds

    def calc_lambda(self, match):
        if not isinstance(match, dict):
            return 1.4, 1.2
        league_avg = 2.65
        h = match.get("home", {}) if isinstance(match.get("home"), dict) else {}
        a = match.get("away", {}) if isinstance(match.get("away"), dict) else {}
        hs = h.get("recent_stats", {}) if isinstance(h.get("recent_stats"), dict) else {}
        as_ = a.get("recent_stats", {}) if isinstance(a.get("recent_stats"), dict) else {}
        if not hs or not as_:
            return 1.4, 1.2
        home_att = hs.get("goals_scored_pg", 1.4) / (league_avg / 2)
        home_def = hs.get("goals_conceded_pg", 1.1) / (league_avg / 2)
        away_att = as_.get("goals_scored_pg", 1.2) / (league_avg / 2)
        away_def = as_.get("goals_conceded_pg", 1.3) / (league_avg / 2)
        elo = match.get("predictions", {}).get("elo", {}) if isinstance(match.get("predictions"), dict) else {}
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

    def simulate(self, lambda_h, lambda_a):
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

    def scan_markets(self, match, sim, match_odds):
        picks = []
        mid = match.get("id", "unknown") if isinstance(match, dict) else "unknown"
        home = match.get("home", {}).get("name", "Home") if isinstance(match, dict) and isinstance(match.get("home"), dict) else "Home"
        away = match.get("away", {}).get("name", "Away") if isinstance(match, dict) and isinstance(match.get("away"), dict) else "Away"
        league = match.get("league", "Unknown") if isinstance(match, dict) else "Unknown"
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
                if not isinstance(odd, dict):
                    continue
                if odd.get("market_type") != mkt_type:
                    continue
                sel_str = str(odd.get("selection", "")).lower()
                sel_type = str(odd.get("selection_type", "")).lower()
                if selection_key.lower() in sel_str or selection_key.lower() in sel_type:
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
                    "match_id": mid, "fixture": f"{home} vs {away}", "league": league,
                    "market": mkt_name, "selection": sel, "odds": best_odd,
                    "bookmaker": book, "sim_probability": round(prob, 3),
                    "implied_probability": round(implied, 3), "edge": round(edge, 3),
                    "confidence": conf, "explanation": exp
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
                        "legs": list(combo), "combined_odds": round(combined, 2),
                        "total_edge": round(total_edge, 3), "avg_confidence": round(avg_conf, 3),
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
        all_odds = self.fetch_all_odds()
        all_picks = []
        for match in fixtures:
            if not isinstance(match, dict):
                continue
            match_odds = self.match_odds_to_fixture(match, all_odds)
            lh, la = self.calc_lambda(match)
            sim = self.simulate(lh, la)
            picks = self.scan_markets(match, sim, match_odds)
            all_picks.extend(picks)
        if len(all_picks) < 2:
            return {"status": "no_value", "message": "No value markets found today."}
        accas = self.build_accas(all_picks)
        return {
            "status": "success", "timestamp": datetime.now(SAST).isoformat(),
            "timezone": "SAST", "simulations": self.simulations,
            "matches_scanned": len(fixtures), "markets_scanned": len(all_picks),
            "accumulators": accas
        }
