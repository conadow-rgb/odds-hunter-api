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
BZZOIRO_KEY = os.getenv("BZZOIRO_KEY")
SHARPAPI_KEY = os.getenv("SHARPAPI_KEY")

if not SHARPAPI_KEY:
    raise RuntimeError("SHARPAPI_KEY missing. Set it in Render Environment Variables.")

SAST = ZoneInfo("Africa/Johannesburg")
HEADERS_BZZ = {"Authorization": f"Bearer {BZZOIRO_KEY}"} if BZZOIRO_KEY else {}
HEADERS_SHARP = {"X-API-Key": SHARPAPI_KEY}
BASE_SHARP = "https://api.sharpapi.io/api/v1"

# ─── REALISTIC ODDS RANGES ──────────────────────────
# Reject odds outside these ranges per market type
ODDS_RANGES = {
    "moneyline": {"min": 1.05, "max": 15.0},
    "totals": {"min": 1.3, "max": 5.0},
    "both_teams_to_score": {"min": 1.3, "max": 5.0},
    "double_chance": {"min": 1.05, "max": 5.0},
    "spreads": {"min": 1.3, "max": 5.0},
}

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
    return {"status": "ODDS HUNTER API is live", "endpoints": ["/health", "/analyze", "/diagnostic", "/scan-debug"]}

@app.get("/health")
async def health():
    return {"status": "alive", "time": datetime.now(SAST).isoformat()}

@app.get("/diagnostic")
async def diagnostic():
    today = datetime.now(SAST).strftime("%Y-%m-%d")
    diag = {"date": today, "bzzoiro": {}, "sharpapi": {}}

    if BZZOIRO_KEY:
        urls = [
            "https://api.bzzoiro.com/v1/fixtures",
            "https://sports.bzzoiro.com/api/v1/fixtures",
        ]
        for url in urls:
            try:
                r = requests.get(url, headers=HEADERS_BZZ, params={"sport": "soccer", "date": today, "per_page": 3}, timeout=10)
                diag["bzzoiro"]["last_tested_url"] = url
                diag["bzzoiro"]["status"] = r.status_code
                if r.status_code == 200:
                    diag["bzzoiro"]["preview"] = str(r.text)[:500]
                    break
            except Exception as e:
                diag["bzzoiro"][url] = {"error": str(e)[:80]}
    else:
        diag["bzzoiro"]["status"] = "no_key_configured"

    try:
        r = requests.get(f"{BASE_SHARP}/odds", headers=HEADERS_SHARP,
                        params={"sport": "soccer", "date": today, "per_page": 10}, timeout=15)
        data = r.json()
        odds = data.get("data", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        diag["sharpapi"] = {"status": r.status_code, "odds_count": len(odds)}
    except Exception as e:
        diag["sharpapi"] = {"error": str(e)}

    return diag

@app.get("/scan-debug")
async def scan_debug():
    today = datetime.now(SAST).strftime("%Y-%m-%d")
    debug = {"date": today, "matches": [], "summary": {"total_odds": 0, "realistic_odds": 0, "exotic_odds": 0}}

    all_odds = []
    for page in range(1, 3):
        r = requests.get(f"{BASE_SHARP}/odds", headers=HEADERS_SHARP,
                        params={"sport": "soccer", "date": today, "per_page": 100, "page": page}, timeout=15)
        if r.status_code != 200:
            break
        data = r.json()
        odds_page = data.get("data", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        if not odds_page:
            break
        all_odds.extend(odds_page)

    debug["summary"]["total_odds"] = len(all_odds)

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
            matches[key] = {"league": odd.get("league"), "odds": []}
        matches[key]["odds"].append(odd)

    for key, match in list(matches.items())[:5]:
        odd_debug = []
        realistic_count = 0
        exotic_count = 0

        for odd in match["odds"]:
            mkt_type = str(odd.get("market_type", "")).lower()
            price = get_decimal_odds(odd)

            # Check if realistic
            rng = ODDS_RANGES.get(mkt_type, {"min": 1.0, "max": 100.0})
            is_realistic = rng["min"] <= price <= rng["max"]

            if is_realistic:
                realistic_count += 1
            else:
                exotic_count += 1

            odd_debug.append({
                "market_type": mkt_type,
                "selection": odd.get("selection"),
                "price": price,
                "realistic": is_realistic,
                "range": rng,
            })

        debug["matches"].append({
            "fixture": key,
            "league": match["league"],
            "total_odds": len(match["odds"]),
            "realistic_odds": realistic_count,
            "exotic_odds": exotic_count,
            "odds_scan": odd_debug[:10]
        })

        debug["summary"]["realistic_odds"] += realistic_count
        debug["summary"]["exotic_odds"] += exotic_count

    return debug

@app.get("/analyze")
@app.post("/analyze")
async def analyze():
    try:
        engine = FallbackEngine(simulations=20000)
        return engine.run()
    except Exception as e:
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}

# ─── FALLBACK ENGINE ────────────────────────────────
class FallbackEngine:
    def __init__(self, simulations: int = 20000):
        self.simulations = simulations

    def run(self):
        today = datetime.now(SAST).strftime("%Y-%m-%d")

        # ── TRY BZZOIRO FIRST ──
        if BZZOIRO_KEY:
            try:
                bzz_fixtures = self._try_bzzoiro(today)
                if bzz_fixtures:
                    all_odds = self._fetch_sharpapi_odds(today)
                    result = self._analyze_with_bzzoiro(bzz_fixtures, all_odds)
                    if result.get("accumulators"):
                        result["source"] = "Bzzoiro ML + SharpAPI odds"
                        result["fallback"] = False
                        return result
            except Exception:
                pass

        # ── FALLBACK TO SHARPAPI + MONTE CARLO ──
        try:
            matches = self._fetch_sharpapi_matches(today)
            if matches:
                result = self._analyze_with_monte_carlo(matches)
                result["source"] = "SharpAPI odds + Monte Carlo simulation"
                result["fallback"] = True
                return result
        except Exception as e:
            return {"status": "error", "message": f"Both Bzzoiro and SharpAPI failed: {str(e)}"}

        return {"status": "no_matches", "message": "No matches or odds found today."}

    def _try_bzzoiro(self, today):
        urls = [
            "https://api.bzzoiro.com/v1/fixtures",
            "https://sports.bzzoiro.com/api/v1/fixtures",
        ]
        for url in urls:
            try:
                r = requests.get(url, headers=HEADERS_BZZ, params={"sport": "soccer", "date": today, "per_page": 100}, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    fixtures = data.get("data", []) if isinstance(data, dict) else data if isinstance(data, list) else []
                    if fixtures:
                        return fixtures
            except Exception:
                continue
        return None

    def _analyze_with_bzzoiro(self, fixtures, all_odds):
        all_picks = []
        for fixture in fixtures:
            if not isinstance(fixture, dict):
                continue
            home = fixture.get("home_team", "") if isinstance(fixture.get("home_team"), str) else fixture.get("home", {}).get("name", "Home")
            away = fixture.get("away_team", "") if isinstance(fixture.get("away_team"), str) else fixture.get("away", {}).get("name", "Away")
            league = fixture.get("league", "Unknown")
            match_id = fixture.get("id", f"{home}-{away}")

            predictions = fixture.get("predictions", {}) if isinstance(fixture.get("predictions"), dict) else {}
            if not predictions:
                predictions = fixture.get("ml", {}) if isinstance(fixture.get("ml"), dict) else {}
            if not predictions:
                predictions = fixture.get("probabilities", {}) if isinstance(fixture.get("probabilities"), dict) else {}

            home_prob = predictions.get("home_win", predictions.get("home", 0.33))
            draw_prob = predictions.get("draw", predictions.get("draw", 0.33))
            away_prob = predictions.get("away_win", predictions.get("away", 0.33))
            confidence = predictions.get("confidence", 0.5)

            match_odds = []
            for odd in all_odds:
                if not isinstance(odd, dict):
                    continue
                oh = str(odd.get("home_team", "")).lower()
                oa = str(odd.get("away_team", "")).lower()
                if (home.lower() in oh or oh in home.lower()) and (away.lower() in oa or oa in away.lower()):
                    match_odds.append(odd)

            markets = [
                ("1X2", "Home Win", home_prob, "moneyline", ["home"]),
                ("1X2", "Draw", draw_prob, "moneyline", ["draw"]),
                ("1X2", "Away Win", away_prob, "moneyline", ["away"]),
            ]

            for mkt_name, sel, prob, mkt_type, keywords in markets:
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
                    matched = any(kw in sel_str or kw in sel_type for kw in keywords)
                    if not matched:
                        continue
                    price = get_decimal_odds(odd)
                    rng = ODDS_RANGES.get(mkt_type, {"min": 1.0, "max": 100.0})
                    if rng["min"] <= price <= rng["max"] and (best_odd is None or price > best_odd):
                        best_odd = price
                        book = odd.get("sportsbook", "unknown")
                if not best_odd:
                    continue
                implied = 1 / best_odd
                edge = prob - implied
                if edge >= 0.03 and prob >= 0.45:
                    conf = "A+" if edge > 0.10 else "A" if edge > 0.06 else "B"
                    exp = f"Bzzoiro ML: {sel} = {prob*100:.1f}% (conf {confidence*100:.0f}%). {book} @ {best_odd} (implied {implied*100:.1f}%). Edge: {edge*100:.1f}%."
                    all_picks.append({
                        "match_id": match_id, "fixture": f"{home} vs {away}", "league": league,
                        "market": mkt_name, "selection": sel, "odds": best_odd,
                        "bookmaker": book, "ml_probability": round(prob, 3),
                        "implied_probability": round(implied, 3), "edge": round(edge, 3),
                        "confidence": conf, "ml_confidence": round(confidence, 2),
                        "explanation": exp
                    })

        accas = self._build_accas(all_picks)
        return {
            "status": "success" if accas else "no_value",
            "timestamp": datetime.now(SAST).isoformat(),
            "timezone": "SAST",
            "matches_scanned": len(fixtures),
            "markets_scanned": len(all_picks),
            "accumulators": accas
        }

    def _fetch_sharpapi_odds(self, today):
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
        return all_odds

    def _fetch_sharpapi_matches(self, today):
        all_odds = self._fetch_sharpapi_odds(today)
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
                    "odds": []
                }
            matches[key]["odds"].append(odd)
        return list(matches.values())

    def _analyze_with_monte_carlo(self, matches):
        all_picks = []
        for match in matches:
            sim = self._simulate(match["avg_goals"], match["home_adv"])
            picks = self._scan_mc_markets(match, sim)
            all_picks.extend(picks)

        accas = self._build_accas(all_picks)
        return {
            "status": "success" if accas else "no_value",
            "timestamp": datetime.now(SAST).isoformat(),
            "timezone": "SAST",
            "simulations": self.simulations,
            "matches_scanned": len(matches),
            "markets_scanned": len(all_picks),
            "accumulators": accas
        }

    def _simulate(self, avg_goals, home_adv):
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

    def _scan_mc_markets(self, match, sim):
        picks = []
        home = match["home"]
        away = match["away"]
        league = match["league"]

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

            price = get_decimal_odds(odd)
            if not price:
                continue

            # STRICT: Check realistic range
            rng = ODDS_RANGES.get(mkt_type, {"min": 1.0, "max": 100.0})
            if not (rng["min"] <= price <= rng["max"]):
                continue  # Skip exotic/prop odds

            selection = str(odd.get("selection", "")).lower()
            sel_type = str(odd.get("selection_type", "")).lower()
            book = odd.get("sportsbook", "unknown")

            for mkt_name, sel_label, sim_key, keywords in market_map[mkt_type]:
                matched = any(kw in selection or kw in sel_type for kw in keywords)
                if not matched:
                    continue

                prob = sim.get(sim_key, 0)
                if prob < 0.10:
                    continue

                implied = 1 / price
                if "Under" in sel_label or "No" in sel_label:
                    prob = 1 - prob

                edge = prob - implied
                if edge >= 0.02 and prob >= 0.40:
                    conf = "A+" if edge > 0.10 else "A" if edge > 0.06 else "B"
                    exp = f"Monte Carlo ({self.simulations:,} runs): {sel_label} = {prob*100:.1f}%. {book} @ {price} (implied {implied*100:.1f}%). Edge: {edge*100:.1f}%."
                    picks.append({
                        "match_id": f"{home}-{away}",
                        "fixture": f"{home} vs {away}",
                        "league": league,
                        "market": mkt_name,
                        "selection": sel_label,
                        "odds": price,
                        "bookmaker": book,
                        "sim_probability": round(prob, 3),
                        "implied_probability": round(implied, 3),
                        "edge": round(edge, 3),
                        "confidence": conf,
                        "explanation": exp
                    })

        picks.sort(key=lambda x: x["edge"], reverse=True)
        return picks

    def _build_accas(self, picks, target=10.0, max_legs=6):
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
                    avg_conf = sum(p.get("sim_probability", p.get("ml_probability", 0)) for p in combo) / legs
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

# ─── STANDALONE HELPER ──────────────────────────────
def get_decimal_odds(odd):
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
