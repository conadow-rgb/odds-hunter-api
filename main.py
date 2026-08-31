from fastapi import FastAPI
import requests
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

if not BZZOIRO_KEY:
    raise RuntimeError("BZZOIRO_KEY missing. Set it in Render Environment Variables.")
if not SHARPAPI_KEY:
    raise RuntimeError("SHARPAPI_KEY missing. Set it in Render Environment Variables.")

SAST = ZoneInfo("Africa/Johannesburg")

# Bzzoiro
HEADERS_BZZ = {"Authorization": f"Bearer {BZZOIRO_KEY}"}
BASE_BZZ = "https://api.bzzoiro.com/v1"  # Adjust if different

# SharpAPI
HEADERS_SHARP = {"X-API-Key": SHARPAPI_KEY}
BASE_SHARP = "https://api.sharpapi.io/api/v1"

# ─── ENDPOINTS ──────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ODDS HUNTER API is live", "endpoints": ["/health", "/analyze", "/diagnostic"]}

@app.get("/health")
async def health():
    return {"status": "alive", "time": datetime.now(SAST).isoformat()}

@app.get("/diagnostic")
async def diagnostic():
    """Test Bzzoiro + SharpAPI integration."""
    today = datetime.now(SAST).strftime("%Y-%m-%d")
    diag = {"date": today, "bzzoiro": {}, "sharpapi": {}}

    # Test Bzzoiro — try common endpoint patterns
    endpoints_to_test = [
        f"{BASE_BZZ}/fixtures",
        f"{BASE_BZZ}/matches",
        f"{BASE_BZZ}/predictions",
        f"{BASE_BZZ}/games",
        "https://sports.bzzoiro.com/api/v1/fixtures",
        "https://sports.bzzoiro.com/api/v1/matches",
        "https://sports.bzzoiro.com/api/v1/predictions",
    ]

    for url in endpoints_to_test:
        try:
            r = requests.get(url, headers=HEADERS_BZZ, params={"sport": "soccer", "date": today, "per_page": 3}, timeout=15)
            if r.status_code == 200:
                diag["bzzoiro"]["working_url"] = url
                diag["bzzoiro"]["status"] = r.status_code
                diag["bzzoiro"]["preview"] = str(r.text)[:800]
                break
        except Exception as e:
            diag["bzzoiro"][url] = {"error": str(e)[:100]}

    # Test SharpAPI
    try:
        r = requests.get(f"{BASE_SHARP}/odds", headers=HEADERS_SHARP,
                        params={"sport": "soccer", "date": today, "per_page": 10}, timeout=15)
        data = r.json()
        odds = data.get("data", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        diag["sharpapi"] = {"status": r.status_code, "odds_count": len(odds)}
    except Exception as e:
        diag["sharpapi"] = {"error": str(e)}

    return diag

@app.get("/analyze")
@app.post("/analyze")
async def analyze():
    try:
        engine = BzzoiroEngine()
        return engine.run()
    except Exception as e:
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}

# ─── BZZOIRO ENGINE ─────────────────────────────────
class BzzoiroEngine:
    def __init__(self):
        self.bzz_base = self._discover_bzzoiro_base()

    def _discover_bzzoiro_base(self):
        """Try to find the correct Bzzoiro base URL."""
        candidates = [
            "https://api.bzzoiro.com/v1",
            "https://sports.bzzoiro.com/api/v1",
            "https://api.sports.bzzoiro.com/v1",
        ]
        for base in candidates:
            try:
                r = requests.get(f"{base}/health", headers=HEADERS_BZZ, timeout=10)
                if r.status_code == 200:
                    return base
            except:
                pass
        # Default to most likely
        return "https://sports.bzzoiro.com/api/v1"

    def fetch_bzzoiro_fixtures(self):
        """Fetch fixtures + predictions from Bzzoiro."""
        today = datetime.now(SAST).strftime("%Y-%m-%d")
        url = f"{self.bzz_base}/fixtures"
        params = {"sport": "soccer", "date": today, "per_page": 100}
        r = requests.get(url, headers=HEADERS_BZZ, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            return data.get("data", [])
        if isinstance(data, list):
            return data
        return []

    def fetch_sharpapi_odds(self):
        """Fetch ALL odds from SharpAPI."""
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
        return all_odds

    def match_odds_to_fixture(self, fixture, all_odds):
        """Match SharpAPI odds to Bzzoiro fixture by team names."""
        home = ""
        away = ""
        if isinstance(fixture, dict):
            home = fixture.get("home_team", "") if isinstance(fixture.get("home_team"), str) else fixture.get("home", {}).get("name", "")
            away = fixture.get("away_team", "") if isinstance(fixture.get("away_team"), str) else fixture.get("away", {}).get("name", "")
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

    def scan_markets(self, fixture, odds):
        """Compare Bzzoiro ML predictions vs SharpAPI odds."""
        picks = []

        if not isinstance(fixture, dict):
            return picks

        home = fixture.get("home_team", "") if isinstance(fixture.get("home_team"), str) else fixture.get("home", {}).get("name", "Home")
        away = fixture.get("away_team", "") if isinstance(fixture.get("away_team"), str) else fixture.get("away", {}).get("name", "Away")
        league = fixture.get("league", "Unknown")
        match_id = fixture.get("id", f"{home}-{away}")

        # Get Bzzoiro predictions
        predictions = fixture.get("predictions", {}) if isinstance(fixture.get("predictions"), dict) else {}
        if not predictions:
            predictions = fixture.get("ml", {}) if isinstance(fixture.get("ml"), dict) else {}
        if not predictions:
            predictions = fixture.get("probabilities", {}) if isinstance(fixture.get("probabilities"), dict) else {}

        home_prob = predictions.get("home_win", predictions.get("home", 0.33))
        draw_prob = predictions.get("draw", predictions.get("draw", 0.33))
        away_prob = predictions.get("away_win", predictions.get("away", 0.33))
        confidence = predictions.get("confidence", 0.5)

        # Markets to scan
        markets = [
            ("1X2", "Home Win", home_prob, "moneyline", ["home"]),
            ("1X2", "Draw", draw_prob, "moneyline", ["draw"]),
            ("1X2", "Away Win", away_prob, "moneyline", ["away"]),
        ]

        # Try to get Over/Under and BTTS from Bzzoiro if available
        ou = fixture.get("over_under", {}) if isinstance(fixture.get("over_under"), dict) else {}
        if ou:
            over25 = ou.get("over_2.5", 0)
            if over25:
                markets.append(("O/U 2.5", "Over 2.5", over25, "totals", ["over"]))
            under25 = ou.get("under_2.5", 0)
            if under25:
                markets.append(("O/U 2.5", "Under 2.5", 1 - over25 if over25 else under25, "totals", ["under"]))

        btts = fixture.get("btts", {}) if isinstance(fixture.get("btts"), dict) else {}
        if btts:
            btts_yes = btts.get("yes", 0)
            if btts_yes:
                markets.append(("BTTS", "Yes", btts_yes, "btts", ["yes"]))
            btts_no = btts.get("no", 0)
            if btts_no:
                markets.append(("BTTS", "No", btts_no, "btts", ["no"]))

        for mkt_name, sel, prob, mkt_type, keywords in markets:
            if prob < 0.15:
                continue

            best_odd = None
            book = "N/A"

            for odd in odds:
                if not isinstance(odd, dict):
                    continue
                if odd.get("market_type") != mkt_type:
                    continue

                sel_str = str(odd.get("selection", "")).lower()
                sel_type = str(odd.get("selection_type", "")).lower()

                matched = False
                for kw in keywords:
                    if kw in sel_str or kw in sel_type:
                        matched = True
                        break
                if not matched:
                    continue

                price = self.get_decimal_odds(odd)
                if price > 1.1 and price < 15.0 and (best_odd is None or price > best_odd):
                    best_odd = price
                    book = odd.get("sportsbook", "unknown")

            if not best_odd:
                continue

            implied = 1 / best_odd
            edge = prob - implied

            if edge >= 0.03 and prob >= 0.45:
                conf = "A+" if edge > 0.10 else "A" if edge > 0.06 else "B"
                exp = f"Bzzoiro ML model: {sel} = {prob*100:.1f}% (confidence {confidence*100:.0f}%). {book} @ {best_odd} (implied {implied*100:.1f}%). Edge: {edge*100:.1f}%."
                picks.append({
                    "match_id": match_id,
                    "fixture": f"{home} vs {away}",
                    "league": league,
                    "market": mkt_name,
                    "selection": sel,
                    "odds": best_odd,
                    "bookmaker": book,
                    "ml_probability": round(prob, 3),
                    "implied_probability": round(implied, 3),
                    "edge": round(edge, 3),
                    "confidence": conf,
                    "ml_confidence": round(confidence, 2),
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
                    avg_conf = sum(p["ml_probability"] for p in combo) / legs
                    ml_conf = sum(p["ml_confidence"] for p in combo) / legs
                    results.append({
                        "legs": list(combo),
                        "combined_odds": round(combined, 2),
                        "total_edge": round(total_edge, 3),
                        "avg_probability": round(avg_conf, 3),
                        "avg_ml_confidence": round(ml_conf, 2),
                        "leg_count": legs,
                        "rating": "A+" if total_edge > 0.25 else "A" if total_edge > 0.15 else "B"
                    })
            if results:
                results.sort(key=lambda x: (-x["total_edge"], abs(x["combined_odds"] - target)))
                return results[:5]
        return results

    def run(self):
        fixtures = self.fetch_bzzoiro_fixtures()
        if not fixtures:
            return {"status": "no_matches", "message": "No fixtures from Bzzoiro today."}

        all_odds = self.fetch_sharpapi_odds()

        all_picks = []
        for fixture in fixtures:
            match_odds = self.match_odds_to_fixture(fixture, all_odds)
            picks = self.scan_markets(fixture, match_odds)
            all_picks.extend(picks)

        if len(all_picks) < 2:
            return {"status": "no_value", "message": "No value markets found. Bzzoiro predictions may align with market odds."}

        accas = self.build_accas(all_picks)

        return {
            "status": "success",
            "timestamp": datetime.now(SAST).isoformat(),
            "timezone": "SAST",
            "source": "Bzzoiro ML + SharpAPI odds",
            "matches_scanned": len(fixtures),
            "markets_scanned": len(all_picks),
            "accumulators": accas
        }
