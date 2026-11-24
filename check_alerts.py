import os
import sys
import json
import datetime
from zoneinfo import ZoneInfo
import requests
import yfinance as yf

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "levels.json")
POI_BAND_PCT = 0.001


def is_ny_session():
    event_name = os.getenv("GITHUB_EVENT_NAME", "")
    if event_name == "workflow_dispatch":
        return True
    ny_now = datetime.datetime.now(ZoneInfo("America/New_York"))
    if ny_now.weekday() >= 5:
        return False
    return 8 <= ny_now.hour < 17


def get_live_xauusd():
    try:
        resp = requests.get(
            "https://data-asg.goldprice.org/dbXRates/USD",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            if items:
                return float(items[0]["xauPrice"])
    except Exception:
        pass

    gc = yf.Ticker("GC=F").history(period="1d", interval="1m")
    if not gc.empty:
        return float(gc["Close"].iloc[-1])
    return None


def main():
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        print("NTFY_TOPIC secret not set. Skipping push check.")
        sys.exit(0)

    if not is_ny_session():
        print("Outside NY session (08:00-17:00 ET). Exiting.")
        sys.exit(0)

    if not os.path.exists(DATA_PATH):
        print("levels.json not found.")
        sys.exit(0)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    price = get_live_xauusd()
    if not price:
        print("Could not fetch live XAUUSD price.")
        sys.exit(0)

    levels = data.get("levels", {})
    flip = levels.get("gamma_flip", price)
    regime = "STABILIZING (Fade Walls)" if price >= flip else "VOLATILE (Favor Breakouts)"

    poi_candidates = [
        ("Call Wall", levels.get("call_wall"), "SELL"),
        ("Put Wall", levels.get("put_wall"), "BUY"),
        ("Gamma Flip", levels.get("gamma_flip"), "PIVOT"),
        ("Peak GEX", levels.get("peak_gex"), "MAGNET"),
        ("IV Wall Above", levels.get("iv_wall_above"), "SELL"),
        ("IV Wall Below", levels.get("iv_wall_below"), "BUY"),
    ]

    for item in data.get("top_positive_gex", [])[:2]:
        poi_candidates.append((f"Top +GEX ({item['strike']})", item["strike"], "BUY"))
    for item in data.get("top_negative_gex", [])[:2]:
        poi_candidates.append((f"Top -GEX ({item['strike']})", item["strike"], "SELL"))

    triggered = []
    for name, lvl, bias in poi_candidates:
        if not lvl:
            continue
        dist_pct = abs(price - lvl) / lvl
        if dist_pct <= POI_BAND_PCT:
            triggered.append((name, lvl, bias, dist_pct * 100))

    if not triggered:
        print(f"XAUUSD at {price:.2f} — no POI zones breached.")
        return

    triggered.sort(key=lambda x: x[3])
    name, lvl, bias, pct = triggered[0]
    title = f"XAUUSD {bias} ALERT @ {price:.2f}"
    body = (
        f"Price entered {name} POI ({lvl:.2f}, {pct:.3f}% away).\n"
        f"Regime: {regime}\n"
        f"Note: GitHub cron alerts may be delayed."
    )

    resp = requests.post(
        f"https://ntfy.sh/{topic}",
        data=body.encode("utf-8"),
        headers={
            "Title": title,
            "Priority": "high",
            "Tags": "rotating_light,chart_with_upwards_trend",
        },
        timeout=10,
    )
    print(f"ntfy.sh push sent ({resp.status_code}): {title}")


if __name__ == "__main__":
    main()
