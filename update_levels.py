import os
import sys
import json
import math
import datetime
from zoneinfo import ZoneInfo
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "levels.json")


def check_ny_time():
    event_name = os.getenv("GITHUB_EVENT_NAME", "")
    if event_name == "workflow_dispatch":
        print("Manual dispatch detected; bypassing NY time window check.")
        return

    ny_now = datetime.datetime.now(ZoneInfo("America/New_York"))
    print(f"Current New York time: {ny_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    if not (ny_now.hour == 10 and 30 <= ny_now.minute <= 59):
        print("Outside 10:30-10:59 AM ET window (DST offset cron). Exiting cleanly.")
        sys.exit(0)


def get_spot_prices():
    xau_spot = None
    gld_spot = None

    try:
        resp = requests.get(
            "https://data-asg.goldprice.org/dbXRates/USD",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            if items:
                xau_spot = float(items[0]["xauPrice"])
    except Exception as e:
        print(f"goldprice.org fetch failed: {e}")

    if not xau_spot:
        gc = yf.Ticker("GC=F").history(period="1d", interval="1m")
        if not gc.empty:
            xau_spot = float(gc["Close"].iloc[-1])

    gld_hist = yf.Ticker("GLD").history(period="1d", interval="1m")
    if not gld_hist.empty:
        gld_spot = float(gld_hist["Close"].iloc[-1])

    if not xau_spot or not gld_spot:
        raise RuntimeError(f"Failed to resolve spot prices: XAU={xau_spot}, GLD={gld_spot}")

    return xau_spot, gld_spot


def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def parse_cboe_symbol(sym):
    idx = 0
    while idx < len(sym) and sym[idx].isalpha():
        idx += 1
    rest = sym[idx:]
    exp_str = rest[:6]
    opt_type = "call" if rest[6] == "C" else "put"
    strike = int(rest[7:]) / 1000.0
    exp_date = datetime.datetime.strptime("20" + exp_str, "%Y%m%d").date()
    return exp_date, opt_type, strike


def fetch_cboe_options(gld_spot):
    url = "https://cdn.cboe.com/api/global/delayed_quotes/options/GLD.json"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    data_obj = payload.get("data", {})
    if data_obj.get("current_price"):
        gld_spot = float(data_obj["current_price"])

    options = data_obj.get("options", [])
    today = datetime.datetime.now(ZoneInfo("America/New_York")).date()
    max_date = today + datetime.timedelta(days=10)

    rows = []
    for opt in options:
        try:
            exp_date, opt_type, strike = parse_cboe_symbol(opt["option"])
            if not (today <= exp_date <= max_date):
                continue
            oi = float(opt.get("open_interest") or 0)
            vol = float(opt.get("volume") or 0)
            iv = float(opt.get("iv") or 0)
            gamma = float(opt.get("gamma") or 0)
            if gamma == 0 and iv > 0:
                T = max((exp_date - today).days, 0.5) / 365.0
                gamma = bs_gamma(gld_spot, strike, T, 0.045, iv)
            rows.append({
                "strike": strike,
                "type": opt_type,
                "oi": oi,
                "volume": vol,
                "iv": iv,
                "gamma": gamma,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("CBOE returned 0 valid 0-10 DTE contracts.")
    return df, gld_spot, "CBOE"


def fetch_yfinance_options(gld_spot):
    ticker = yf.Ticker("GLD")
    today = datetime.datetime.now(ZoneInfo("America/New_York")).date()
    max_date = today + datetime.timedelta(days=10)

    rows = []
    for exp_str in ticker.options:
        exp_date = datetime.datetime.strptime(exp_str, "%Y-%m-%d").date()
        if not (today <= exp_date <= max_date):
            continue
        T = max((exp_date - today).days, 0.5) / 365.0
        chain = ticker.option_chain(exp_str)
        for opt_type, sub_df in [("call", chain.calls), ("put", chain.puts)]:
            for _, r in sub_df.iterrows():
                strike = float(r["strike"])
                oi = float(r.get("openInterest") or 0)
                vol = float(r.get("volume") or 0)
                iv = float(r.get("impliedVolatility") or 0)
                gamma = bs_gamma(gld_spot, strike, T, 0.045, iv)
                rows.append({
                    "strike": strike,
                    "type": opt_type,
                    "oi": oi,
                    "volume": vol,
                    "iv": iv,
                    "gamma": gamma,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("yfinance returned 0 valid 0-10 DTE contracts.")
    return df, gld_spot, "yfinance"


def compute_levels(df, gld_spot, xau_spot, source):
    ratio = xau_spot / gld_spot

    df["raw_gex"] = df["gamma"] * df["oi"] * 100.0 * (gld_spot ** 2) * 0.01
    df["signed_gex"] = np.where(df["type"] == "call", df["raw_gex"], -df["raw_gex"])

    grouped = df.groupby("strike").agg(
        call_gex=("signed_gex", lambda s: s[s > 0].sum()),
        put_gex=("signed_gex", lambda s: s[s < 0].sum()),
        net_gex=("signed_gex", "sum"),
        total_oi=("oi", "sum"),
        weighted_iv=("iv", lambda s: np.average(s, weights=df.loc[s.index, "oi"] + 1)),
    ).reset_index()

    grouped["xau_strike"] = (grouped["strike"] * ratio).round(2)
    grouped = grouped.sort_values("xau_strike").reset_index(drop=True)

    call_wall_row = grouped.loc[grouped["call_gex"].idxmax()]
    call_wall = float(call_wall_row["xau_strike"])

    put_wall_row = grouped.loc[grouped["put_gex"].idxmin()]
    put_wall = float(put_wall_row["xau_strike"])

    peak_gex_row = grouped.loc[grouped["net_gex"].abs().idxmax()]
    peak_gex = float(peak_gex_row["xau_strike"])

    near = grouped[
        (grouped["xau_strike"] >= xau_spot * 0.92) &
        (grouped["xau_strike"] <= xau_spot * 1.08)
    ].reset_index(drop=True)
    if len(near) < 2:
        near = grouped

    gamma_flip = None
    best_dist = float("inf")
    for i in range(len(near) - 1):
        g1, g2 = near.loc[i, "net_gex"], near.loc[i + 1, "net_gex"]
        s1, s2 = near.loc[i, "xau_strike"], near.loc[i + 1, "xau_strike"]
        if (g1 <= 0 and g2 >= 0) or (g1 >= 0 and g2 <= 0):
            if g2 != g1:
                cross = s1 + (0 - g1) * (s2 - s1) / (g2 - g1)
            else:
                cross = (s1 + s2) / 2.0
            if abs(cross - xau_spot) < best_dist:
                best_dist = abs(cross - xau_spot)
                gamma_flip = round(float(cross), 2)

    if gamma_flip is None:
        zero_idx = near["net_gex"].abs().idxmin()
        gamma_flip = float(near.loc[zero_idx, "xau_strike"])

    grouped["iv_concentration"] = grouped["weighted_iv"] * grouped["total_oi"]
    above = grouped[grouped["xau_strike"] > xau_spot]
    below = grouped[grouped["xau_strike"] < xau_spot]

    iv_wall_above = (
        float(above.loc[above["iv_concentration"].idxmax(), "xau_strike"])
        if not above.empty else round(xau_spot * 1.01, 2)
    )
    iv_wall_below = (
        float(below.loc[below["iv_concentration"].idxmax(), "xau_strike"])
        if not below.empty else round(xau_spot * 0.99, 2)
    )

    top_pos = grouped[grouped["net_gex"] > 0].nlargest(5, "net_gex")
    top_neg = grouped[grouped["net_gex"] < 0].nsmallest(5, "net_gex")

    top_positive_gex = [
        {"strike": float(r["xau_strike"]), "gex": round(float(r["net_gex"]), 2)}
        for _, r in top_pos.iterrows()
    ]
    top_negative_gex = [
        {"strike": float(r["xau_strike"]), "gex": round(float(r["net_gex"]), 2)}
        for _, r in top_neg.iterrows()
    ]

    profile_df = grouped[
        (grouped["xau_strike"] >= xau_spot * 0.95) &
        (grouped["xau_strike"] <= xau_spot * 1.05)
    ].copy()
    profile_df["cum_gex"] = profile_df["net_gex"].cumsum()

    strike_profile = [
        {
            "strike": float(r["xau_strike"]),
            "gld_strike": float(r["strike"]),
            "net_gex": round(float(r["net_gex"]), 2),
            "cum_gex": round(float(r["cum_gex"]), 2),
            "call_gex": round(float(r["call_gex"]), 2),
            "put_gex": round(float(r["put_gex"]), 2),
        }
        for _, r in profile_df.iterrows()
    ]

    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "updated_at": now_utc,
        "source": source,
        "spot_xauusd": round(xau_spot, 2),
        "spot_gld": round(gld_spot, 2),
        "ratio": round(ratio, 4),
        "levels": {
            "call_wall": call_wall,
            "put_wall": put_wall,
            "gamma_flip": gamma_flip,
            "peak_gex": peak_gex,
            "iv_wall_above": iv_wall_above,
            "iv_wall_below": iv_wall_below,
        },
        "top_positive_gex": top_positive_gex,
        "top_negative_gex": top_negative_gex,
        "strike_profile": strike_profile,
    }


def main():
    check_ny_time()
    xau_spot, gld_spot = get_spot_prices()
    print(f"Spots -> XAUUSD: {xau_spot:.2f} | GLD: {gld_spot:.2f}")

    try:
        df, gld_spot, source = fetch_cboe_options(gld_spot)
    except Exception as e:
        print(f"CBOE fetch failed ({e}), falling back to yfinance...")
        df, gld_spot, source = fetch_yfinance_options(gld_spot)

    result = compute_levels(df, gld_spot, xau_spot, source)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Successfully wrote {DATA_PATH} using {source}.")


if __name__ == "__main__":
    main()
