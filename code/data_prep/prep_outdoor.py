"""Real outdoor PM2.5 for the simulations: hourly readings of the U.S. Embassy Dhaka monitor.

Source: the AirNow feed of the U.S. Embassy Dhaka monitor as relayed on public ThingSpeak channel 1541265
(field1 'AirNow PM2.5', hourly, 2021-10-18 to 2024-02-01; the official AirNow embassy archive was taken offline in
March 2025). Provenance check: the relay's daily means are compared with the official daily means of the same
monitor published by the U.S. EPA (Sarwar et al. 2023, data DOI 10.23719/1528454, Figure_1_data.xlsx) for the
overlapping period (2021-10-19 to 2021-12-31).
Output: data/outdoor_dhaka_pm25.csv (local time, hourly), data/outdoor_dhaka_days.csv (working days with complete
07:00-17:00 coverage), results/platform/outdoor_check.json."""
import json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TS = os.path.join(ROOT, "data", "external", "airnow_dhaka")
EPA = os.path.join(ROOT, "data", "external", "epa_dhaka", "Figure_1_data.xlsx")
OUT = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results", "platform")


def load_relay():
    rows = []
    for f in sorted(os.listdir(TS)):
        if f.endswith(".json"):
            rows += json.load(open(os.path.join(TS, f), encoding="utf-8"))["feeds"]
    d = pd.DataFrame(rows)
    d["t_utc"] = pd.to_datetime(d.created_at, utc=True)
    d["pm25"] = pd.to_numeric(d.field1, errors="coerce")
    d = d.drop_duplicates("t_utc").set_index("t_utc").sort_index()
    d["t_local"] = d.index.tz_convert("Asia/Dhaka").tz_localize(None)
    s = d.set_index("t_local").pm25
    s[(s < 0) | (s > 1000)] = np.nan                                     # invalid codes / outliers
    return s


def load_epa_daily():
    x = pd.read_excel(EPA, sheet_name="data", header=None)
    x = x.iloc[2:, :3]
    x.columns = ["jday", "pm25", "year"]
    x = x.dropna(subset=["pm25"])
    x["jday"] = x.jday.astype(int)
    x["date"] = pd.Timestamp("2016-01-01") + pd.to_timedelta(x.jday - 1, unit="D")
    return x.set_index("date").pm25.astype(float)


def main():
    os.makedirs(RES, exist_ok=True)
    s = load_relay()
    epa = load_epa_daily()
    # provenance check on the overlapping period (daily means of days with >= 18 valid hours)
    day = s.groupby(s.index.normalize()).agg(["mean", "count"])
    day = day[day["count"] >= 18]["mean"]
    common = day.index.intersection(epa.index)
    a, b = day[common], epa[common]
    chk = {"n_days_overlap": int(len(common)), "relay_mean": float(a.mean()), "epa_mean": float(b.mean()),
           "r": float(np.corrcoef(a, b)[0, 1]) if len(common) > 2 else None,
           "median_abs_rel_diff": float(np.median(np.abs(a - b) / b)) if len(common) else None,
           "epa_last_date": str(epa.index.max().date()), "relay_first": str(s.index.min()), "relay_last": str(s.index.max()),
           "relay_hours": int(s.notna().sum())}
    # working days (Sunday-Thursday) with >= 10 of the 11 hourly values from 07:00 to 17:00
    hourly = s.resample("1h").mean()
    days = []
    for d0, g in hourly.groupby(hourly.index.normalize()):
        w = g.between_time("07:00", "17:00")
        if d0.dayofweek in (4, 5):                                      # Friday, Saturday: weekend in Bangladesh
            continue
        if w.notna().sum() >= 10 and len(w) >= 11:
            days.append({"date": d0.date(), "mean_0717": float(w.mean()), "min": float(w.min()), "max": float(w.max())})
    days = pd.DataFrame(days)
    chk.update({"working_days": int(len(days)), "day_mean_median": float(days.mean_0717.median()),
                "day_mean_p5": float(days.mean_0717.quantile(0.05)), "day_mean_p95": float(days.mean_0717.quantile(0.95))})
    hourly.rename("pm25").to_frame().to_csv(os.path.join(OUT, "outdoor_dhaka_pm25.csv"))
    days.to_csv(os.path.join(OUT, "outdoor_dhaka_days.csv"), index=False)
    json.dump(chk, open(os.path.join(RES, "outdoor_check.json"), "w"), indent=1)
    print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
