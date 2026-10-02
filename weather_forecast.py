"""
自己算天气预报：扬州未来 7 天 · MOS + 集成版（双 NWP）
================================================================
模型池：
  M1 气候态 + EWMA 加权异常订正（窗口 15 天，更灵敏）
  M2 傅里叶季节回归（Ridge）
  M3 Holt-Winters 三次指数平滑
  M4 XGBoost 监督学习
  M5 GFS 数值预报（NOAA）
  M6 ECMWF 数值预报（欧洲中心）
集成：每个模型用回测段算 MAE，按 1/MAE 倒数加权平均
"""

import warnings, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = [
    "WenQuanYi Zen Hei", "Microsoft YaHei", "PingFang SC",
    "Noto Sans CJK SC", "SimHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

LAT, LON = 32.39, 119.42
HISTORY_YEARS = 10
FORECAST_DAYS = 7
BACKTEST_DAYS = 60
ANOMALY_WINDOW = 15           # ★ 近期异常订正窗口（缩短，更灵敏反映季节转换）
NWP_BACKTEST_DAYS = 7
RAIN_THRESHOLD = 1.0
CITY_NAME = "扬州"

NUMERIC_VARS = {
    "temperature_2m_max":          "最高温(°C)",
    "temperature_2m_min":          "最低温(°C)",
    "temperature_2m_mean":        "平均温(°C)",
    "apparent_temperature_max":   "体感最高(°C)",
    "apparent_temperature_min":   "体感最低(°C)",
    "apparent_temperature_mean":  "体感平均(°C)",
    "precipitation_sum":          "降水量(mm)",
    "precipitation_hours":         "降水时数(h)",
    "wind_speed_10m_max":         "最大风速(km/h)",
    "wind_gusts_10m_max":         "最大阵风(km/h)",
    "shortwave_radiation_sum":    "日辐射(MJ/m²)",
    "surface_pressure_mean":       "平均气压(hPa)",
    "cloud_cover_mean":           "平均云量(%)",
    "relative_humidity_2m_mean":   "平均湿度(%)",
}
DAILY_FIELD = ",".join(NUMERIC_VARS.keys()) + ",wind_direction_10m_dominant"
NWP_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
    "shortwave_radiation_sum", "surface_pressure_mean", "cloud_cover_mean",
    "relative_humidity_2m_mean",
]
NWP_SOURCES = {
    "M5_GFS": "gfs_seamless",
    "M6_ECMWF": "ecmwf_ifs025",
}


# =============================================================
# 1. 数据
# =============================================================
def fetch_history(years: int) -> pd.DataFrame:
    end = (datetime.now() - timedelta(days=2)).date()
    start = end - timedelta(days=365 * years)
    print(f"[1/7] 拉历史观测 {start} ~ {end} ...")
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": LAT, "longitude": LON,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": DAILY_FIELD, "timezone": "Asia/Shanghai",
    }, timeout=90)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({"ds": pd.to_datetime(d["time"]), **{k: d[k] for k in NUMERIC_VARS}})
    df["wind_dir"] = d.get("wind_direction_10m_dominant")
    df = df.dropna(subset=list(NUMERIC_VARS.keys())).reset_index(drop=True)
    df["doy"] = df["ds"].dt.dayofyear
    print(f"      {len(df)} 天历史观测")
    return df


def fetch_nwp(model_id: str) -> pd.DataFrame:
    """拉某个 NWP 模型对过去 NWP_BACKTEST_DAYS 天 + 未来 FORECAST_DAYS 天的预报。"""
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": LAT, "longitude": LON,
        "daily": ",".join(NWP_VARS),
        "models": model_id,
        "past_days": NWP_BACKTEST_DAYS, "forecast_days": FORECAST_DAYS,
        "timezone": "Asia/Shanghai",
    }, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame({"ds": pd.to_datetime(d["time"]), **{k: d[k] for k in NWP_VARS}})


def fetch_hourly(hours: int = 48) -> pd.DataFrame:
    """拉未来 hours 小时逐小时预报（GFS/ECMWF 均值），供前端逐小时曲线。"""
    print(f"      拉逐小时预报（未来 {hours} 小时）...")
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,apparent_temperature,precipitation_probability,"
                  "precipitation,weathercode,wind_speed_10m,relative_humidity_2m,cloud_cover",
        "forecast_days": max(2, hours // 24 + 1),
        "timezone": "Asia/Shanghai",
    }, timeout=60)
    r.raise_for_status()
    d = r.json()["hourly"]
    df = pd.DataFrame({
        "时间": pd.to_datetime(d["time"]),
        "温度(°C)": d["temperature_2m"],
        "体感(°C)": d["apparent_temperature"],
        "降水概率(%)": d["precipitation_probability"],
        "降水量(mm)": d["precipitation"],
        "天气代码": d["weathercode"],
        "风速(km/h)": d["wind_speed_10m"],
        "湿度(%)": d["relative_humidity_2m"],
        "云量(%)": d["cloud_cover"],
    }).head(hours)
    return df


def fetch_warnings(province_kw: str = "江苏", city_kw: str = "扬州") -> list:
    """从中央气象台拉全国预警，过滤出目标省份/城市。失败返回空列表（不影响主流程）。"""
    try:
        r = requests.get("http://www.nmc.cn/rest/findAlarm", params={"type": "1"}, timeout=20,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        items = r.json()["data"]["page"]["list"]
        out = []
        for it in items:
            title = it.get("title", "")
            if province_kw in title or city_kw in title:
                out.append({
                    "title": title,
                    "time": it.get("issuetime", ""),
                    "url": "http://www.nmc.cn" + it.get("url", ""),
                    "pic": it.get("pic", ""),
                })
        return out
    except Exception as e:
        print(f"      预警拉取失败（忽略）: {e}")
        return []


# =============================================================
# 2. 四个统计模型
# =============================================================
def m1_clim_ewma(train_s, train_doys, future_doys, n):
    tmp = pd.DataFrame({"doy": train_doys.values, "y": train_s.values})
    clim = tmp.groupby("doy")["y"].mean()
    anom = tmp["y"].values - tmp["doy"].map(clim).values
    anom = pd.Series(anom).ewm(span=ANOMALY_WINDOW).mean().iloc[-1]
    return np.array([clim[d] + anom for d in future_doys])


def m2_fourier(train_s, train_doys, future_doys, n):
    from sklearn.linear_model import Ridge
    X, y = [], []
    for i, v in enumerate(train_s.values):
        doy = train_doys.values[i]
        row = [1.0, i]
        for k in (1, 2, 3):
            row += [np.sin(2*np.pi*k*doy/365.25), np.cos(2*np.pi*k*doy/365.25)]
        X.append(row); y.append(v)
    m = Ridge().fit(X, y)
    last_idx = len(train_s)
    out = []
    for j, doy in enumerate(future_doys, start=1):
        row = [1.0, last_idx + j]
        for k in (1, 2, 3):
            row += [np.sin(2*np.pi*k*doy/365.25), np.cos(2*np.pi*k*doy/365.25)]
        out.append(m.predict([row])[0])
    return np.array(out)


def m3_hw(train_s, n):
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        m = ExponentialSmoothing(
            train_s.values, trend="add", seasonal="add",
            seasonal_periods=365, initialization_method="estimated",
        ).fit(optimized=True)
        return np.array(m.forecast(n))
    except Exception:
        return np.full(n, train_s.iloc[-1])


def _build_row(df_hist, target_col, next_doy):
    s = df_hist[target_col].values
    x = []
    for lag in (1, 2, 3, 7, 14, 21):
        x.append(s[-lag] if len(s) >= lag else 0.0)
    for w in (ANOMALY_WINDOW,):
        x.append(np.nanmean(s[-w:])); x.append(np.nanstd(s[-w:]))
    for c in list(NUMERIC_VARS.keys())[:6]:
        x.append(df_hist[c].iloc[-1])
    for k in (1, 2, 3, 4):
        x += [np.sin(2*np.pi*k*next_doy/365.25), np.cos(2*np.pi*k*next_doy/365.25)]
    return np.nan_to_num(np.array(x, dtype=float), nan=0.0).reshape(1, -1)


def m4_xgb(df, target_col, n, start_date=None):
    import xgboost as xgb
    s = df[target_col].astype(float)
    X_rows, y_rows = [], []
    for i in range(30, len(df)):
        sub = df.iloc[:i]
        X_rows.append(_build_row(sub, target_col, sub["doy"].iloc[-1])[0])
        y_rows.append(s.iloc[i])
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2,
    ).fit(np.array(X_rows), np.array(y_rows))
    hist = df.copy(); preds = []; last_date = hist["ds"].iloc[-1]
    # 预测从 start_date 开始（回测时传 train 末+1，最终预测传今天）
    start = start_date if start_date is not None else (last_date + timedelta(days=1))
    for j in range(n):
        nd = start + timedelta(days=j); ndoy = nd.timetuple().tm_yday
        yhat = float(model.predict(_build_row(hist, target_col, ndoy))[0])
        preds.append(yhat)
        nr = {c: hist[c].iloc[-1] for c in NUMERIC_VARS}
        nr[target_col] = yhat; nr["ds"] = nd; nr["doy"] = ndoy
        hist = pd.concat([hist, pd.DataFrame([nr])], ignore_index=True)
    return np.array(preds)


# =============================================================
# 3. 单要素：跑所有模型
# =============================================================
def forecast_variable(df, nwps, col, n, start_date=None):
    s = df[col].astype(float).reset_index(drop=True)
    doys = df["doy"].reset_index(drop=True)
    # 预测第一天：默认 = 今天；回测时用 train 末+1
    if start_date is None:
        start_date = (df["ds"].iloc[-1] + timedelta(days=1)).date()
    future_doys = [(start_date + timedelta(days=i)).timetuple().tm_yday
                   for i in range(n)]

    cut = len(s) - BACKTEST_DAYS
    train_s, test_s = s.iloc[:cut], s.iloc[cut:].values
    train_doys = doys.iloc[:cut].reset_index(drop=True)
    test_doys = doys.iloc[cut:].reset_index(drop=True)

    models = {}
    try:
        p1t = m1_clim_ewma(train_s, train_doys, test_doys.tolist(), BACKTEST_DAYS)
        p1f = m1_clim_ewma(s, doys, future_doys, n)
        models["M1_EWMA"] = (p1t, p1f)
    except Exception: pass
    try:
        p2t = m2_fourier(train_s, train_doys, test_doys.tolist(), BACKTEST_DAYS)
        p2f = m2_fourier(s, doys, future_doys, n)
        models["M2_傅里叶"] = (p2t, p2f)
    except Exception: pass
    try:
        p3t = m3_hw(train_s, BACKTEST_DAYS)
        p3f = m3_hw(s, n)
        models["M3_HW"] = (p3t, p3f)
    except Exception: pass
    try:
        p4t = m4_xgb(df.iloc[:cut].copy(), col, BACKTEST_DAYS,
                     start_date=(df["ds"].iloc[cut] if cut < len(df) else None))
        p4f = m4_xgb(df, col, n, start_date=start_date)
        models["M4_XGB"] = (p4t, p4f)
    except Exception: pass

    # NWP：GFS + ECMWF
    for nwp_name, g in nwps.items():
        if col not in g.columns: continue
        past = g[g["ds"] <= df["ds"].iloc[-1]][col].dropna().tail(NWP_BACKTEST_DAYS).values
        fut = g[g["ds"] > df["ds"].iloc[-1]][col].dropna().tail(n).values
        if len(past) >= 5 and len(fut) >= n:
            obs_recent = s.tail(len(past)).values
            models[nwp_name] = (past, fut[:n])

    scored = {}
    for name, (pt, pf) in models.items():
        k = min(len(pt), len(test_s))
        if k < 5: continue
        mae = float(np.mean(np.abs(pt[-k:] - test_s[-k:])))
        scored[name] = (pf[:n], mae)
    return scored


def ensemble(scored):
    names = list(scored.keys())
    maes = np.array([scored[n][1] for n in names])
    weights = 1.0 / maes
    weights = weights / weights.sum()
    preds = np.vstack([scored[n][0] for n in names])
    final = (preds * weights[:, None]).sum(axis=0)
    return final, dict(zip(names, np.round(weights, 3)))


def forecast_rain_prob(df, n, start_date=None):
    has_rain = (df["precipitation_sum"] > RAIN_THRESHOLD).astype(int)
    tmp = pd.DataFrame({"doy": df["doy"], "h": has_rain})
    clim = tmp.groupby("doy")["h"].mean()
    s = pd.concat([clim.iloc[-7:], clim, clim.iloc[:7]])
    clim = s.rolling(15, center=True).mean().iloc[7:-7]
    bias = has_rain.tail(ANOMALY_WINDOW).mean() - clim.loc[df.tail(ANOMALY_WINDOW)["doy"]].mean()
    out = []
    if start_date is None:
        start_date = (df["ds"].iloc[-1] + timedelta(days=1)).date()
    for i in range(n):
        d = (start_date + timedelta(days=i)).timetuple().tm_yday
        out.append(np.clip(clim[d] + bias, 0.02, 0.98))
    return np.array(out)


def wind_dir_label(deg):
    dirs = ["北","东北偏北","东北","东北偏东","东","东南偏东","东南","东南偏南",
            "南","西南偏南","西南","西南偏西","西","西北偏西","西北","西北偏北"]
    return dirs[int((deg % 360)/22.5 + 0.5) % 16]


def main():
    df = fetch_history(HISTORY_YEARS)

    print(f"[2/7] 拉 NWP 数值预报（GFS + ECMWF，回看{NWP_BACKTEST_DAYS}天 + 未来{FORECAST_DAYS}天）...")
    nwps = {}
    for name, mid in NWP_SOURCES.items():
        try:
            nwps[name] = fetch_nwp(mid)
            print(f"      {name} OK")
        except Exception as e:
            print(f"      {name} 失败: {e}")

    # 预报锚定今天：未来 N 天 = 今天 ~ 今天+N-1（而非从历史末+1）
    start_date = datetime.now().date()

    # 逐小时预报（GFS/ECMWF 均值，48 小时）
    try:
        hourly = fetch_hourly(48)
        hourly["时间"] = hourly["时间"].dt.strftime("%Y-%m-%d %H:%M")   # 带年份，前端可锁定当前时段
        hourly.to_csv("hourly_forecast.csv", index=False, encoding="utf-8-sig")
        print(f"      已保存 hourly_forecast.csv（{len(hourly)} 小时）")
    except Exception as e:
        print(f"      逐小时拉取失败（忽略）: {e}")

    # 突发天气预警（中央气象台）
    try:
        import json
        warnings = fetch_warnings(province_kw="江苏", city_kw="扬州")
        with open("warning.json", "w", encoding="utf-8") as f:
            json.dump(warnings, f, ensure_ascii=False, indent=2)
        print(f"      已保存 warning.json（{len(warnings)} 条预警）")
    except Exception as e:
        print(f"      预警保存失败（忽略）: {e}")

    dates = [start_date + timedelta(days=i) for i in range(FORECAST_DAYS)]
    result = pd.DataFrame({"日期": dates})
    log = []

    print("[3/7] 逐要素跑 6 模型 + 集成 ...")
    for api_col, out_col in NUMERIC_VARS.items():
        scored = forecast_variable(df, nwps, api_col, FORECAST_DAYS, start_date)
        final, weights = ensemble(scored)
        result[out_col] = np.round(final, 1)
        wstr = "  ".join(f"{k}:{v}" for k, v in weights.items())
        log.append(f"  {out_col:14s}  {wstr}")

    result["降水概率"] = [f"{int(p)}%" for p in (forecast_rain_prob(df, FORECAST_DAYS, start_date)*100).round(0)]
    wdir = df.groupby("doy")["wind_dir"].mean()
    future_doys = [(start_date + timedelta(days=i)).timetuple().tm_yday for i in range(FORECAST_DAYS)]
    result["主导风向"] = [wind_dir_label(wdir[d]) for d in future_doys]
    result["昼夜温差(°C)"] = (result["最高温(°C)"] - result["最低温(°C)"]).round(1)
    result["日期"] = pd.to_datetime(result["日期"]).dt.date

    print("\n[4/7] 各模型集成权重：")
    print("\n".join(log))
    print("\n[5/7] 未来 7 天预报：")
    print(result.to_string(index=False))
    result.to_csv("yangzhou_forecast.csv", index=False, encoding="utf-8-sig")

    print("[6/7] 画图 ...")
    recent60 = df[df["ds"] >= datetime.now() - timedelta(days=60)]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(recent60["ds"], recent60["temperature_2m_mean"], "-", color="#888", label="历史平均温(近60天)")
    ax.fill_between(recent60["ds"], recent60["temperature_2m_min"], recent60["temperature_2m_max"],
                    color="#ccc", alpha=0.4, label="历史最低~最高")
    fd = pd.to_datetime(result["日期"])
    ax.plot(fd, result["平均温(°C)"], "o-", color="#d62728", label="集成预报平均温")
    ax.plot(fd, result["最高温(°C)"], "s--", color="#ff7f0e", label="集成预报最高温")
    ax.plot(fd, result["最低温(°C)"], "^--", color="#1f77b4", label="集成预报最低温")
    ax.set_title(f"{CITY_NAME}未来{FORECAST_DAYS}天·气温预报(双NWP+集成)")
    ax.set_ylabel("°C"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("chart_temperature.png", dpi=130)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0,0].bar(fd, result["降水量(mm)"], color="#1f77b4", alpha=0.7)
    axes[0,0].set_title("预报降水量(mm)"); axes[0,0].grid(alpha=0.3)
    axes[0,1].bar(fd, result["平均湿度(%)"], color="#2ca02c", alpha=0.7)
    axes[0,1].set_title("预报平均湿度(%)"); axes[0,1].grid(alpha=0.3)
    axes[1,0].plot(fd, result["最大风速(km/h)"], "o-", label="风速", color="#ff7f0e")
    axes[1,0].plot(fd, result["最大阵风(km/h)"], "s--", label="阵风", color="#d62728")
    axes[1,0].set_title("预报风速/阵风(km/h)"); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)
    axes[1,1].plot(fd, result["平均气压(hPa)"], "o-", color="#9467bd")
    axes[1,1].set_title("预报平均气压(hPa)"); axes[1,1].grid(alpha=0.3)
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.suptitle(f"{CITY_NAME}未来{FORECAST_DAYS}天·其他要素预报", fontsize=13)
    plt.tight_layout(); plt.savefig("chart_others.png", dpi=130)
    print("\n[7/7] 已保存：yangzhou_forecast.csv, hourly_forecast.csv, warning.json, chart_temperature.png, chart_others.png")


if __name__ == "__main__":
    main()
