from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Prophet Forecast App", layout="wide")
st.title("📈 E-commerce Book Sales Data Forecast")

MODELS_DIR = Path("models")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_artifact(path: Path):
    return joblib.load(path)

@st.cache_resource
def load_all_artifacts(models_dir: Path):
    """
    Load all joblibs once and build an index:
      index[category][target] -> {"path": Path, "meta": dict, "artifact": dict}
    """
    index = {}
    for p in sorted(models_dir.glob("model__*.joblib")):
        art = load_artifact(p)
        meta = art.get("meta", {})
        cat = meta.get("category", None)
        tgt = meta.get("target", None)
        if not cat or not tgt:
            continue
        index.setdefault(cat, {})[tgt] = {"path": p, "meta": meta, "artifact": art}
    return index

def safe_name(s: str) -> str:
    return (
        str(s).strip().lower()
        .replace("&", "and")
        .replace(" ", "_")
        .replace("__", "_")
    )

def weekly_from_daily(ds, y, week_rule="W-SUN"):
    ds = pd.to_datetime(ds)
    y = pd.to_numeric(y, errors="coerce")
    wk = ds.dt.to_period(week_rule).dt.end_time.dt.normalize()
    s = pd.Series(y.to_numpy(), index=wk).groupby(level=0).sum(min_count=1).sort_index()
    return s

def format_for_display(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Display-friendly:
      - qty -> rounded int
      - revenue -> $ with 2 decimals
      - rename columns to readable names
      - remove 00:00:00 from ds/week_end (show date only)
    """
    df2 = df.copy()

    if "ds" in df2.columns:
        df2["ds"] = pd.to_datetime(df2["ds"]).dt.date
    if "week_end" in df2.columns:
        df2["week_end"] = pd.to_datetime(df2["week_end"]).dt.date

    rename_map = {
        "yhat": "Forecast",
        "yhat_lower": "Lower bound (80%)",
        "yhat_upper": "Upper bound (80%)",
        "yhat_weekly": "Weekly forecast",
    }
    df2 = df2.rename(columns={k: v for k, v in rename_map.items() if k in df2.columns})

    if target == "revenue":
        money_cols = [c for c in ["Forecast", "Lower bound (80%)", "Upper bound (80%)", "Weekly forecast"] if c in df2.columns]
        for c in money_cols:
            df2[c] = df2[c].astype(float).round(2)
            df2[c] = df2[c].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    else:
        num_cols = [c for c in ["Forecast", "Lower bound (80%)", "Upper bound (80%)", "Weekly forecast"] if c in df2.columns]
        for c in num_cols:
            df2[c] = df2[c].astype(float).round(0).astype("Int64")

    return df2

def format_for_download(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Download-friendly (numeric, rounded):
      - qty -> nearest int
      - revenue -> 2 decimals
      - rename columns to readable names (snake_case)
    """
    df2 = df.copy()
    rename_map = {
        "yhat": "forecast",
        "yhat_lower": "lower_bound_80",
        "yhat_upper": "upper_bound_80",
        "yhat_weekly": "weekly_forecast",
    }
    df2 = df2.rename(columns={k: v for k, v in rename_map.items() if k in df2.columns})

    if target == "revenue":
        for c in ["forecast", "lower_bound_80", "upper_bound_80", "weekly_forecast"]:
            if c in df2.columns:
                df2[c] = df2[c].astype(float).round(2)
    else:
        for c in ["forecast", "lower_bound_80", "upper_bound_80", "weekly_forecast"]:
            if c in df2.columns:
                df2[c] = df2[c].astype(float).round(0).astype("Int64")

    return df2

# -----------------------------
# Load model index (category x target)
# -----------------------------
index = load_all_artifacts(MODELS_DIR)
if not index:
    st.error("No saved models found in ./models (expected files like model__<cat>__<tgt>.joblib).")
    st.stop()

categories = sorted(index.keys())

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Model selection")
cat = st.sidebar.selectbox("Category", categories, index=0)
targets = sorted(index[cat].keys())
tgt = st.sidebar.selectbox("Target", targets, index=0)

st.sidebar.header("Forecast controls")
forecast_years = st.sidebar.number_input("Forecast years", min_value=1, max_value=10, value=3, step=1)
include_history = st.sidebar.checkbox("Include history", value=True)
show_intervals = st.sidebar.checkbox("Show uncertainty bounds", value=True)
show_weekly = st.sidebar.checkbox("Also show weekly aggregation", value=True)

# -----------------------------
# Load selected model
# -----------------------------
selected = index[cat][tgt]
artifact = selected["artifact"]
meta = selected["meta"]
m = artifact["model"]

cap_clip = meta.get("cap_clip", None)
if isinstance(cap_clip, (int, float)) and cap_clip is not None:
    cap_clip_str = f"${cap_clip:,.2f}" if tgt == "revenue" else f"{cap_clip:,.0f}"
else:
    cap_clip_str = str(cap_clip)

st.subheader("By Aini Nur Zahiyah Maulani")

st.caption(
    f"**Category:** {meta.get('category','?')}  •  "
    f"**Target:** {meta.get('target','?')}  •  "
    f"**Mode:** {meta.get('seasonality_mode')}  •  "
    f"**Logistic:** {meta.get('use_logistic')}  •  "
    f"**Winsor:** q={meta.get('winsor_q')} cap={cap_clip_str}  •  "
    f"**Semester window:** {meta.get('sem_window_days')} days  •  "
    f"**Monthly:** {meta.get('use_monthly')}  •  "
    f"**Semester seasonality:** {meta.get('use_semester')}"
)

# -----------------------------
# Forecast
# -----------------------------
last_ds = pd.to_datetime(m.history["ds"]).max()
end_ds = last_ds + pd.DateOffset(years=int(forecast_years))
periods = int((end_ds.normalize() - last_ds.normalize()).days)

future = m.make_future_dataframe(periods=periods, freq="D", include_history=include_history)

if bool(meta.get("use_logistic", False)):
    floor_log = meta.get("floor_log", None)
    cap_log = meta.get("cap_log", None)
    if floor_log is None or cap_log is None:
        st.error("This model uses logistic growth but floor_log/cap_log are missing in meta.")
        st.stop()
    future["floor"] = float(floor_log)
    future["cap"] = float(cap_log)

pred = m.predict(future)

out = pd.DataFrame({"ds": pred["ds"]})
for col in ["yhat", "yhat_lower", "yhat_upper"]:
    if col in pred.columns:
        vals = pred[col].to_numpy(dtype=float)
        if meta.get("needs_expm1", True):
            vals = np.expm1(vals)
        out[col] = np.clip(vals, 0.0, None)

# -----------------------------
# Daily display
# -----------------------------
st.subheader("Daily forecast")

cols_to_show = ["ds", "yhat"]
if show_intervals and "yhat_lower" in out.columns and "yhat_upper" in out.columns:
    cols_to_show += ["yhat_lower", "yhat_upper"]

daily_display = format_for_display(out[cols_to_show], target=tgt)
st.dataframe(daily_display.tail(120), use_container_width=True)

st.line_chart(out.set_index("ds")[["yhat"]])

# -----------------------------
# Weekly aggregation
# -----------------------------
wk_df = None
if show_weekly:
    st.subheader("Weekly forecast (sum of daily)")

    week_rule = meta.get("week_rule", "W-SUN")
    wk = weekly_from_daily(out["ds"], out["yhat"], week_rule=week_rule)

    wk_df = wk.reset_index()
    wk_df.columns = ["week_end", "yhat_weekly"]

    weekly_display = format_for_display(wk_df, target=tgt)
    st.dataframe(weekly_display.tail(120), use_container_width=True)

    st.line_chart(wk_df.set_index("week_end")[["yhat_weekly"]])

# -----------------------------
# Downloads
# -----------------------------
daily_download = format_for_download(out[cols_to_show], target=tgt)
st.download_button(
    "Download daily forecast CSV",
    data=daily_download.to_csv(index=False).encode("utf-8"),
    file_name=f"forecast_daily_{safe_name(meta.get('category','cat'))}_{safe_name(meta.get('target','tgt'))}.csv",
    mime="text/csv"
)

if show_weekly and wk_df is not None:
    weekly_download = format_for_download(wk_df, target=tgt)
    st.download_button(
        "Download weekly forecast CSV",
        data=weekly_download.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_weekly_{safe_name(meta.get('category','cat'))}_{safe_name(meta.get('target','tgt'))}.csv",
        mime="text/csv"
)
