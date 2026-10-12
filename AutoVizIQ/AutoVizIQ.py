from __future__ import annotations

import io
import json
import math
import re
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)

# Optional advanced analytics dependencies. The core dashboard still works if absent.
try:
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

APP_NAME = "AutoVizIQ"
APP_SUBTITLE = "Automated data exploration • quality intelligence • statistical insights"
MAX_CHART_CATEGORIES = 25
MAX_SCATTER_POINTS = 6000
MAX_HEATMAP_COLUMNS = 35


# ------------------------------- Page setup ----------------------------------
st.set_page_config(
    page_title="AutoVizIQ | Intelligent Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1600px;}
    .hero {
      padding: 1.35rem 1.6rem; border-radius: 18px;
      background: linear-gradient(120deg, rgba(42,89,195,.17), rgba(19,166,146,.10));
      border: 1px solid rgba(120,140,170,.28); margin-bottom: 1.1rem;
    }
    .hero h1 {margin: 0; font-size: 2.25rem; letter-spacing: -.04em;}
    .hero p {margin: .4rem 0 0 0; opacity: .78; font-size: 1rem;}
    .insight {
      padding: 1rem 1.05rem; border: 1px solid rgba(120,140,170,.28);
      border-radius: 14px; min-height: 125px; margin-bottom: .5rem;
      background: rgba(128,128,128,.035);
    }
    .insight h4 {margin: 0 0 .45rem 0; font-size: 1rem;}
    .insight p {margin: 0; line-height: 1.45; font-size: .92rem;}
    .muted {opacity: .72; font-size: .88rem;}
    div[data-testid="stMetric"] {
      border: 1px solid rgba(120,140,170,.25); padding: .85rem 1rem;
      border-radius: 13px; background: rgba(128,128,128,.035);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------- Utilities ----------------------------------
def safe_name(value: Any) -> str:
    """Convert arbitrary values to readable labels without throwing."""
    if pd.isna(value):
        return "(Missing)"
    text = str(value)
    return text if len(text) <= 70 else text[:67] + "..."


def bytes_to_text(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024**2):.2f} MB"


def read_uploaded_file(uploaded_file, csv_separator: str = "Auto", encoding: str = "utf-8") -> pd.DataFrame:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".csv") or name.endswith(".txt"):
        sep = None if csv_separator == "Auto" else csv_separator
        try:
            return pd.read_csv(io.BytesIO(raw), sep=sep, engine="python" if sep is None else "c", encoding=encoding)
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), sep=sep, engine="python" if sep is None else "c", encoding="latin-1")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    if name.endswith(".json"):
        try:
            return pd.read_json(io.BytesIO(raw))
        except ValueError:
            return pd.read_json(io.BytesIO(raw), lines=True)
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))
    raise ValueError("Unsupported file type. Upload CSV, Excel, JSON, or Parquet.")


def make_demo_data(rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic retail dataset for the demo mode."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    regions = rng.choice(["West", "South", "North", "East", "Central"], rows, p=[.28, .23, .2, .17, .12])
    channels = rng.choice(["Online", "Store", "Marketplace"], rows, p=[.48, .34, .18])
    categories = rng.choice(["Electronics", "Home", "Fashion", "Beauty", "Grocery"], rows)
    quantity = rng.integers(1, 9, rows)
    price_base = pd.Series(categories).map({
        "Electronics": 7200, "Home": 2100, "Fashion": 1300,
        "Beauty": 750, "Grocery": 260
    }).to_numpy()
    unit_price = np.maximum(50, rng.normal(price_base, price_base * .22))
    discount = np.clip(rng.beta(2, 8, rows) * .65, 0, .55)
    seasonal = 1 + .12 * np.sin(np.arange(rows) * 2 * np.pi / 365)
    revenue = quantity * unit_price * (1 - discount) * seasonal
    cost_ratio = rng.uniform(.48, .82, rows)
    profit = revenue * (1 - cost_ratio) - revenue * .035
    customer_age = np.clip(rng.normal(34, 11, rows), 18, 78).round()
    rating = np.clip(rng.normal(4.05 - discount * 0.7, .55, rows), 1, 5).round(1)
    df = pd.DataFrame({
        "Order_Date": dates,
        "Region": regions,
        "Sales_Channel": channels,
        "Product_Category": categories,
        "Quantity": quantity,
        "Unit_Price_INR": unit_price.round(2),
        "Discount_Rate": discount.round(3),
        "Revenue_INR": revenue.round(2),
        "Profit_INR": profit.round(2),
        "Customer_Age": customer_age,
        "Customer_Rating": rating,
        "Customer_ID": [f"C{n:05d}" for n in rng.integers(1, 360, rows)],
    })
    # Intentional realistic data-quality issues, disclosed in demo mode.
    for col, fraction in [("Customer_Rating", .035), ("Discount_Rate", .018), ("Region", .012)]:
        idx = rng.choice(rows, max(1, int(rows * fraction)), replace=False)
        df.loc[idx, col] = np.nan
    if rows > 300:
        df.loc[rng.choice(rows, 3, replace=False), "Revenue_INR"] *= 7
    return df


def infer_and_convert_types(df: pd.DataFrame, date_threshold: float = 0.82) -> tuple[pd.DataFrame, list[str]]:
    """Conservatively infer date-like object columns; return a copy and conversion notes."""
    out = df.copy()
    notes = []
    for col in out.select_dtypes(include=["object", "string"]).columns:
        sample = out[col].dropna()
        if len(sample) < 3:
            continue
        # Avoid converting ID-like columns just because they contain digits.
        id_like = bool(re.search(r"(^id$|_id$|id_|code|zip|postal|phone|account)", str(col), re.I))
        if id_like:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        success_rate = float(parsed.notna().mean()) if len(sample) else 0
        if success_rate >= date_threshold:
            out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")
            notes.append(f"Detected date/time column: **{col}** ({success_rate:.0%} parse success).")
    return out, notes


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric + datetime_cols + bool_cols]
    numeric = [c for c in numeric if df[c].nunique(dropna=True) > 1]
    categorical = [c for c in categorical if df[c].nunique(dropna=True) > 0]
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime_cols, "boolean": bool_cols}


def quality_report(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    unique = df.nunique(dropna=True)
    dtypes = df.dtypes.astype(str)
    rows = []
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        rows.append({
            "Column": col,
            "Type": dtypes[col],
            "Missing": int(missing[col]),
            "Missing %": round(float(missing[col] / max(n_rows, 1) * 100), 2),
            "Unique": int(unique[col]),
            "Unique %": round(float(unique[col] / max(non_null, 1) * 100), 2),
            "Duplicate values": int(non_null - unique[col]),
            "Constant": bool(unique[col] <= 1),
        })
    profile = pd.DataFrame(rows).sort_values(["Missing %", "Unique"], ascending=[False, True])
    duplicate_rows = int(df.duplicated().sum())
    summary = {
        "rows": n_rows,
        "columns": n_cols,
        "missing_cells": int(missing.sum()),
        "missing_pct": float(missing.sum() / max(n_rows * n_cols, 1) * 100),
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": float(duplicate_rows / max(n_rows, 1) * 100),
        "constant_columns": int((unique <= 1).sum()),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
    }
    return profile, summary


def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame()
    desc = df[cols].describe(percentiles=[.05, .25, .5, .75, .95]).T
    desc["missing"] = df[cols].isna().sum()
    desc["skewness"] = df[cols].skew(numeric_only=True)
    desc["kurtosis"] = df[cols].kurtosis(numeric_only=True)
    desc["zero_count"] = (df[cols] == 0).sum()
    desc["negative_count"] = (df[cols] < 0).sum()
    return desc.reset_index(names="column")


def outlier_bounds(series: pd.Series, method: str = "IQR", multiplier: float = 1.5) -> tuple[float, float]:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return (float("nan"), float("nan"))
    if method == "Z-score":
        mean, std = x.mean(), x.std(ddof=0)
        if std == 0 or pd.isna(std):
            return (float("-inf"), float("inf"))
        return (float(mean - 3 * std), float(mean + 3 * std))
    q1, q3 = x.quantile(.25), x.quantile(.75)
    iqr = q3 - q1
    return (float(q1 - multiplier * iqr), float(q3 + multiplier * iqr))


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    for col in df.columns:
        s = df[col]
        sample = ", ".join(s.dropna().astype(str).unique()[:3])
        result.append({
            "Column": col,
            "Inferred type": str(s.dtype),
            "Non-null": int(s.notna().sum()),
            "Missing": int(s.isna().sum()),
            "Distinct": int(s.nunique(dropna=True)),
            "Example values": sample[:160],
        })
    return pd.DataFrame(result)


def generate_insights(df: pd.DataFrame, classes: dict[str, list[str]], profile: pd.DataFrame) -> list[dict[str, str]]:
    """Rule-based, auditable insights. These are associations, not causal conclusions."""
    findings: list[dict[str, str]] = []
    if df.empty:
        return findings
    missing_total = int(df.isna().sum().sum())
    if missing_total:
        worst = profile.sort_values("Missing %", ascending=False).iloc[0]
        findings.append({
            "level": "Data quality",
            "title": "Missing data detected",
            "body": f"{missing_total:,} cells are missing. **{worst['Column']}** has the highest missing rate ({worst['Missing %']:.1f}%). Check whether the missingness is systematic before filling values.",
        })
    dupes = int(df.duplicated().sum())
    if dupes:
        findings.append({
            "level": "Data quality",
            "title": "Potential duplicate records",
            "body": f"{dupes:,} exact duplicate rows ({dupes / max(len(df), 1):.1%} of rows) were found. Review whether they are repeated transactions or genuine duplicates before removal.",
        })
    for col in classes["numeric"]:
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(x) >= 10 and x.nunique() > 1:
            low, high = outlier_bounds(x)
            count = int(((x < low) | (x > high)).sum())
            pct = count / len(x) * 100
            if pct >= 3:
                findings.append({
                    "level": "Distribution",
                    "title": f"Unusual values in {col}",
                    "body": f"The IQR rule flags {count:,} values ({pct:.1f}%) as potential outliers. They may be errors or meaningful rare events; inspect them before deciding.",
                })
                if len(findings) >= 8:
                    break
    if classes["numeric"]:
        nums = df[classes["numeric"]].replace([np.inf, -np.inf], np.nan)
        corr = nums.corr(min_periods=5)
        pairs = []
        for i, c1 in enumerate(corr.columns):
            for c2 in corr.columns[i + 1:]:
                val = corr.loc[c1, c2]
                if pd.notna(val):
                    pairs.append((abs(float(val)), float(val), c1, c2))
        if pairs:
            strength, val, c1, c2 = max(pairs)
            if strength >= .65:
                findings.append({
                    "level": "Relationship",
                    "title": "Strong numeric association",
                    "body": f"**{c1}** and **{c2}** have a Pearson correlation of {val:.2f}. This indicates a linear association, not causation; check for shared formulas, confounding, or duplicated information.",
                })
    if classes["categorical"]:
        for col in classes["categorical"][:8]:
            counts = df[col].fillna("(Missing)").astype(str).value_counts()
            if len(counts) >= 2:
                share = counts.iloc[0] / max(counts.sum(), 1)
                if share >= .75:
                    findings.append({
                        "level": "Category balance",
                        "title": f"{col} is dominated by one value",
                        "body": f"**{counts.index[0]}** represents {share:.0%} of non-missing records in **{col}**. If this variable is used for modeling or comparisons, class imbalance may matter.",
                    })
                    break
    if not findings:
        findings.append({
            "level": "Overview",
            "title": "No high-priority rule-based warning found",
            "body": "The quick checks did not detect a major issue using the current thresholds. This does not prove the dataset is error-free; validate column meaning and business rules.",
        })
    return findings[:10]


def style_chart(fig, title: str | None = None):
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=15, r=15, t=60 if title else 25, b=15),
        font=dict(family="Arial, sans-serif", size=12),
        legend_title_text="",
    )
    return fig


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def to_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False).encode("utf-8")


def html_report(df: pd.DataFrame, quality: dict, insights: list[dict[str, str]], numeric_desc: pd.DataFrame) -> bytes:
    """Create a standalone HTML report without external dependencies."""
    cards = "".join(
        f"<section><h3>{item['title']}</h3><p>{item['body']}</p><small>{item['level']}</small></section>"
        for item in insights
    )
    metrics = "".join(
        f"<div class='metric'><b>{k.replace('_', ' ').title()}</b><span>{v:,.2f}</span></div>"
        for k, v in quality.items() if isinstance(v, (int, float))
    )
    table = numeric_desc.to_html(index=False, border=0) if not numeric_desc.empty else "<p>No numeric columns available.</p>"
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>AutoVizIQ Report</title>
    <style>body{{font-family:Arial,sans-serif;margin:35px;color:#172033}}h1{{margin-bottom:4px}}
    .sub{{color:#667085}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}
    .metric,section{{border:1px solid #d8deea;border-radius:10px;padding:14px;margin:8px 0}}
    .metric span{{display:block;font-size:1.3rem;margin-top:8px}}table{{border-collapse:collapse;width:100%;font-size:12px}}
    th,td{{border-bottom:1px solid #ddd;padding:7px;text-align:left}}section p{{line-height:1.5}}
    </style></head><body><h1>AutoVizIQ — Dataset Analysis Report</h1>
    <p class='sub'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} • {len(df):,} rows × {len(df.columns):,} columns</p>
    <h2>Dataset overview</h2><div class='metrics'>{metrics}</div>
    <h2>Automated findings</h2>{cards}<h2>Numeric summary</h2>{table}
    <p class='sub'>Findings are heuristic and descriptive. Correlation does not establish causation. Validate results with domain knowledge.</p>
    </body></html>"""
    return html.encode("utf-8")


# ------------------------------- Sidebar -------------------------------------
st.sidebar.markdown("## 📊 AutoVizIQ")
st.sidebar.caption("Automated exploratory data analysis")
st.sidebar.divider()

uploaded = st.sidebar.file_uploader(
    "Upload a dataset",
    type=["csv", "txt", "xlsx", "xls", "json", "parquet"],
    help="Supported: CSV, TXT, Excel, JSON, Parquet",
)
with st.sidebar.expander("CSV import options"):
    csv_separator = st.selectbox("Delimiter", ["Auto", ",", ";", "\\t", "|"], index=0)
    encoding = st.selectbox("Text encoding", ["utf-8", "latin-1", "utf-16"], index=0)

use_demo = st.sidebar.toggle("Use built-in retail demo dataset", value=uploaded is None)
demo_rows = st.sidebar.slider("Demo dataset size", 300, 5000, 1200, step=100, disabled=not use_demo)
st.sidebar.divider()
st.sidebar.markdown("**Analysis settings**")
sample_limit = st.sidebar.select_slider("Chart sample limit", options=[1000, 2500, 5000, 10000, 25000], value=5000)
correlation_method = st.sidebar.selectbox("Correlation method", ["Pearson", "Spearman", "Kendall"], index=0)
outlier_method = st.sidebar.selectbox("Outlier detection rule", ["IQR", "Z-score"], index=0)
show_advanced = st.sidebar.toggle("Enable advanced analytics", value=True)
st.sidebar.caption("Tip: Upload your own CSV/Excel file to analyze your project data.")


# ------------------------------- Load data -----------------------------------
st.markdown(
    f"""<div class="hero"><h1>📊 {APP_NAME}</h1>
    <p>{APP_SUBTITLE}</p></div>""",
    unsafe_allow_html=True,
)

if uploaded is not None:
    try:
        raw_df = read_uploaded_file(uploaded, csv_separator, encoding)
        source_label = uploaded.name
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
        st.stop()
elif use_demo:
    raw_df = make_demo_data(demo_rows)
    source_label = "Built-in synthetic retail dataset"
else:
    st.info("Upload a dataset from the sidebar or enable the demo dataset to begin.")
    st.stop()

if raw_df.empty:
    st.error("The loaded dataset has no rows. Please upload a non-empty file.")
    st.stop()

# Normalize column labels but preserve original names as much as possible.
df_original = raw_df.copy()
if len(set(map(str, raw_df.columns))) != len(raw_df.columns):
    st.warning("Duplicate column labels detected. Pandas may have renamed some labels while importing.")
df_inferred, type_notes = infer_and_convert_types(raw_df)
classes_original = classify_columns(df_inferred)

# Cleaning choices are applied to an analysis copy only. Original upload remains available.
with st.sidebar.expander("Data preparation", expanded=False):
    drop_exact_duplicates = st.checkbox("Exclude exact duplicate rows", value=False)
    drop_empty_columns = st.checkbox("Exclude completely empty columns", value=True)
    strip_text = st.checkbox("Trim spaces in text columns", value=True)
    convert_numeric_text = st.checkbox("Convert numeric-looking text", value=False)
    missing_display = st.selectbox("Missing values in charts", ["Keep as missing", "Show as (Missing)"], index=0)

df = df_inferred.copy()
prep_log = []
if strip_text:
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    prep_log.append("Trimmed leading/trailing whitespace in text columns.")
if convert_numeric_text:
    for col in df.select_dtypes(include=["object", "string"]).columns:
        converted = pd.to_numeric(df[col].astype("string").str.replace(",", "", regex=False), errors="coerce")
        nonmissing = df[col].notna().sum()
        if nonmissing and converted.notna().sum() / nonmissing >= .9:
            df[col] = converted
            prep_log.append(f"Converted numeric-like text column: {col}.")
if drop_empty_columns:
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        df = df.drop(columns=empty_cols)
        prep_log.append(f"Excluded {len(empty_cols)} completely empty column(s).")
if drop_exact_duplicates:
    before = len(df)
    df = df.drop_duplicates().copy()
    prep_log.append(f"Excluded {before - len(df):,} exact duplicate rows.")
if df.empty or len(df.columns) == 0:
    st.error("Preparation settings removed all usable data. Change the settings in the sidebar.")
    st.stop()

classes = classify_columns(df)
profile, quality = quality_report(df)
insights = generate_insights(df, classes, profile)
num_desc = numeric_summary(df, classes["numeric"])
chart_df = df if len(df) <= sample_limit else df.sample(sample_limit, random_state=42)

# ------------------------------- Top overview -------------------------------
st.caption(f"**Source:** {source_label}  ·  **Analysis size:** {len(df):,} rows × {len(df.columns):,} columns  ·  **Memory:** {bytes_to_text(int(df.memory_usage(deep=True).sum()))}")
if type_notes:
    with st.expander("Automatic type detection notes"):
        for note in type_notes:
            st.markdown(f"- {note}")
if prep_log:
    with st.expander("Preparation steps applied to the analysis copy"):
        for item in prep_log:
            st.markdown(f"- {item}")
        st.caption("The source DataFrame is not overwritten. Export the prepared data explicitly if you want to keep these changes.")

metric_cols = st.columns(5)
metric_cols[0].metric("Rows", f"{len(df):,}")
metric_cols[1].metric("Columns", f"{len(df.columns):,}")
metric_cols[2].metric("Missing cells", f"{quality['missing_cells']:,}", f"{quality['missing_pct']:.1f}% of cells")
metric_cols[3].metric("Duplicate rows", f"{quality['duplicate_rows']:,}", f"{quality['duplicate_pct']:.1f}% of rows")
metric_cols[4].metric("Numeric features", f"{len(classes['numeric']):,}")

tab_overview, tab_quality, tab_univariate, tab_relationships, tab_time, tab_advanced, tab_clean, tab_export = st.tabs([
    "🏠 Overview", "🧹 Data quality", "📈 Distributions", "🔗 Relationships",
    "📅 Time analysis", "🧠 Advanced analytics", "🛠️ Cleaning workbench", "⬇️ Export"
])


# -------------------------------- Overview -----------------------------------
with tab_overview:
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Dataset preview")
        st.dataframe(df.head(12), use_container_width=True, hide_index=True)
        st.subheader("Column data dictionary")
        st.dataframe(build_data_dictionary(df), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Column type composition")
        type_counts = pd.Series({
            "Numeric": len(classes["numeric"]),
            "Categorical": len(classes["categorical"]),
            "Date/time": len(classes["datetime"]),
            "Boolean": len(classes["boolean"]),
        })
        fig = px.pie(values=type_counts.values, names=type_counts.index, hole=.55)
        fig.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(style_chart(fig), use_container_width=True)
        st.subheader("Automated findings")
        for item in insights[:5]:
            st.markdown(
                f"""<div class="insight"><h4>{item['title']}</h4>
                <p>{item['body']}</p><p class="muted">{item['level']}</p></div>""",
                unsafe_allow_html=True,
            )
    if classes["numeric"]:
        st.subheader("Numeric feature snapshot")
        selected_nums = st.multiselect(
            "Choose numeric columns for comparison",
            classes["numeric"],
            default=classes["numeric"][:min(4, len(classes["numeric"]))],
            key="overview_numeric",
        )
        if selected_nums:
            scaled = df[selected_nums].copy()
            for col in selected_nums:
                lo, hi = scaled[col].min(), scaled[col].max()
                scaled[col] = (scaled[col] - lo) / (hi - lo) if pd.notna(lo) and pd.notna(hi) and hi != lo else 0
            fig = px.line(scaled.reset_index(drop=True), y=selected_nums, title="Min-max scaled comparison (0–1)")
            st.plotly_chart(style_chart(fig), use_container_width=True)
            st.caption("Scaling is only for visual comparison; values here are not the original units.")


# ------------------------------- Data quality -------------------------------
with tab_quality:
    st.subheader("Quality scorecard")
    # Transparent heuristic score, not a certified data-quality standard.
    missing_penalty = min(40, quality["missing_pct"] * 1.2)
    duplicate_penalty = min(25, quality["duplicate_pct"] * 1.5)
    constant_penalty = min(15, quality["constant_columns"] / max(len(df.columns), 1) * 30)
    quality_score = max(0, round(100 - missing_penalty - duplicate_penalty - constant_penalty))
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Heuristic quality score", f"{quality_score}/100")
    q2.metric("Missingness", f"{quality['missing_pct']:.2f}%")
    q3.metric("Exact duplicate rows", f"{quality['duplicate_rows']:,}")
    q4.metric("Constant columns", f"{quality['constant_columns']:,}")
    st.caption("The score is an educational heuristic based only on missingness, duplicate rows, and constant columns. It is not a universal quality certification.")
    st.subheader("Column-level profile")
    st.dataframe(profile, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        miss = profile[profile["Missing"] > 0].sort_values("Missing %", ascending=True)
        if not miss.empty:
            fig = px.bar(miss, x="Missing %", y="Column", orientation="h", title="Missingness by column", text="Missing %")
            st.plotly_chart(style_chart(fig), use_container_width=True)
        else:
            st.success("No missing cells detected.")
    with right:
        if classes["numeric"]:
            selected_missing_col = st.selectbox("Missingness pattern — numeric columns", classes["numeric"], key="missing_pattern_col")
            sample_missing = df[classes["numeric"]].head(min(len(df), 500))
            fig = go.Figure()
            for col in classes["numeric"][:min(15, len(classes["numeric"]))]:
                fig.add_trace(go.Scatter(
                    x=list(range(len(sample_missing))),
                    y=sample_missing[col].isna().astype(int),
                    mode="markers",
                    name=col,
                    marker=dict(size=4),
                ))
            fig.update_layout(title="Missingness map (first 500 rows)", xaxis_title="Row index", yaxis_title="Missing flag (1 = missing)")
            st.plotly_chart(style_chart(fig), use_container_width=True)
        else:
            st.info("No numeric columns available for this view.")
    st.subheader("Potential data issues")
    issue_rows = []
    for _, row in profile.iterrows():
        if row["Missing %"] > 0:
            issue_rows.append({"Severity": "Review", "Issue": "Missing values", "Column": row["Column"], "Evidence": f"{row['Missing']} missing ({row['Missing %']:.1f}%)"})
        if row["Constant"]:
            issue_rows.append({"Severity": "Review", "Issue": "Constant/near-empty feature", "Column": row["Column"], "Evidence": "At most one distinct non-null value"})
        if row["Unique %"] > 98 and row["Unique"] > 20:
            issue_rows.append({"Severity": "Info", "Issue": "Possible identifier", "Column": row["Column"], "Evidence": f"{row['Unique %']:.1f}% unique non-null values"})
    if issue_rows:
        st.dataframe(pd.DataFrame(issue_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No issues matched the current quick-check rules.")


# ------------------------------ Distributions -------------------------------
with tab_univariate:
    st.subheader("Explore a single feature")
    all_features = df.columns.tolist()
    feature = st.selectbox("Feature", all_features, key="uni_feature")
    series = df[feature]
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{numeric.mean():,.3g}" if len(numeric) else "—")
        c2.metric("Median", f"{numeric.median():,.3g}" if len(numeric) else "—")
        c3.metric("Std. deviation", f"{numeric.std():,.3g}" if len(numeric) > 1 else "—")
        c4.metric("Skewness", f"{numeric.skew():.2f}" if len(numeric) > 2 else "—")
        bins = st.slider("Histogram bins", 5, 100, 30, key="hist_bins")
        fig = px.histogram(chart_df, x=feature, nbins=bins, marginal="box", title=f"Distribution of {feature}")
        st.plotly_chart(style_chart(fig), use_container_width=True)
        low, high = outlier_bounds(series, outlier_method)
        flagged = series.notna() & ((series < low) | (series > high))
        st.info(f"{int(flagged.sum()):,} potential outlier(s) flagged using {outlier_method}. Bounds: {low:,.4g} to {high:,.4g}.")
        with st.expander("Show flagged records"):
            st.dataframe(df.loc[flagged].head(500), use_container_width=True)
    elif pd.api.types.is_datetime64_any_dtype(series):
        date_counts = series.dropna().dt.to_period("M").astype(str).value_counts().sort_index()
        if not date_counts.empty:
            fig = px.bar(x=date_counts.index, y=date_counts.values, labels={"x": "Month", "y": "Records"}, title=f"Records over time: {feature}")
            st.plotly_chart(style_chart(fig), use_container_width=True)
        st.write(series.describe(datetime_is_numeric=True) if hasattr(series, "describe") else series.describe())
    else:
        values = series.fillna("(Missing)").astype(str).value_counts().head(MAX_CHART_CATEGORIES).reset_index()
        values.columns = ["Category", "Count"]
        if not values.empty:
            fig = px.bar(values.sort_values("Count"), x="Count", y="Category", orientation="h", title=f"Top values in {feature}", text="Count")
            st.plotly_chart(style_chart(fig), use_container_width=True)
        st.caption(f"Showing up to {MAX_CHART_CATEGORIES} most frequent values.")
        st.dataframe(values, use_container_width=True, hide_index=True)
    if classes["numeric"]:
        st.divider()
        st.subheader("Numeric distribution gallery")
        gallery_cols = st.multiselect(
            "Select numeric features",
            classes["numeric"],
            default=classes["numeric"][:min(4, len(classes["numeric"]))],
            key="gallery_cols",
        )
        if gallery_cols:
            for start in range(0, len(gallery_cols), 2):
                pair = gallery_cols[start:start + 2]
                plot_cols = st.columns(len(pair))
                for slot, col in zip(plot_cols, pair):
                    with slot:
                        fig = px.histogram(chart_df, x=col, marginal="box", nbins=30, title=col)
                        st.plotly_chart(style_chart(fig), use_container_width=True)


# ------------------------------ Relationships ------------------------------
with tab_relationships:
    st.subheader("Relationships and group comparisons")
    if len(classes["numeric"]) >= 2:
        st.markdown("#### Correlation matrix")
        corr_cols = st.multiselect(
            "Numeric columns for correlation",
            classes["numeric"],
            default=classes["numeric"][:min(15, len(classes["numeric"]))],
            key="corr_cols",
        )
        if len(corr_cols) >= 2:
            corr = df[corr_cols].corr(method=correlation_method.lower(), min_periods=5)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title=f"{correlation_method} correlation")
            st.plotly_chart(style_chart(fig), use_container_width=True)
            pairs = []
            for i, a in enumerate(corr.columns):
                for b in corr.columns[i + 1:]:
                    value = corr.loc[a, b]
                    if pd.notna(value):
                        pairs.append({"Feature A": a, "Feature B": b, "Correlation": value, "Absolute strength": abs(value)})
            if pairs:
                st.dataframe(pd.DataFrame(pairs).sort_values("Absolute strength", ascending=False).drop(columns="Absolute strength").head(20),
                             use_container_width=True, hide_index=True)
    else:
        st.info("At least two non-constant numeric columns are needed for correlation analysis.")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Scatter explorer")
        if len(classes["numeric"]) >= 2:
            x_col = st.selectbox("X-axis", classes["numeric"], index=0, key="scatter_x")
            y_options = [c for c in classes["numeric"] if c != x_col]
            y_col = st.selectbox("Y-axis", y_options, key="scatter_y")
            color_options = ["(None)"] + classes["categorical"][:100]
            color_col = st.selectbox("Color by category", color_options, key="scatter_color")
            scatter_data = chart_df[[x_col, y_col] + ([color_col] if color_col != "(None)" else [])].dropna(subset=[x_col, y_col])
            fig = px.scatter(scatter_data, x=x_col, y=y_col, color=None if color_col == "(None)" else color_col,
                             opacity=.7, render_mode="webgl", title=f"{y_col} vs {x_col}")
            st.plotly_chart(style_chart(fig), use_container_width=True)
            if len(scatter_data) > MAX_SCATTER_POINTS:
                st.caption(f"Scatter plot is sampled to {MAX_SCATTER_POINTS:,} points for responsiveness.")
        else:
            st.info("Two numeric columns are required.")
    with c2:
        st.markdown("#### Compare numeric values across groups")
        if classes["categorical"] and classes["numeric"]:
            group_col = st.selectbox("Group/category", classes["categorical"], key="group_col")
            measure_col = st.selectbox("Numeric measure", classes["numeric"], key="measure_col")
            agg = st.selectbox("Aggregation", ["Mean", "Median", "Sum", "Count"], key="group_agg")
            work = df[[group_col, measure_col]].copy()
            work[group_col] = work[group_col].fillna("(Missing)").astype(str)
            work = work[work[group_col].isin(work[group_col].value_counts().head(MAX_CHART_CATEGORIES).index)]
            func = {"Mean": "mean", "Median": "median", "Sum": "sum", "Count": "count"}[agg]
            grouped = work.groupby(group_col, dropna=False)[measure_col].agg(func).sort_values(ascending=False).reset_index()
            grouped.columns = [group_col, f"{agg} of {measure_col}"]
            fig = px.bar(grouped, x=group_col, y=f"{agg} of {measure_col}", title=f"{agg} {measure_col} by {group_col}")
            st.plotly_chart(style_chart(fig), use_container_width=True)
            st.dataframe(grouped, use_container_width=True, hide_index=True)
        else:
            st.info("A categorical and a numeric column are needed for group comparisons.")
    st.divider()
    st.markdown("#### Pivot-table builder")
    if classes["categorical"]:
        pivot_rows = st.multiselect("Rows", classes["categorical"], default=classes["categorical"][:1], key="pivot_rows")
        pivot_cols = st.selectbox("Columns (optional)", ["(None)"] + classes["categorical"], key="pivot_cols")
        pivot_values = st.selectbox("Values", ["(Count rows)"] + classes["numeric"], key="pivot_values")
        pivot_agg = st.selectbox("Pivot aggregation", ["mean", "sum", "median", "count"], key="pivot_agg")
        if st.button("Build pivot table", key="build_pivot"):
            if not pivot_rows:
                st.warning("Choose at least one row grouping.")
            else:
                vals = None if pivot_values == "(Count rows)" else pivot_values
                col_arg = None if pivot_cols == "(None)" else pivot_cols
                try:
                    pivot = pd.pivot_table(
                        df, index=pivot_rows, columns=col_arg, values=vals,
                        aggfunc=pivot_agg if vals else "size", fill_value=0, observed=False
                    )
                    st.dataframe(pivot, use_container_width=True)
                    st.download_button("Download pivot CSV", to_csv_bytes(pivot.reset_index()), "autoviziq_pivot.csv", "text/csv")
                except Exception as exc:
                    st.error(f"Could not build pivot table: {exc}")
    else:
        st.info("Categorical columns are required for a pivot table.")


# -------------------------------- Time analysis ------------------------------
with tab_time:
    st.subheader("Time-series analysis")
    if not classes["datetime"]:
        st.info("No date/time column was detected. If a column contains dates, check its format or convert it before uploading.")
    else:
        date_col = st.selectbox("Date/time column", classes["datetime"], key="time_date_col")
        time_measure_options = ["(Record count)"] + classes["numeric"]
        time_measure = st.selectbox("Measure", time_measure_options, key="time_measure")
        frequency = st.selectbox("Time granularity", ["Day", "Week", "Month", "Quarter", "Year"], index=2, key="time_freq")
        freq_code = {"Day": "D", "Week": "W", "Month": "MS", "Quarter": "QS", "Year": "YS"}[frequency]
        time_df = df[df[date_col].notna()].copy()
        if not time_df.empty:
            time_df["_period"] = time_df[date_col].dt.to_period(
                {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}[frequency]
            ).dt.start_time
            if time_measure == "(Record count)":
                trend = time_df.groupby("_period").size().rename("Records").reset_index()
                y_name = "Records"
            else:
                trend = time_df.groupby("_period")[time_measure].agg(["sum", "mean", "median"]).reset_index()
                trend = trend.rename(columns={"_period": "Period"})
                y_name = st.selectbox("Aggregation to plot", ["sum", "mean", "median"], key="time_agg")
                trend = trend.rename(columns={"Period": "_period"})
            if time_measure == "(Record count)":
                pass
            fig = px.line(trend, x="_period", y=y_name, markers=True, title=f"{y_name} by {frequency.lower()}")
            st.plotly_chart(style_chart(fig), use_container_width=True)
            if time_measure != "(Record count)":
                st.dataframe(trend.rename(columns={"_period": "Period"}), use_container_width=True, hide_index=True)
            with st.expander("Period-over-period change"):
                if time_measure == "(Record count)":
                    trend["Change %"] = trend["Records"].pct_change() * 100
                else:
                    trend["Change %"] = trend[y_name].pct_change() * 100
                st.dataframe(trend.rename(columns={"_period": "Period"}), use_container_width=True, hide_index=True)
        else:
            st.warning("The selected date column contains no valid timestamps.")


# ------------------------------ Advanced analytics ---------------------------
with tab_advanced:
    st.subheader("Advanced analysis lab")
    if not show_advanced:
        st.info("Enable advanced analytics in the sidebar to use clustering and anomaly detection.")
    else:
        st.caption("Advanced outputs are exploratory. They are not automatically a prediction or business recommendation.")
        adv_tab1, adv_tab2, adv_tab3 = st.tabs(["Anomaly detection", "Clustering", "Feature relationships"])
        with adv_tab1:
            st.markdown("#### Multivariate anomaly detection")
            if not SKLEARN_AVAILABLE:
                st.warning("scikit-learn is not installed. Install it from requirements.txt to enable this feature.")
            elif len(classes["numeric"]) < 2:
                st.info("At least two numeric features are required.")
            else:
                anomaly_features = st.multiselect(
                    "Features for anomaly detection",
                    classes["numeric"],
                    default=classes["numeric"][:min(5, len(classes["numeric"]))],
                    key="anomaly_features",
                )
                contamination = st.slider("Expected anomaly fraction", .01, .20, .05, .01, key="anomaly_contamination")
                if st.button("Run anomaly detection", key="run_anomaly"):
                    if len(anomaly_features) < 2:
                        st.warning("Select at least two numeric features.")
                    else:
                        model_data = df[anomaly_features].replace([np.inf, -np.inf], np.nan)
                        imputer = SimpleImputer(strategy="median")
                        filled = imputer.fit_transform(model_data)
                        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
                        labels = model.fit_predict(filled)
                        scores = model.decision_function(filled)
                        anomaly_results = df.copy()
                        anomaly_results["Anomaly_Flag"] = np.where(labels == -1, "Potential anomaly", "Typical")
                        anomaly_results["Anomaly_Score"] = scores
                        st.session_state["anomaly_results"] = anomaly_results
                        st.session_state["anomaly_features_run"] = anomaly_features
                if "anomaly_results" in st.session_state:
                    results = st.session_state["anomaly_results"]
                    anomaly_rows = results[results["Anomaly_Flag"] == "Potential anomaly"]
                    st.metric("Potential anomalies", f"{len(anomaly_rows):,}", f"{len(anomaly_rows) / max(len(results), 1):.1%} of rows")
                    st.dataframe(anomaly_rows.sort_values("Anomaly_Score").head(100), use_container_width=True, hide_index=True)
                    fig = px.scatter(results.sample(min(len(results), sample_limit), random_state=42),
                                     x=st.session_state["anomaly_features_run"][0],
                                     y=st.session_state["anomaly_features_run"][1],
                                     color="Anomaly_Flag", hover_data=["Anomaly_Score"])
                    st.plotly_chart(style_chart(fig, "Isolation Forest results"), use_container_width=True)
                    st.download_button("Download anomaly-labelled dataset", to_csv_bytes(results), "autoviziq_anomalies.csv", "text/csv")
                    st.caption("Isolation Forest identifies unusual combinations of selected numeric values. An anomaly is not necessarily an error or fraud.")
        with adv_tab2:
            st.markdown("#### Customer/record segmentation with K-means")
            if not SKLEARN_AVAILABLE:
                st.warning("scikit-learn is required for clustering.")
            elif len(classes["numeric"]) < 2:
                st.info("At least two numeric features are required.")
            else:
                cluster_features = st.multiselect(
                    "Features for clustering",
                    classes["numeric"],
                    default=classes["numeric"][:min(4, len(classes["numeric"]))],
                    key="cluster_features",
                )
                k = st.slider("Number of clusters (k)", 2, 8, 3, key="cluster_k")
                if st.button("Run clustering", key="run_cluster"):
                    if len(cluster_features) < 2:
                        st.warning("Select at least two numeric features.")
                    else:
                        X = df[cluster_features].replace([np.inf, -np.inf], np.nan)
                        imputer = SimpleImputer(strategy="median")
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(imputer.fit_transform(X))
                        if len(X_scaled) < k:
                            st.error("There must be at least k rows.")
                        else:
                            model = KMeans(n_clusters=k, random_state=42, n_init=10)
                            labels = model.fit_predict(X_scaled)
                            clustered = df.copy()
                            clustered["Cluster"] = labels.astype(str)
                            st.session_state["cluster_results"] = clustered
                            st.session_state["cluster_features_run"] = cluster_features
                            st.session_state["cluster_k_run"] = k
                            if len(X_scaled) > 2 and len(np.unique(labels)) > 1:
                                try:
                                    st.session_state["cluster_silhouette"] = float(silhouette_score(
                                        X_scaled if len(X_scaled) <= 5000 else X_scaled[np.random.default_rng(42).choice(len(X_scaled), 5000, replace=False)],
                                        labels if len(labels) <= 5000 else labels[np.random.default_rng(42).choice(len(labels), 5000, replace=False)]
                                    ))
                                except Exception:
                                    st.session_state["cluster_silhouette"] = None
                if "cluster_results" in st.session_state:
                    clustered = st.session_state["cluster_results"]
                    feature_run = st.session_state["cluster_features_run"]
                    counts = clustered["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Rows")
                    st.dataframe(counts, use_container_width=True, hide_index=True)
                    silhouette = st.session_state.get("cluster_silhouette")
                    if silhouette is not None:
                        st.metric("Approx. silhouette score", f"{silhouette:.3f}")
                    c1, c2 = st.columns(2)
                    with c1:
                        if len(feature_run) >= 2:
                            fig = px.scatter(clustered.sample(min(len(clustered), sample_limit), random_state=42),
                                             x=feature_run[0], y=feature_run[1], color="Cluster",
                                             title="Cluster projection on selected features")
                            st.plotly_chart(style_chart(fig), use_container_width=True)
                    with c2:
                        profile_cluster = clustered.groupby("Cluster")[feature_run].mean().reset_index()
                        st.dataframe(profile_cluster, use_container_width=True, hide_index=True)
                    st.download_button("Download clustered dataset", to_csv_bytes(clustered), "autoviziq_clusters.csv", "text/csv")
                    st.caption("K-means groups rows by numeric similarity after scaling. Cluster labels are arbitrary; interpret them using domain context.")
        with adv_tab3:
            st.markdown("#### Feature importance (predictive association)")
            st.caption("This trains a quick Random Forest to estimate predictive usefulness for a selected target. It is not causal importance and may overfit.")
            if not SKLEARN_AVAILABLE:
                st.warning("scikit-learn is required.")
            else:
                target_candidates = classes["numeric"] + classes["categorical"]
                if not target_candidates:
                    st.info("No usable target candidates.")
                else:
                    target = st.selectbox("Target column", target_candidates, key="importance_target")
                    possible_features = [c for c in df.columns if c != target and c not in classes["datetime"]]
                    chosen_features = st.multiselect("Input features", possible_features, default=possible_features[:min(8, len(possible_features))], key="importance_features")
                    if st.button("Estimate feature importance", key="run_importance"):
                        if not chosen_features:
                            st.warning("Select at least one input feature.")
                        else:
                            work = df[chosen_features + [target]].dropna(subset=[target]).copy()
                            # Keep model runtime bounded and make the sample deterministic.
                            if len(work) > 15000:
                                work = work.sample(15000, random_state=42)
                            y = work[target]
                            is_classification = (not pd.api.types.is_numeric_dtype(y)) or y.nunique() <= 10
                            if is_classification and y.nunique() > 100:
                                st.error("Target has too many categories for this quick classification analysis.")
                            elif len(work) < 30 or y.nunique() < 2:
                                st.error("Need at least 30 usable rows and at least two target values.")
                            else:
                                X = work[chosen_features]
                                numeric_features = X.select_dtypes(include=np.number).columns.tolist()
                                categorical_features = [c for c in chosen_features if c not in numeric_features]
                                transformers = []
                                if numeric_features:
                                    transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features))
                                if categorical_features:
                                    transformers.append(("cat", Pipeline([
                                        ("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=40))
                                    ]), categorical_features))
                                preprocessor = ColumnTransformer(transformers, remainder="drop")
                                model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced") if is_classification else RandomForestRegressor(n_estimators=150, random_state=42)
                                pipeline = Pipeline([("prep", preprocessor), ("model", model)])
                                try:
                                    pipeline.fit(X, y)
                                    names = pipeline.named_steps["prep"].get_feature_names_out()
                                    importances = pipeline.named_steps["model"].feature_importances_
                                    importance_df = pd.DataFrame({"Encoded feature": names, "Importance": importances})
                                    importance_df["Source column"] = importance_df["Encoded feature"].str.replace(r"^(num|cat)__", "", regex=True).str.split("_").str[0]
                                    source_importance = importance_df.groupby("Source column", as_index=False)["Importance"].sum().sort_values("Importance", ascending=False)
                                    st.session_state["importance_results"] = source_importance
                                    st.session_state["importance_target_run"] = target
                                except Exception as exc:
                                    st.error(f"Model could not be fitted: {exc}")
                    if "importance_results" in st.session_state:
                        importance_df = st.session_state["importance_results"]
                        fig = px.bar(importance_df.sort_values("Importance"), x="Importance", y="Source column", orientation="h",
                                     title=f"Feature importance for {st.session_state['importance_target_run']}")
                        st.plotly_chart(style_chart(fig), use_container_width=True)
                        st.dataframe(importance_df, use_container_width=True, hide_index=True)


# ----------------------------- Cleaning workbench ---------------------------
with tab_clean:
    st.subheader("Non-destructive cleaning workbench")
    st.caption("Build a proposed cleaned dataset, inspect the change summary, then download it. The uploaded source is not modified.")
    clean_df = df.copy()
    action_log = []
    c1, c2 = st.columns(2)
    with c1:
        remove_duplicates = st.checkbox("Remove exact duplicate rows", value=True, key="clean_dupes")
        remove_empty_cols = st.checkbox("Remove all-empty columns", value=True, key="clean_empty_cols")
        remove_constant_cols = st.checkbox("Remove constant columns", value=False, key="clean_constant_cols")
        trim_text_clean = st.checkbox("Trim text whitespace", value=True, key="clean_trim")
    with c2:
        missing_strategy = st.selectbox(
            "Missing numeric values",
            ["Leave unchanged", "Fill with median", "Fill with mean", "Fill with zero"],
            key="clean_numeric_missing",
        )
        cat_strategy = st.selectbox(
            "Missing categorical values",
            ["Leave unchanged", "Fill with mode", "Fill with (Missing)"],
            key="clean_cat_missing",
        )
        convert_dates_clean = st.checkbox("Convert detected date strings", value=True, key="clean_dates")
        drop_missing_target = st.checkbox("Do not drop rows automatically", value=True, disabled=True)
    if st.button("Preview proposed cleaning", type="primary", key="preview_clean"):
        if trim_text_clean:
            for col in clean_df.select_dtypes(include=["object", "string"]).columns:
                clean_df[col] = clean_df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
            action_log.append("Trimmed whitespace in text columns.")
        if remove_duplicates:
            before = len(clean_df)
            clean_df = clean_df.drop_duplicates()
            action_log.append(f"Removed {before - len(clean_df):,} exact duplicate rows.")
        if remove_empty_cols:
            empty = clean_df.columns[clean_df.isna().all()].tolist()
            clean_df = clean_df.drop(columns=empty)
            if empty:
                action_log.append(f"Removed all-empty columns: {', '.join(empty)}.")
        if remove_constant_cols:
            constant = [c for c in clean_df.columns if clean_df[c].nunique(dropna=True) <= 1]
            clean_df = clean_df.drop(columns=constant)
            if constant:
                action_log.append(f"Removed constant columns: {', '.join(constant)}.")
        if convert_dates_clean:
            converted, notes = infer_and_convert_types(clean_df)
            clean_df = converted
            action_log.extend([n.replace("**", "") for n in notes])
        numeric_cols = clean_df.select_dtypes(include=np.number).columns
        if missing_strategy != "Leave unchanged":
            for col in numeric_cols:
                if clean_df[col].isna().any():
                    if missing_strategy == "Fill with median":
                        value = clean_df[col].median()
                    elif missing_strategy == "Fill with mean":
                        value = clean_df[col].mean()
                    else:
                        value = 0
                    clean_df[col] = clean_df[col].fillna(value)
                    action_log.append(f"Filled missing numeric values in {col} using {missing_strategy.lower()}.")
        if cat_strategy != "Leave unchanged":
            for col in clean_df.select_dtypes(include=["object", "string", "category"]).columns:
                if clean_df[col].isna().any():
                    if cat_strategy == "Fill with mode":
                        mode = clean_df[col].mode(dropna=True)
                        fill_value = mode.iloc[0] if not mode.empty else "(Missing)"
                    else:
                        fill_value = "(Missing)"
                    clean_df[col] = clean_df[col].fillna(fill_value)
                    action_log.append(f"Filled missing categorical values in {col}.")
        st.session_state["cleaned_df"] = clean_df
        st.session_state["cleaning_log"] = action_log
    if "cleaned_df" in st.session_state:
        cleaned = st.session_state["cleaned_df"]
        log = st.session_state.get("cleaning_log", [])
        before_q = quality_report(df)[1]
        after_q = quality_report(cleaned)[1]
        st.subheader("Before vs after")
        cc = st.columns(4)
        cc[0].metric("Rows", f"{len(cleaned):,}", f"{len(cleaned) - len(df):+,} rows")
        cc[1].metric("Columns", f"{len(cleaned.columns):,}", f"{len(cleaned.columns) - len(df.columns):+} columns")
        cc[2].metric("Missing cells", f"{after_q['missing_cells']:,}", f"{after_q['missing_cells'] - before_q['missing_cells']:+,}")
        cc[3].metric("Duplicate rows", f"{after_q['duplicate_rows']:,}", f"{after_q['duplicate_rows'] - before_q['duplicate_rows']:+,}")
        st.markdown("**Actions proposed/applied to the preview:**")
        if log:
            for entry in log:
                st.markdown(f"- {entry}")
        else:
            st.write("No changes were required by the selected rules.")
        st.dataframe(cleaned.head(100), use_container_width=True, hide_index=True)
        st.download_button("Download cleaned dataset (CSV)", to_csv_bytes(cleaned), "autoviziq_cleaned.csv", "text/csv", type="primary")
        st.download_button("Download cleaning log (TXT)", "\n".join(log).encode("utf-8"), "autoviziq_cleaning_log.txt", "text/plain")
        if st.button("Clear cleaning preview", key="clear_clean"):
            st.session_state.pop("cleaned_df", None)
            st.session_state.pop("cleaning_log", None)
            st.rerun()


# ----------------------------------- Export ----------------------------------
with tab_export:
    st.subheader("Export your analysis")
    st.write("Download the prepared data, statistical summaries, data dictionary, automated findings, or a standalone HTML report.")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("Prepared dataset · CSV", to_csv_bytes(df), "autoviziq_prepared_data.csv", "text/csv", use_container_width=True)
    with e2:
        st.download_button("Data dictionary · CSV", to_csv_bytes(build_data_dictionary(df)), "autoviziq_data_dictionary.csv", "text/csv", use_container_width=True)
    with e3:
        st.download_button("Column quality profile · CSV", to_csv_bytes(profile), "autoviziq_quality_profile.csv", "text/csv", use_container_width=True)
    if not num_desc.empty:
        st.download_button("Numeric summary · CSV", to_csv_bytes(num_desc), "autoviziq_numeric_summary.csv", "text/csv")
    report_json = {
        "app": APP_NAME,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source_label,
        "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "quality": quality,
        "column_types": classes,
        "preparation_steps": prep_log,
        "insights": insights,
        "numeric_summary": num_desc.to_dict(orient="records") if not num_desc.empty else [],
    }
    st.download_button("Analysis metadata · JSON", to_json_bytes(report_json), "autoviziq_analysis.json", "application/json")
    st.download_button("Standalone report · HTML", html_report(df, quality, insights, num_desc), "autoviziq_report.html", "text/html", type="primary")
    st.divider()
    st.subheader("Analysis notes")
    st.markdown(
        """
        - **Privacy:** analysis runs inside the app process. This code does not send your dataset to an external analytics API.
        - **Sampling:** some visualizations sample large datasets for responsiveness; the full prepared dataset is still available for download.
        - **Inference:** detected types and insights are heuristics. Confirm them against the meaning of your data.
        - **Correlation:** associations are not proof of causation.
        - **Anomaly detection and clustering:** exploratory outputs require human interpretation.
        """
    )

st.divider()
st.caption("AutoVizIQ • Automated analytics prototype • Built with Streamlit, Pandas, NumPy, Plotly, and optional scikit-learn")