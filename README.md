# AutoVizIQ Advanced

A single-file Streamlit analytics dashboard with automated profiling, data-quality checks, distributions, relationship analysis, time-series summaries, a non-destructive cleaning workbench, exports, and optional scikit-learn analytics.

## Run locally (Mac)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Supported files
- CSV / TXT
- Excel (`.xlsx`, `.xls`)
- JSON
- Parquet

## Features
- Upload a dataset or use the synthetic retail demo dataset
- Conservative date inference and optional numeric-text conversion
- Data dictionary and column-level quality profile
- Missingness and duplicate checks
- Numeric/categorical distributions and outlier flags
- Pearson, Spearman, or Kendall correlation heatmap
- Scatter plots, group comparisons, and pivot-table builder
- Time-series aggregation by day, week, month, quarter, or year
- Optional Isolation Forest anomaly detection
- Optional K-means clustering
- Optional Random Forest feature-importance analysis
- Cleaning preview that does not overwrite the uploaded source
- CSV, JSON, TXT, and standalone HTML report exports

## Important notes
- The quality score is a transparent heuristic, not a standard certification.
- Outliers are not necessarily errors.
- Correlation and feature importance do not prove causality.
- The model features are exploratory and should be interpreted using domain knowledge.
- Large charts may sample rows for responsiveness, while exports preserve the prepared dataset.

## Project integration warning
This implementation uses Streamlit. Keep a backup of the existing AutoVizIQ project before replacing files. If your existing repository uses Reflex or another framework, update the launch configuration and dependencies rather than mixing the old and new framework code.
