
import io
import traceback
from typing import Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import reflex as rx


# ============================================================
# AUTOVIZIQ — AUTOMATIC DATA ANALYTICS DASHBOARD
# Reflex + Pandas + Plotly
# ============================================================

LIGHT_BG = "#F1F7F7"
LIGHT_SURFACE = "#FFFFFF"
LIGHT_ALT = "#D5E5E5"
PINK = "#F7CBCA"
TAUPE = "#D3C2C3"
MINT = "#B0D1D0"

DARK_BG = "#172020"
DARK_SURFACE = "#202C2C"
DARK_ALT = "#2B3939"
DARK_MINT = "#6F9694"
DARK_PINK = "#D99C9C"
DARK_TAUPE = "#8E7F80"

MAX_CHARTS = 7


# ============================================================
# DATA PROCESSING
# ============================================================

def read_dataset(filename: str, content: bytes) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file."""
    extension = filename.lower().rsplit(".", 1)[-1]

    if extension == "csv":
        try:
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(
                io.BytesIO(content),
                encoding="latin-1",
                low_memory=False,
            )
    elif extension in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise ValueError(
            "Unsupported file type. Upload a CSV, XLSX, or XLS file."
        )

    if df.empty:
        raise ValueError("The uploaded dataset has no rows.")

    if len(df.columns) == 0:
        raise ValueError("The uploaded dataset has no columns.")

    # Make column labels safe and readable.
    df.columns = [str(column).strip() for column in df.columns]

    # Ensure column names are unique.
    seen = {}
    new_columns = []

    for column in df.columns:
        count = seen.get(column, 0)
        seen[column] = count + 1

        if count == 0:
            new_columns.append(column)
        else:
            new_columns.append(f"{column}_{count + 1}")

    df.columns = new_columns
    return df


def identify_and_convert_types(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Detect numeric, categorical, and date columns."""
    df = df.copy()
    date_columns = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            date_columns.append(column)
            continue

        if pd.api.types.is_numeric_dtype(series):
            continue

        non_null = series.dropna()

        if non_null.empty:
            continue

        # Convert numeric-looking text into numeric values.
        numeric_values = pd.to_numeric(
            series.astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

        numeric_ratio = numeric_values.notna().sum() / max(
            non_null.shape[0], 1
        )

        if numeric_ratio >= 0.90:
            df[column] = numeric_values
            continue

        # Prefer columns whose names suggest dates.
        name_suggests_date = any(
            word in column.lower()
            for word in ("date", "time", "timestamp", "datetime")
        )

        parsed_dates = pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
        )

        date_ratio = parsed_dates.notna().sum() / max(
            non_null.shape[0], 1
        )

        if name_suggests_date and date_ratio >= 0.50:
            df[column] = parsed_dates
            date_columns.append(column)
        elif date_ratio >= 0.85 and non_null.dtype == object:
            df[column] = parsed_dates
            date_columns.append(column)

    numeric_columns = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    ]

    categorical_columns = [
        column
        for column in df.columns
        if column not in numeric_columns
        and column not in date_columns
    ]

    return df, numeric_columns, categorical_columns, date_columns


def clean_dataframe(
    original_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Clean empty rows, empty columns, duplicates, and missing values."""
    df = original_df.copy()
    log = []

    original_rows = len(df)
    original_columns = len(df.columns)

    original_missing = int(df.isna().sum().sum())
    original_duplicates = int(df.duplicated().sum())

    # Remove completely empty columns.
    empty_columns = [
        column for column in df.columns if df[column].isna().all()
    ]

    if empty_columns:
        df = df.drop(columns=empty_columns)
        log.append(
            f"Removed {len(empty_columns)} completely empty column(s)."
        )

    # Remove completely empty rows.
    empty_rows = int(df.isna().all(axis=1).sum())

    if empty_rows:
        df = df.dropna(how="all")
        log.append(f"Removed {empty_rows} completely empty row(s).")

    # Remove duplicate rows.
    duplicates_before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = duplicates_before - len(df)

    if duplicates_removed:
        log.append(
            f"Removed {duplicates_removed} duplicate row(s)."
        )

    # Detect column types before filling missing values.
    df, numeric_columns, categorical_columns, date_columns = (
        identify_and_convert_types(df)
    )

    missing_filled = 0

    for column in numeric_columns:
        missing = int(df[column].isna().sum())

        if missing:
            median = df[column].median()

            if pd.notna(median):
                df[column] = df[column].fillna(median)
                missing_filled += missing
                log.append(
                    f"Filled {missing} missing value(s) in "
                    f"'{column}' using the median."
                )

    for column in categorical_columns:
        missing = int(df[column].isna().sum())

        if missing:
            mode_values = df[column].mode(dropna=True)

            if not mode_values.empty:
                df[column] = df[column].fillna(mode_values.iloc[0])
                missing_filled += missing
                log.append(
                    f"Filled {missing} missing value(s) in "
                    f"'{column}' using the most frequent value."
                )

    # Date columns remain missing when no reliable date replacement
    # can be determined.
    if not log:
        log.append(
            "No major cleaning changes were required."
        )

    final_missing = int(df.isna().sum().sum())

    stats = {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "original_missing": original_missing,
        "original_duplicates": original_duplicates,
        "duplicates_removed": duplicates_removed,
        "missing_filled": missing_filled,
        "final_missing": final_missing,
        "rows_removed": original_rows - len(df),
        "columns_removed": original_columns - len(df.columns),
    }

    return df, log, stats


# ============================================================
# CHART GENERATION
# ============================================================

def style_figure(
    fig: go.Figure,
    title: str,
    dark_mode: bool,
) -> go.Figure:
    """Apply AutoVizIQ styling to every Plotly chart."""
    background = DARK_SURFACE if dark_mode else LIGHT_SURFACE
    text_color = "#F1F7F7" if dark_mode else "#172020"
    grid_color = "#405050" if dark_mode else "#E5ECEC"

    fig.update_layout(
        title={
            "text": title,
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 17},
        },
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor=background,
        plot_bgcolor=background,
        font={"color": text_color, "family": "Arial, sans-serif"},
        margin={"l": 35, "r": 25, "t": 65, "b": 40},
        height=390,
        legend={"orientation": "h", "y": -0.18},
    )

    fig.update_xaxes(
        gridcolor=grid_color,
        zerolinecolor=grid_color,
    )
    fig.update_yaxes(
        gridcolor=grid_color,
        zerolinecolor=grid_color,
    )

    return fig


def build_charts(
    df: pd.DataFrame,
    dark_mode: bool = False,
) -> list[go.Figure]:
    """Automatically generate visualizations from available columns."""
    figures = []

    if df.empty:
        return figures

    df, numeric_columns, categorical_columns, date_columns = (
        identify_and_convert_types(df)
    )

    palette = [
        "#B0D1D0",
        "#F7CBCA",
        "#6F9694",
        "#D3C2C3",
        "#D99C9C",
        "#91B7B5",
        "#C5A5A7",
    ]

    # 1. Distribution of the first numeric column.
    if numeric_columns:
        column = numeric_columns[0]
        values = df[column].dropna()

        if not values.empty:
            fig = px.histogram(
                df,
                x=column,
                nbins=30,
                color_discrete_sequence=[palette[0]],
                labels={column: column},
            )
            figures.append(
                style_figure(
                    fig,
                    f"Distribution of {column}",
                    dark_mode,
                )
            )

    # 2. Most frequent categories.
    if categorical_columns:
        column = categorical_columns[0]
        counts = (
            df[column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(10)
            .sort_values()
        )

        if not counts.empty:
            fig = go.Figure(
                go.Bar(
                    x=counts.values,
                    y=counts.index,
                    orientation="h",
                    marker_color=palette[1],
                )
            )
            fig.update_layout(
                xaxis_title="Count",
                yaxis_title=column,
            )
            figures.append(
                style_figure(
                    fig,
                    f"Top categories in {column}",
                    dark_mode,
                )
            )

    # 3. Relationship between two numeric columns.
    if len(numeric_columns) >= 2:
        x_column = numeric_columns[0]
        y_column = numeric_columns[1]

        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            color_discrete_sequence=[palette[2]],
            opacity=0.75,
            hover_data=[categorical_columns[0]]
            if categorical_columns
            else None,
        )

        figures.append(
            style_figure(
                fig,
                f"{y_column} vs {x_column}",
                dark_mode,
            )
        )

    # 4. Numeric outlier overview.
    if numeric_columns:
        column = numeric_columns[0]

        fig = px.box(
            df,
            y=column,
            color_discrete_sequence=[palette[3]],
            points="outliers",
        )

        figures.append(
            style_figure(
                fig,
                f"Outlier overview: {column}",
                dark_mode,
            )
        )

    # 5. Correlation heatmap.
    if len(numeric_columns) >= 2:
        correlation = df[numeric_columns].corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale=[
                "#F7CBCA",
                "#F1F7F7",
                "#6F9694",
            ],
            zmin=-1,
            zmax=1,
            aspect="auto",
        )

        figures.append(
            style_figure(
                fig,
                "Numeric correlation heatmap",
                dark_mode,
            )
        )

    # 6. Trend over time.
    if date_columns and numeric_columns:
        date_column = date_columns[0]
        value_column = numeric_columns[0]

        trend_df = df[[date_column, value_column]].dropna().copy()

        if not trend_df.empty:
            trend_df = trend_df.sort_values(date_column)

            fig = px.line(
                trend_df,
                x=date_column,
                y=value_column,
                markers=True,
                color_discrete_sequence=[palette[2]],
            )

            figures.append(
                style_figure(
                    fig,
                    f"{value_column} over time",
                    dark_mode,
                )
            )

    # 7. Share of categories.
    if categorical_columns:
        column = categorical_columns[0]
        counts = (
            df[column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(8)
        )

        if not counts.empty:
            fig = px.pie(
                names=counts.index,
                values=counts.values,
                hole=0.48,
                color_discrete_sequence=palette,
            )

            figures.append(
                style_figure(
                    fig,
                    f"Category share: {column}",
                    dark_mode,
                )
            )

    # Always show something informative if the dataset cannot
    # support the standard visualizations.
    if not figures:
        fig = go.Figure()
        fig.add_annotation(
            text=(
                "No suitable numeric or categorical columns were found. "
                "Check the uploaded dataset's contents."
            ),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 15},
        )
        figures.append(
            style_figure(
                fig,
                "Dataset visualization",
                dark_mode,
            )
        )

    return figures[:MAX_CHARTS]


# ============================================================
# AUTOMATIC INSIGHTS
# ============================================================

def generate_insights(
    df: pd.DataFrame,
    original_stats: dict[str, int],
) -> list[str]:
    insights = []

    if df.empty:
        return ["The cleaned dataset contains no rows to analyze."]

    rows, columns = df.shape

    insights.append(
        f"The cleaned dataset contains {rows:,} rows and "
        f"{columns:,} columns."
    )

    missing = original_stats["original_missing"]
    total_original_cells = (
        original_stats["original_rows"]
        * original_stats["original_columns"]
    )

    if total_original_cells:
        completeness = (
            100 * (1 - missing / total_original_cells)
        )
        insights.append(
            f"Original data completeness was approximately "
            f"{max(0, completeness):.1f}%."
        )

    if original_stats["duplicates_removed"]:
        insights.append(
            f"{original_stats['duplicates_removed']:,} duplicate "
            f"row(s) were removed during cleaning."
        )
    else:
        insights.append("No duplicate rows were detected.")

    df, numeric_columns, categorical_columns, date_columns = (
        identify_and_convert_types(df)
    )

    if numeric_columns:
        column = numeric_columns[0]
        values = df[column].dropna()

        if not values.empty:
            insights.append(
                f"'{column}' has an average of "
                f"{values.mean():,.2f} and a median of "
                f"{values.median():,.2f}."
            )

            insights.append(
                f"The observed range for '{column}' is "
                f"{values.min():,.2f} to {values.max():,.2f}."
            )

    if categorical_columns:
        column = categorical_columns[0]
        mode_values = df[column].mode(dropna=True)

        if not mode_values.empty:
            insights.append(
                f"The most common value in '{column}' is "
                f"'{mode_values.iloc[0]}'."
            )

    if len(numeric_columns) >= 2:
        correlation = df[numeric_columns].corr()

        pairs = []

        for i, first in enumerate(numeric_columns):
            for second in numeric_columns[i + 1:]:
                value = correlation.loc[first, second]

                if pd.notna(value):
                    pairs.append((abs(value), first, second, value))

        if pairs:
            _, first, second, value = max(pairs)

            strength = (
                "strong"
                if abs(value) >= 0.70
                else "moderate"
                if abs(value) >= 0.40
                else "weak"
            )

            insights.append(
                f"The strongest observed numeric relationship is "
                f"between '{first}' and '{second}' "
                f"(correlation {value:.2f}, {strength}). "
                f"Correlation does not establish causation."
            )

    if date_columns:
        column = date_columns[0]
        dates = df[column].dropna()

        if not dates.empty:
            insights.append(
                f"'{column}' spans "
                f"{dates.min().date()} to {dates.max().date()}."
            )

    return insights[:8]


# ============================================================
# REFLEX STATE
# ============================================================

class State(rx.State):
    # Interface state.
    dark_mode: bool = False
    is_processing: bool = False
    has_dataset: bool = False

    # Upload and status.
    dataset_name: str = ""
    status_message: str = "Upload a dataset to get started."
    error_message: str = ""

    # Dataset KPIs.
    row_count: int = 0
    column_count: int = 0
    numeric_count: int = 0
    categorical_count: int = 0
    date_count: int = 0
    missing_count: int = 0
    duplicate_count: int = 0
    completeness: float = 0.0

    # Cleaning and analysis.
    cleaning_log: list[str] = []
    insights: list[str] = []
    column_summary: str = ""
    preview_text: str = ""

    # Store the cleaned dataset as JSON so the charts can be
    # rebuilt when the theme changes.
    cleaned_data_json: str = ""

    # IMPORTANT:
    # Each chart has its own Figure state.
    # Do not loop over a list of Figure state variables with
    # rx.foreach when passing the result to rx.plotly.
    chart_count: int = 0
    chart_1: go.Figure = go.Figure()
    chart_2: go.Figure = go.Figure()
    chart_3: go.Figure = go.Figure()
    chart_4: go.Figure = go.Figure()
    chart_5: go.Figure = go.Figure()
    chart_6: go.Figure = go.Figure()
    chart_7: go.Figure = go.Figure()

    def store_figures(self, figures: list[go.Figure]):
        """Place figures into fixed, individually typed state fields."""
        self.chart_count = min(len(figures), MAX_CHARTS)

        empty = go.Figure()

        self.chart_1 = figures[0] if len(figures) > 0 else empty
        self.chart_2 = figures[1] if len(figures) > 1 else empty
        self.chart_3 = figures[2] if len(figures) > 2 else empty
        self.chart_4 = figures[3] if len(figures) > 3 else empty
        self.chart_5 = figures[4] if len(figures) > 4 else empty
        self.chart_6 = figures[5] if len(figures) > 5 else empty
        self.chart_7 = figures[6] if len(figures) > 6 else empty

    async def handle_upload(self, files: list[rx.UploadFile]):
        """Read, clean, analyze, and visualize an uploaded dataset."""
        print("DEBUG: UPLOAD HANDLER STARTED")
        print("DEBUG: FILE COUNT:", len(files))

        if not files:
            self.error_message = "No file was received. Please upload again."
            self.status_message = "Upload failed."
            return

        self.is_processing = True
        self.error_message = ""
        self.status_message = "Reading your dataset..."
        yield

        try:
            uploaded_file = files[0]
            filename = uploaded_file.filename or "uploaded_dataset.csv"

            print("DEBUG: FILE NAME:", filename)

            content = await uploaded_file.read()

            if not content:
                raise ValueError("The uploaded file is empty.")

            original_df = read_dataset(filename, content)

            print(
                "DEBUG: DATASET READ:",
                original_df.shape,
            )

            # Preserve original missing-value statistics before cleaning.
            original_missing = int(original_df.isna().sum().sum())
            original_duplicates = int(original_df.duplicated().sum())

            self.status_message = "Cleaning and analyzing your data..."
            yield

            cleaned_df, cleaning_log, stats = clean_dataframe(original_df)

            if cleaned_df.empty:
                raise ValueError(
                    "No usable rows remain after cleaning the dataset."
                )

            (
                cleaned_df,
                numeric_columns,
                categorical_columns,
                date_columns,
            ) = identify_and_convert_types(cleaned_df)

            # Update KPIs.
            self.dataset_name = filename
            self.row_count = len(cleaned_df)
            self.column_count = len(cleaned_df.columns)
            self.numeric_count = len(numeric_columns)
            self.categorical_count = len(categorical_columns)
            self.date_count = len(date_columns)
            self.missing_count = original_missing
            self.duplicate_count = original_duplicates

            total_cells = (
                stats["original_rows"] * stats["original_columns"]
            )

            self.completeness = (
                round(
                    100 * (1 - original_missing / total_cells),
                    1,
                )
                if total_cells
                else 100.0
            )

            self.cleaning_log = cleaning_log
            self.column_summary = (
                "Numeric columns: "
                + (", ".join(numeric_columns) or "None")
                + "\n\nCategorical columns: "
                + (", ".join(categorical_columns) or "None")
                + "\n\nDate columns: "
                + (", ".join(date_columns) or "None")
            )

            # Keep a JSON representation for theme changes.
            self.cleaned_data_json = cleaned_df.to_json(
                orient="split",
                date_format="iso",
            )

            self.preview_text = cleaned_df.head(10).to_string(
                index=False,
                max_cols=12,
            )

            # Generate insights and figures from the uploaded dataset.
            self.status_message = "Generating dashboard visuals..."
            yield

            self.insights = generate_insights(
                cleaned_df,
                stats,
            )

            figures = build_charts(
                cleaned_df,
                dark_mode=self.dark_mode,
            )

            self.store_figures(figures)

            self.has_dataset = True
            self.status_message = (
                f"Dashboard ready — {self.row_count:,} rows analyzed "
                f"and {self.chart_count} visual(s) generated."
            )

            print("DEBUG: CLEANED DATASET:", cleaned_df.shape)
            print("DEBUG: CHART COUNT:", self.chart_count)
            print("DEBUG: DASHBOARD READY")

        except Exception as exc:
            traceback.print_exc()

            self.error_message = (
                f"{type(exc).__name__}: {str(exc)}"
            )
            self.status_message = "Could not generate the dashboard."
            self.has_dataset = False

        finally:
            self.is_processing = False

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode

        if self.cleaned_data_json:
            try:
                df = pd.read_json(
                    io.StringIO(self.cleaned_data_json),
                    orient="split",
                )

                figures = build_charts(
                    df,
                    dark_mode=self.dark_mode,
                )

                self.store_figures(figures)

            except Exception:
                traceback.print_exc()

    def reset_dashboard(self):
        self.has_dataset = False
        self.is_processing = False
        self.dataset_name = ""
        self.status_message = "Upload a dataset to get started."
        self.error_message = ""
        self.row_count = 0
        self.column_count = 0
        self.numeric_count = 0
        self.categorical_count = 0
        self.date_count = 0
        self.missing_count = 0
        self.duplicate_count = 0
        self.completeness = 0.0
        self.cleaning_log = []
        self.insights = []
        self.column_summary = ""
        self.preview_text = ""
        self.cleaned_data_json = ""
        self.store_figures([])


# ============================================================
# UI HELPERS
# ============================================================

def page_style():
    return {
        "background": rx.cond(
            State.dark_mode,
            DARK_BG,
            LIGHT_BG,
        ),
        "color": rx.cond(
            State.dark_mode,
            "#F1F7F7",
            "#172020",
        ),
        "min_height": "100vh",
        "padding": ["18px", "28px", "36px"],
        "font_family": "Arial, sans-serif",
        "transition": "background 0.2s ease",
    }


def panel_style():
    return {
        "background": rx.cond(
            State.dark_mode,
            DARK_SURFACE,
            LIGHT_SURFACE,
        ),
        "color": rx.cond(
            State.dark_mode,
            "#F1F7F7",
            "#172020",
        ),
        "border": "1px solid "
        + rx.cond(State.dark_mode, DARK_ALT, TAUPE),
        "border_radius": "18px",
        "padding": "20px",
        "width": "100%",
        "min_width": "0",
        "box_shadow": "0 5px 18px rgba(20, 40, 40, 0.04)",
    }


def section_title(title: str, subtitle: str = ""):
    return rx.vstack(
        rx.heading(title, size="5"),
        rx.text(
            subtitle,
            color=rx.cond(State.dark_mode, "#BDCECE", "#667777"),
            size="2",
        ),
        align="start",
        spacing="1",
        width="100%",
    )


def metric_card(label: str, value: Any, description: str):
    return rx.box(
        rx.vstack(
            rx.text(
                label,
                size="2",
                color=rx.cond(
                    State.dark_mode,
                    "#BDCECE",
                    "#667777",
                ),
            ),
            rx.heading(value, size="7"),
            rx.text(description, size="1"),
            align="start",
            spacing="2",
            width="100%",
        ),
        style=panel_style(),
    )


def chart_card(title: str, figure: Any):
    return rx.box(
        rx.vstack(
            rx.heading(title, size="4"),
            # IMPORTANT: figure is passed directly from a fixed
            # Figure state field, not from rx.foreach.
            rx.plotly(
                data=figure,
                width="100%",
                height="400px",
            ),
            align="start",
            spacing="3",
            width="100%",
        ),
        style=panel_style(),
    )


def insight_card(insight: Any):
    return rx.box(
        rx.hstack(
            rx.text("✦", color="#6F9694", size="4"),
            rx.text(insight, size="2"),
            align="start",
            spacing="3",
            width="100%",
        ),
        style=panel_style(),
    )


def log_item(item: Any):
    return rx.hstack(
        rx.text("✓", color="#6F9694", weight="bold"),
        rx.text(item, size="2"),
        align="start",
        spacing="3",
        width="100%",
    )


# ============================================================
# PAGE SECTIONS
# ============================================================

def header():
    return rx.hstack(
        rx.vstack(
            rx.heading(
                "AutoVizIQ",
                size="8",
                color=rx.cond(State.dark_mode, "#F1F7F7", "#172020"),
            ),
            rx.text(
                "From raw data to actionable insights.",
                size="3",
                color=rx.cond(State.dark_mode, "#BDCECE", "#667777"),
            ),
            align="start",
            spacing="1",
        ),
        rx.spacer(),
        rx.button(
            rx.cond(
                State.dark_mode,
                "Switch to Light Mode",
                "Switch to Night Mode",
            ),
            on_click=State.toggle_theme,
            variant="outline",
            color_scheme="gray",
        ),
        align="center",
        width="100%",
        padding_bottom="10px",
    )


def upload_section():
    return rx.box(
        rx.vstack(
            rx.heading("Upload your dataset", size="5"),
            rx.text(
                "AutoVizIQ will clean the data, identify column types, "
                "calculate key metrics, and generate charts automatically.",
                size="2",
            ),
            rx.upload(
                rx.vstack(
                    rx.text("Drop your CSV or Excel file here", size="4"),
                    rx.text(
                        "Or click to browse files",
                        size="2",
                        color="#667777",
                    ),
                    rx.text(
                        "Supported formats: .csv, .xlsx, .xls",
                        size="1",
                        color="#667777",
                    ),
                    align="center",
                    spacing="2",
                    padding="30px",
                    width="100%",
                ),
                id="dataset_upload",
                multiple=False,
                max_files=1,
                accept={
                    ".csv": ["text/csv"],
                    ".xlsx": [
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ],
                    ".xls": ["application/vnd.ms-excel"],
                },
                on_drop=State.handle_upload(
                    rx.upload_files(upload_id="dataset_upload")
                ),
                border="2px dashed "
                + rx.cond(State.dark_mode, DARK_MINT, MINT),
                border_radius="16px",
                width="100%",
                cursor="pointer",
            ),
            rx.cond(
                State.is_processing,
                rx.hstack(
                    rx.spinner(),
                    rx.text(State.status_message),
                    align="center",
                    spacing="3",
                ),
                rx.text(
                    State.status_message,
                    size="2",
                    color=rx.cond(
                        State.dark_mode,
                        "#BDCECE",
                        "#667777",
                    ),
                ),
            ),
            rx.cond(
                State.error_message != "",
                rx.box(
                    rx.text(
                        State.error_message,
                        color="#B42318",
                        size="2",
                    ),
                    background="#FDECEC",
                    padding="12px",
                    border_radius="10px",
                    width="100%",
                ),
                rx.fragment(),
            ),
            rx.cond(
                State.has_dataset,
                rx.hstack(
                    rx.text(
                        "Loaded dataset:",
                        weight="bold",
                        size="2",
                    ),
                    rx.text(State.dataset_name, size="2"),
                    rx.spacer(),
                    rx.button(
                        "Upload another dataset",
                        on_click=State.reset_dashboard,
                        variant="outline",
                        color_scheme="gray",
                    ),
                    align="center",
                    wrap="wrap",
                    width="100%",
                ),
                rx.fragment(),
            ),
            align="start",
            spacing="4",
            width="100%",
        ),
        style=panel_style(),
    )


def overview_section():
    return rx.vstack(
        section_title(
            "Dataset overview",
            "A quick summary of the uploaded data.",
        ),
        rx.grid(
            metric_card(
                "Rows",
                State.row_count.to_string(),
                "Rows after cleaning",
            ),
            metric_card(
                "Columns",
                State.column_count.to_string(),
                "Columns retained",
            ),
            metric_card(
                "Numeric columns",
                State.numeric_count.to_string(),
                "Suitable for numeric analysis",
            ),
            metric_card(
                "Completeness",
                State.completeness.to_string() + "%",
                "Before cleaning",
            ),
            metric_card(
                "Missing values",
                State.missing_count.to_string(),
                "In the original dataset",
            ),
            metric_card(
                "Duplicate rows",
                State.duplicate_count.to_string(),
                "Found in original data",
            ),
            columns={
                "initial": "1",
                "sm": "2",
                "lg": "3",
            },
            spacing="4",
            width="100%",
        ),
        align="start",
        spacing="4",
        width="100%",
    )


def cleaning_section():
    return rx.box(
        rx.vstack(
            section_title(
                "Data cleaning report",
                "Changes applied before generating your dashboard.",
            ),
            rx.foreach(State.cleaning_log, log_item),
            rx.divider(),
            rx.heading("Detected column types", size="3"),
            rx.text(
                State.column_summary,
                white_space="pre-wrap",
                size="2",
            ),
            align="start",
            spacing="3",
            width="100%",
        ),
        style=panel_style(),
    )


def insights_section():
    return rx.vstack(
        section_title(
            "Automatic insights",
            "Initial observations calculated from your dataset.",
        ),
        rx.grid(
            rx.foreach(State.insights, insight_card),
            columns={
                "initial": "1",
                "md": "2",
            },
            spacing="4",
            width="100%",
        ),
        align="start",
        spacing="4",
        width="100%",
    )


def charts_section():
    # Fixed chart slots prevent the previous dynamic-figure
    # rx.foreach rendering issue.
    return rx.vstack(
        section_title(
            "Interactive dashboard",
            "Charts are generated from the columns detected in your file.",
        ),
        rx.grid(
            rx.cond(
                State.chart_count >= 1,
                chart_card("Visualization 1", State.chart_1),
                rx.fragment(),
            ),
            rx.cond(
                State.chart_count >= 2,
                chart_card("Visualization 2", State.chart_2),
                rx.fragment(),
            ),
            rx.cond(
                State.chart_count >= 3,
                chart_card("Visualization 3", State.chart_3),
                rx.fragment(),
            ),
            rx.cond(
                State.chart_count >= 4,
                chart_card("Visualization 4", State.chart_4),
                rx.fragment(),
            ),
            rx.cond(
                State.chart_count >= 5,
                chart_card("Visualization 5", State.chart_5),
                rx.fragment(),
            ),
            rx.cond(
                State.chart_count >= 6,
                chart_card("Visualization 6", State.chart_6),
                rx.fragment(),
            ),
            rx.cond(
                State.chart_count >= 7,
                chart_card("Visualization 7", State.chart_7),
                rx.fragment(),
            ),
            columns={
                "initial": "1",
                "xl": "2",
            },
            spacing="4",
            width="100%",
        ),
        align="start",
        spacing="4",
        width="100%",
    )


def preview_section():
    return rx.box(
        rx.vstack(
            section_title(
                "Data preview",
                "The first 10 rows of the cleaned dataset.",
            ),
            rx.box(
                rx.text(
                    State.preview_text,
                    white_space="pre",
                    font_family="monospace",
                    font_size="12px",
                    overflow_x="auto",
                ),
                width="100%",
                overflow_x="auto",
                padding="12px",
                background=rx.cond(
                    State.dark_mode,
                    DARK_BG,
                    "#F8FAFA",
                ),
                border_radius="10px",
            ),
            align="start",
            spacing="3",
            width="100%",
        ),
        style=panel_style(),
    )


def dashboard_section():
    return rx.vstack(
        overview_section(),
        cleaning_section(),
        insights_section(),
        charts_section(),
        preview_section(),
        align="start",
        spacing="6",
        width="100%",
    )


def footer():
    return rx.center(
        rx.text(
            "AutoVizIQ · Automated data understanding and visualization",
            size="1",
            color=rx.cond(State.dark_mode, "#BDCECE", "#667777"),
        ),
        padding_top="10px",
        padding_bottom="10px",
        width="100%",
    )


# ============================================================
# MAIN PAGE
# ============================================================

def index():
    return rx.box(
        rx.vstack(
            header(),
            upload_section(),
            rx.cond(
                State.has_dataset,
                dashboard_section(),
                rx.box(
                    rx.vstack(
                        rx.text("✦", size="7", color="#6F9694"),
                        rx.heading(
                            "Your dashboard will appear here",
                            size="5",
                        ),
                        rx.text(
                            "Upload a dataset above to generate your "
                            "KPIs, data-quality report, insights, "
                            "and interactive visualizations.",
                            size="2",
                            text_align="center",
                            max_width="600px",
                        ),
                        align="center",
                        spacing="3",
                        width="100%",
                    ),
                    style=panel_style(),
                    width="100%",
                    padding="36px",
                ),
            ),
            footer(),
            align="start",
            spacing="6",
            width="100%",
            max_width="1500px",
            margin="0 auto",
        ),
        style=page_style(),
        width="100%",
    )


app = rx.App()
app.add_page(index, route="/", title="AutoVizIQ")