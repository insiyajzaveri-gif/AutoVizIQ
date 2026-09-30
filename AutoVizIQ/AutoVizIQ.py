import reflex as rx
import pandas as pd
import numpy as np
import io
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# AUTOVIZIQ - AUTOMATIC DATA ANALYTICS DASHBOARD
# Pomelli-Inspired Theme
# ============================================================


# ============================================================
# AUTOVIZIQ COLOR SYSTEM
# ============================================================

# -----------------------------
# LIGHT MODE
# -----------------------------

LIGHT_BG = "#F1F7F7"
LIGHT_SURFACE = "#FFFFFF"
LIGHT_SURFACE_ALT = "#D5E5E5"
LIGHT_MINT = "#B0D1D0"
LIGHT_PINK = "#F7CBCA"
LIGHT_TAUPE = "#D3C2C3"

# Main text is deliberately BLACK
LIGHT_TEXT = "#000000"
LIGHT_TEXT_SECONDARY = "#374151"
LIGHT_BORDER = "#D3C2C3"


# -----------------------------
# NIGHT MODE
# -----------------------------

DARK_BG = "#172020"
DARK_SURFACE = "#202C2C"
DARK_SURFACE_ALT = "#2B3939"
DARK_MINT = "#6F9694"
DARK_PINK = "#D99C9C"
DARK_TAUPE = "#8E7F80"

DARK_TEXT = "#FFFFFF"
DARK_TEXT_SECONDARY = "#D1D5D5"
DARK_BORDER = "#526161"


# ============================================================
# THEME HELPERS
# ============================================================

def theme_text(dark_mode):
    return rx.cond(
        dark_mode,
        DARK_TEXT,
        LIGHT_TEXT,
    )


def theme_secondary_text(dark_mode):
    return rx.cond(
        dark_mode,
        DARK_TEXT_SECONDARY,
        LIGHT_TEXT_SECONDARY,
    )


def theme_background(dark_mode):
    return rx.cond(
        dark_mode,
        DARK_BG,
        LIGHT_BG,
    )


def theme_surface(dark_mode):
    return rx.cond(
        dark_mode,
        DARK_SURFACE,
        LIGHT_SURFACE,
    )


def theme_surface_alt(dark_mode):
    return rx.cond(
        dark_mode,
        DARK_SURFACE_ALT,
        LIGHT_SURFACE_ALT,
    )


def theme_border(dark_mode):
    return rx.cond(
        dark_mode,
        DARK_BORDER,
        LIGHT_BORDER,
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_number(value):
    """Format numbers nicely for KPI cards."""

    if value is None:
        return "—"

    try:
        value = float(value)

        if abs(value) >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"

        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"

        if abs(value) >= 1_000:
            return f"{value / 1_000:.2f}K"

        if value.is_integer():
            return f"{int(value):,}"

        return f"{value:,.2f}"

    except Exception:
        return str(value)


# ============================================================
# PLOTLY THEME
# ============================================================

def style_figure(fig, title, dark_mode=False):
    """Apply AutoVizIQ Pomelli-inspired styling to Plotly charts."""

    if dark_mode:
        background = DARK_SURFACE
        paper_background = DARK_SURFACE
        text_color = DARK_TEXT
        grid_color = "#405050"
        axis_color = DARK_TEXT_SECONDARY
    else:
        background = LIGHT_SURFACE
        paper_background = LIGHT_SURFACE
        text_color = LIGHT_TEXT
        grid_color = "#E3EAEA"
        axis_color = LIGHT_TEXT

    fig.update_layout(

        # ----------------------------------------------------
        # Background
        # ----------------------------------------------------

        paper_bgcolor=paper_background,

        plot_bgcolor=background,

        # ----------------------------------------------------
        # Typography
        # ----------------------------------------------------

        font={
            "family": "Arial, sans-serif",
            "color": text_color,
            "size": 13,
        },

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {
                "size": 18,
                "family": "Arial, sans-serif",
                "color": text_color,
            },
        },

        # ----------------------------------------------------
        # Size
        # ----------------------------------------------------

        height=430,

        margin=dict(
            l=50,
            r=30,
            t=70,
            b=50,
        ),

        hovermode="closest",

        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,

            font=dict(
                color=text_color,
            ),
        ),

        # ----------------------------------------------------
        # Hover
        # ----------------------------------------------------

        hoverlabel=dict(
            bgcolor=(
                DARK_SURFACE_ALT
                if dark_mode
                else LIGHT_PINK
            ),

            font=dict(
                color=(
                    "#FFFFFF"
                    if dark_mode
                    else "#000000"
                ),
            ),
        ),

        # ----------------------------------------------------
        # X Axis
        # ----------------------------------------------------

        xaxis=dict(
            color=axis_color,
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            linecolor=grid_color,
        ),

        # ----------------------------------------------------
        # Y Axis
        # ----------------------------------------------------

        yaxis=dict(
            color=axis_color,
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            linecolor=grid_color,
        ),
    )

    return fig


# ============================================================
# DATA CLEANING
# ============================================================

def clean_dataframe(df):
    """Basic automatic cleaning."""

    df = df.copy()

    # Remove completely empty rows
    df = df.dropna(
        axis=0,
        how="all"
    )

    # Remove completely empty columns
    df = df.dropna(
        axis=1,
        how="all"
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Try to detect date columns
    for column in df.columns:

        if df[column].dtype == "object":

            sample = (
                df[column]
                .dropna()
                .astype(str)
            )

            if len(sample) > 0:

                converted = pd.to_datetime(
                    sample,
                    errors="coerce"
                )

                valid_ratio = (
                    converted.notna().mean()
                )

                if valid_ratio >= 0.80:

                    df[column] = pd.to_datetime(
                        df[column],
                        errors="coerce"
                    )

    return df


# ============================================================
# DATA ANALYSIS
# ============================================================

def analyze_dataframe(df):
    """Analyze dataset structure."""

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    date_columns = (
        df.select_dtypes(
            include=[
                "datetime64[ns]",
                "datetime64[ns, UTC]",
            ]
        )
        .columns
        .tolist()
    )

    # Detect numeric-looking object columns
    for column in df.columns:

        if (
            column not in numeric_columns
            and column not in date_columns
        ):

            if df[column].dtype == "object":

                converted = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

                if (
                    converted.notna().mean()
                    >= 0.90
                ):

                    df[column] = converted

                    numeric_columns.append(
                        column
                    )

                    if column in categorical_columns:
                        categorical_columns.remove(
                            column
                        )

    return (
        df,
        numeric_columns,
        categorical_columns,
        date_columns,
    )


# ============================================================
# DASHBOARD STATE
# ============================================================

class State(rx.State):

    # ========================================================
    # THEME
    # ========================================================

    dark_mode: bool = False

    theme_name: str = "Light Mode"

    # ========================================================
    # UPLOAD INFORMATION
    # ========================================================

    is_uploaded: bool = False

    filename: str = ""

    error_message: str = ""

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    row_count: int = 0

    col_count: int = 0

    columns_list: list[str] = []

    numeric_columns: list[str] = []

    categorical_columns: list[str] = []

    date_columns: list[str] = []

    # ========================================================
    # KPI VALUES
    # ========================================================

    total_rows: str = "—"

    total_columns: str = "—"

    numeric_count: str = "—"

    missing_values: str = "—"

    duplicate_rows: str = "—"

    completeness: str = "—"

    # ========================================================
    # AUTOMATIC INSIGHTS
    # ========================================================

    insight_1: str = ""

    insight_2: str = ""

    insight_3: str = ""

    insight_4: str = ""

    # ========================================================
    # PLOTLY FIGURES
    # ========================================================

    chart_1: go.Figure = go.Figure()

    chart_2: go.Figure = go.Figure()

    chart_3: go.Figure = go.Figure()

    chart_4: go.Figure = go.Figure()

    chart_5: go.Figure = go.Figure()

    chart_6: go.Figure = go.Figure()

    chart_7: go.Figure = go.Figure()

    # ========================================================
    # INTERNAL DATASET PATH
    # ========================================================

    dataset_path: str = ""

    # ========================================================
    # THEME TOGGLE
    # ========================================================

    @rx.event
    def toggle_theme(self):

        self.dark_mode = not self.dark_mode

        if self.dark_mode:
            self.theme_name = "Night Mode"
        else:
            self.theme_name = "Light Mode"

        # Re-style existing charts
        charts = [
            self.chart_1,
            self.chart_2,
            self.chart_3,
            self.chart_4,
            self.chart_5,
            self.chart_6,
            self.chart_7,
        ]

        for chart in charts:

            if chart is not None:

                title = ""

                try:
                    title = chart.layout.title.text
                except Exception:
                    title = ""

                style_figure(
                    chart,
                    title,
                    self.dark_mode,
                )

    # ========================================================
    # UPLOAD HANDLER
    # ========================================================

    @rx.event
    async def handle_upload(
        self,
        files: list[rx.UploadFile],
    ):

        if not files:
            return

        self.error_message = ""

        try:

            file = files[0]

            filename = file.name

            extension = (
                Path(filename)
                .suffix
                .lower()
            )

            # ------------------------------------------------
            # Validate file type
            # ------------------------------------------------

            if extension not in [
                ".csv",
                ".xlsx",
                ".xls",
            ]:

                self.error_message = (
                    "Unsupported file type. "
                    "Please upload a CSV or Excel file."
                )

                return

            # ------------------------------------------------
            # Read uploaded file
            # ------------------------------------------------

            file_data = await file.read()

            if extension == ".csv":

                df = pd.read_csv(
                    io.BytesIO(file_data)
                )

            elif extension in [".xlsx", ".xls"]:

                df = pd.read_excel(
                    io.BytesIO(file_data)
                )

            # Make DataFrame writable
            df = df.copy(deep=True)

            # ------------------------------------------------
            # Original duplicate count
            # ------------------------------------------------

            duplicate_count = int(
                df.duplicated().sum()
            )

            # ------------------------------------------------
            # Clean dataframe
            # ------------------------------------------------

            df = clean_dataframe(df)

            # ------------------------------------------------
            # Analyze dataframe
            # ------------------------------------------------

            (
                df,
                numeric_columns,
                categorical_columns,
                date_columns,
            ) = analyze_dataframe(df)

            # ------------------------------------------------
            # Empty dataset check
            # ------------------------------------------------

            if df.empty:

                self.error_message = (
                    "The uploaded dataset appears to be empty."
                )

                return

            # ------------------------------------------------
            # Save uploaded dataset
            # ------------------------------------------------

            upload_directory = rx.get_upload_dir()

            upload_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            safe_filename = (
                "autoviziq_" + filename
            )

            dataset_file = (
                upload_directory
                / safe_filename
            )

            dataset_file.write_bytes(
                file_data
            )

            self.dataset_path = str(
                dataset_file
            )

            # ------------------------------------------------
            # Dataset information
            # ------------------------------------------------

            self.filename = filename

            self.row_count = len(df)

            self.col_count = len(
                df.columns
            )

            self.columns_list = [
                str(column)
                for column in df.columns
            ]

            self.numeric_columns = [
                str(column)
                for column in numeric_columns
            ]

            self.categorical_columns = [
                str(column)
                for column in categorical_columns
            ]

            self.date_columns = [
                str(column)
                for column in date_columns
            ]

            # ------------------------------------------------
            # KPI calculations
            # ------------------------------------------------

            missing_count = int(
                df.isna()
                .sum()
                .sum()
            )

            total_cells = (
                df.shape[0]
                * df.shape[1]
            )

            if total_cells > 0:

                completeness_value = (
                    (
                        1
                        - missing_count
                        / total_cells
                    )
                    * 100
                )

            else:

                completeness_value = 0

            self.total_rows = format_number(
                len(df)
            )

            self.total_columns = format_number(
                len(df.columns)
            )

            self.numeric_count = format_number(
                len(numeric_columns)
            )

            self.missing_values = format_number(
                missing_count
            )

            self.duplicate_rows = format_number(
                duplicate_count
            )

            self.completeness = (
                f"{completeness_value:.1f}%"
            )

            # ------------------------------------------------
            # Generate charts
            # ------------------------------------------------

            self.generate_charts(
                df,
                numeric_columns,
                categorical_columns,
                date_columns,
            )

            # ------------------------------------------------
            # Generate insights
            # ------------------------------------------------

            self.generate_insights(
                df,
                numeric_columns,
                categorical_columns,
                date_columns,
            )

            # ------------------------------------------------
            # Show dashboard
            # ------------------------------------------------

            self.is_uploaded = True

        except Exception as e:

            self.error_message = (
                f"Could not analyze dataset: {str(e)}"
            )

            self.is_uploaded = False

    # ========================================================
    # GENERATE CHARTS
    # ========================================================

    def generate_charts(
        self,
        df,
        numeric_columns,
        categorical_columns,
        date_columns,
    ):

        # ====================================================
        # CHART 1
        # NUMERIC DISTRIBUTION
        # ====================================================

        if numeric_columns:

            column = numeric_columns[0]

            fig = px.histogram(
                df,
                x=column,
                nbins=30,
                marginal="box",
                title=f"Distribution of {column}",
            )

            self.chart_1 = style_figure(
                fig,
                f"Distribution of {column}",
                self.dark_mode,
            )

        else:

            fig = go.Figure()

            fig.add_annotation(
                text="No numeric columns detected",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

            self.chart_1 = style_figure(
                fig,
                "Numeric Distribution",
                self.dark_mode,
            )

        # ====================================================
        # CHART 2
        # CATEGORY BREAKDOWN
        # ====================================================

        if categorical_columns:

            cat = categorical_columns[0]

            counts = (
                df[cat]
                .astype(str)
                .value_counts()
                .head(10)
                .reset_index()
            )

            counts.columns = [
                cat,
                "Count",
            ]

            fig = px.bar(
                counts,
                x=cat,
                y="Count",
                title=f"Top Values in {cat}",
            )

            self.chart_2 = style_figure(
                fig,
                f"Top Values in {cat}",
                self.dark_mode,
            )

        else:

            fig = go.Figure()

            fig.add_annotation(
                text="No categorical columns detected",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

            self.chart_2 = style_figure(
                fig,
                "Category Analysis",
                self.dark_mode,
            )

        # ====================================================
        # CHART 3
        # NUMERIC RELATIONSHIP
        # ====================================================

        if len(numeric_columns) >= 2:

            x_column = numeric_columns[0]

            y_column = numeric_columns[1]

            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                title=(
                    f"{x_column} vs {y_column}"
                ),
            )

            self.chart_3 = style_figure(
                fig,
                f"{x_column} vs {y_column}",
                self.dark_mode,
            )

        else:

            fig = go.Figure()

            fig.add_annotation(
                text="Need at least 2 numeric columns",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

            self.chart_3 = style_figure(
                fig,
                "Relationship Analysis",
                self.dark_mode,
            )

        # ====================================================
        # CHART 4
        # BOX PLOT / OUTLIER ANALYSIS
        # ====================================================

        if numeric_columns:

            box_column = numeric_columns[
                min(
                    1,
                    len(numeric_columns) - 1,
                )
            ]

            fig = px.box(
                df,
                y=box_column,
                points="outliers",
                title=(
                    f"Outlier Analysis — "
                    f"{box_column}"
                ),
            )

            self.chart_4 = style_figure(
                fig,
                f"Outlier Analysis — {box_column}",
                self.dark_mode,
            )

        else:

            fig = go.Figure()

            fig.add_annotation(
                text="No numeric columns available",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

            self.chart_4 = style_figure(
                fig,
                "Outlier Analysis",
                self.dark_mode,
            )

        # ====================================================
        # CHART 5
        # CORRELATION HEATMAP
        # ====================================================

        if len(numeric_columns) >= 2:

            correlation = (
                df[numeric_columns]
                .corr()
            )

            fig = px.imshow(
                correlation,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap",
            )

            self.chart_5 = style_figure(
                fig,
                "Correlation Heatmap",
                self.dark_mode,
            )

        else:

            fig = go.Figure()

            fig.add_annotation(
                text="Need at least 2 numeric columns",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

            self.chart_5 = style_figure(
                fig,
                "Correlation Heatmap",
                self.dark_mode,
            )

        # ====================================================
        # CHART 6
        # DATE TREND / CATEGORY PERFORMANCE
        # ====================================================

        if (
            date_columns
            and numeric_columns
        ):

            date_column = date_columns[0]

            value_column = numeric_columns[0]

            trend_df = df[
                [
                    date_column,
                    value_column,
                ]
            ].dropna()

            trend_df = (
                trend_df
                .sort_values(
                    date_column
                )
            )

            fig = px.line(
                trend_df,
                x=date_column,
                y=value_column,
                markers=True,
                title=(
                    f"{value_column} Over Time"
                ),
            )

            self.chart_6 = style_figure(
                fig,
                f"{value_column} Over Time",
                self.dark_mode,
            )

        elif (
            categorical_columns
            and numeric_columns
        ):

            category_column = (
                categorical_columns[0]
            )

            value_column = (
                numeric_columns[0]
            )

            grouped = (
                df.groupby(
                    category_column
                )[value_column]
                .mean()
                .sort_values(
                    ascending=False
                )
                .head(10)
                .reset_index()
            )

            fig = px.bar(
                grouped,
                x=category_column,
                y=value_column,
                title=(
                    f"Average "
                    f"{value_column} by "
                    f"{category_column}"
                ),
            )

            self.chart_6 = style_figure(
                fig,
                (
                    f"Average {value_column} "
                    f"by {category_column}"
                ),
                self.dark_mode,
            )

        else:

            fig = go.Figure()

            fig.add_annotation(
                text=(
                    "No suitable trend "
                    "variables detected"
                ),
                x=0.5,
                y=0.5,
                showarrow=False,
            )

            self.chart_6 = style_figure(
                fig,
                "Trend Analysis",
                self.dark_mode,
            )

        # ====================================================
        # CHART 7
        # CATEGORY + NUMERIC COMPOSITION
        # ====================================================

        if (
            categorical_columns
            and numeric_columns
        ):

            category_column = (
                categorical_columns[
                    min(
                        1,
                        len(
                            categorical_columns
                        ) - 1,
                    )
                ]
            )

            value_column = (
                numeric_columns[
                    min(
                        1,
                        len(numeric_columns) - 1,
                    )
                ]
            )

            grouped = (
                df.groupby(
                    category_column
                )[value_column]
                .sum()
                .sort_values(
                    ascending=False
                )
                .head(10)
                .reset_index()
            )

            fig = px.pie(
                grouped,
                names=category_column,
                values=value_column,
                hole=0.45,
                title=(
                    f"{value_column} by "
                    f"{category_column}"
                ),
            )

            self.chart_7 = style_figure(
                fig,
                (
                    f"{value_column} by "
                    f"{category_column}"
                ),
                self.dark_mode,
            )

        else:

            fig = go.Figure()

            fig.add_annotation(
                text=(
                    "No categorical + numeric "
                    "combination detected"
                ),
                x=0.5,
                y=0.5,
                showarrow=False,
            )

            self.chart_7 = style_figure(
                fig,
                "Composition Analysis",
                self.dark_mode,
            )

    # ========================================================
    # AUTOMATIC INSIGHTS
    # ========================================================

    def generate_insights(
        self,
        df,
        numeric_columns,
        categorical_columns,
        date_columns,
    ):

        insights = []

        # ----------------------------------------------------
        # Insight 1
        # ----------------------------------------------------

        if numeric_columns:

            column = numeric_columns[0]

            average = df[column].mean()

            median = df[column].median()

            insights.append(
                f"{column} has an average of "
                f"{format_number(average)} "
                f"and a median of "
                f"{format_number(median)}."
            )

        else:

            insights.append(
                "No numeric variables were "
                "detected for statistical analysis."
            )

        # ----------------------------------------------------
        # Insight 2
        # ----------------------------------------------------

        if categorical_columns:

            column = categorical_columns[0]

            counts = (
                df[column]
                .astype(str)
                .value_counts()
            )

            if len(counts) > 0:

                top_category = counts.index[0]

                insights.append(
                    f"The most frequent value "
                    f"in {column} is "
                    f"'{top_category}' with "
                    f"{counts.iloc[0]:,} records."
                )

        else:

            insights.append(
                "No categorical variables "
                "were detected."
            )

        # ----------------------------------------------------
        # Insight 3
        # ----------------------------------------------------

        missing = int(
            df.isna()
            .sum()
            .sum()
        )

        if missing == 0:

            insights.append(
                "The dataset contains "
                "no missing values."
            )

        else:

            insights.append(
                f"The dataset contains "
                f"{missing:,} missing values."
            )

        # ----------------------------------------------------
        # Insight 4
        # ----------------------------------------------------

        if len(numeric_columns) >= 2:

            corr = (
                df[numeric_columns]
                .corr()
                .abs()
            )

            corr_values = corr.to_numpy(
                copy=True
            )

            np.fill_diagonal(
                corr_values,
                0,
            )

            max_corr = corr_values.max()

            location = np.where(
                corr_values == max_corr
            )

            if len(location[0]) > 0:

                row = location[0][0]

                col = location[1][0]

                column_a = (
                    numeric_columns[row]
                )

                column_b = (
                    numeric_columns[col]
                )

                insights.append(
                    f"The strongest numeric "
                    f"relationship is between "
                    f"{column_a} and {column_b} "
                    f"(absolute correlation ≈ "
                    f"{max_corr:.2f})."
                )

            else:

                insights.append(
                    "No strong numeric "
                    "relationships were detected."
                )

        else:

            insights.append(
                "More numeric variables are "
                "needed for relationship analysis."
            )

        # ----------------------------------------------------
        # Guarantee four insights
        # ----------------------------------------------------

        while len(insights) < 4:

            insights.append(
                "AutoVizIQ did not detect "
                "another significant pattern."
            )

        self.insight_1 = insights[0]

        self.insight_2 = insights[1]

        self.insight_3 = insights[2]

        self.insight_4 = insights[3]

    # ========================================================
    # RESET DASHBOARD
    # ========================================================

    @rx.event
    def reset_dashboard(self):

        self.is_uploaded = False

        self.filename = ""

        self.error_message = ""

        self.row_count = 0

        self.col_count = 0

        self.columns_list = []

        self.numeric_columns = []

        self.categorical_columns = []

        self.date_columns = []

        self.dataset_path = ""

        self.total_rows = "—"

        self.total_columns = "—"

        self.numeric_count = "—"

        self.missing_values = "—"

        self.duplicate_rows = "—"

        self.completeness = "—"

        self.insight_1 = ""

        self.insight_2 = ""

        self.insight_3 = ""

        self.insight_4 = ""

        self.chart_1 = go.Figure()

        self.chart_2 = go.Figure()

        self.chart_3 = go.Figure()

        self.chart_4 = go.Figure()

        self.chart_5 = go.Figure()

        self.chart_6 = go.Figure()

        self.chart_7 = go.Figure()


# ============================================================
# UI COMPONENTS
# ============================================================


# ============================================================
# THEME BUTTON
# ============================================================

def theme_toggle():

    return rx.button(

        rx.cond(
            State.dark_mode,

            rx.icon(
                tag="sun",
                size=18,
            ),

            rx.icon(
                tag="moon",
                size=18,
            ),
        ),

        rx.cond(
            State.dark_mode,
            "Light Mode",
            "Night Mode",
        ),

        on_click=State.toggle_theme,

        variant="outline",

        border_radius="12px",

        padding="10px 16px",

        background=rx.cond(
            State.dark_mode,
            DARK_SURFACE_ALT,
            LIGHT_SURFACE,
        ),

        color=rx.cond(
            State.dark_mode,
            DARK_TEXT,
            LIGHT_TEXT,
        ),

        border=rx.cond(
            State.dark_mode,
            f"1px solid {DARK_BORDER}",
            f"1px solid {LIGHT_BORDER}",
        ),

        _hover={
            "background": rx.cond(
                State.dark_mode,
                DARK_PINK,
                LIGHT_PINK,
            ),

            "color": "#000000",
        },
    )


# ============================================================
# KPI CARD
# ============================================================

def kpi_card(
    title,
    value,
    icon,
):

    return rx.box(

        rx.hstack(

            rx.box(

                rx.icon(
                    tag=icon,
                    size=24,

                    color=rx.cond(
                        State.dark_mode,
                        DARK_TEXT,
                        LIGHT_TEXT,
                    ),
                ),

                padding="12px",

                border_radius="12px",

                background=rx.cond(
                    State.dark_mode,
                    DARK_PINK,
                    LIGHT_PINK,
                ),
            ),

            rx.vstack(

                rx.text(
                    title,

                    size="2",

                    color=rx.cond(
                        State.dark_mode,
                        DARK_TEXT_SECONDARY,
                        LIGHT_TEXT,
                    ),
                ),

                rx.text(
                    value,

                    size="6",

                    weight="bold",

                    color=rx.cond(
                        State.dark_mode,
                        DARK_TEXT,
                        LIGHT_TEXT,
                    ),
                ),

                align="start",

                spacing="1",
            ),

            spacing="4",

            align="center",
        ),

        padding="20px",

        border=rx.cond(
            State.dark_mode,
            f"1px solid {DARK_BORDER}",
            f"1px solid {LIGHT_BORDER}",
        ),

        border_radius="16px",

        background=rx.cond(
            State.dark_mode,
            DARK_SURFACE,
            LIGHT_SURFACE,
        ),

        width="100%",

        box_shadow=rx.cond(
            State.dark_mode,
            "0 8px 25px rgba(0,0,0,0.25)",
            "0 4px 15px rgba(93,107,107,0.08)",
        ),
    )


# ============================================================
# CHART CARD
# ============================================================

def chart_card(
    title,
    chart,
):

    return rx.box(

        rx.vstack(

            rx.hstack(

                rx.text(
                    title,

                    size="4",

                    weight="bold",

                    color=rx.cond(
                        State.dark_mode,
                        DARK_TEXT,
                        LIGHT_TEXT,
                    ),
                ),

                rx.spacer(),

                rx.icon(
                    tag="chart-no-axes-combined",
                    size=20,

                    color=rx.cond(
                        State.dark_mode,
                        DARK_MINT,
                        "#5D6B6B",
                    ),
                ),

                width="100%",

                align="center",
            ),

            rx.plotly(
                data=chart,
                use_resize_handler=True,
                width="100%",
            ),

            width="100%",

            spacing="3",
        ),

        padding="18px",

        border=rx.cond(
            State.dark_mode,
            f"1px solid {DARK_BORDER}",
            f"1px solid {LIGHT_BORDER}",
        ),

        border_radius="18px",

        background=rx.cond(
            State.dark_mode,
            DARK_SURFACE,
            LIGHT_SURFACE,
        ),

        width="100%",

        box_shadow=rx.cond(
            State.dark_mode,
            "0 8px 30px rgba(0,0,0,0.22)",
            "0 4px 15px rgba(93,107,107,0.07)",
        ),
    )


# ============================================================
# UPLOAD SCREEN
# ============================================================

def upload_screen():

    return rx.box(

        rx.center(

            rx.vstack(

                # =================================================
                # BRAND
                # =================================================

                rx.vstack(

                    rx.box(

                        rx.icon(
                            tag="bar-chart-3",
                            size=42,

                            color="#000000",
                        ),

                        padding="18px",

                        border_radius="20px",

                        background=LIGHT_PINK,
                    ),

                    rx.heading(
                        "AutoVizIQ",

                        size="9",

                        weight="bold",

                        color=rx.cond(
                            State.dark_mode,
                            DARK_TEXT,
                            LIGHT_TEXT,
                        ),
                    ),

                    rx.text(
                        (
                            "Automatic Data Intelligence "
                            "& Visualization"
                        ),

                        size="4",

                        color=rx.cond(
                            State.dark_mode,
                            DARK_TEXT_SECONDARY,
                            LIGHT_TEXT,
                        ),
                    ),

                    rx.text(
                        "From Data to Decisions, Automatically.",

                        size="3",

                        color=rx.cond(
                            State.dark_mode,
                            DARK_TEXT_SECONDARY,
                            LIGHT_TEXT_SECONDARY,
                        ),
                    ),

                    align="center",

                    spacing="3",
                ),

                # =================================================
                # THEME BUTTON
                # =================================================

                theme_toggle(),

                # =================================================
                # UPLOAD AREA
                # =================================================

                rx.upload(

                    rx.vstack(

                        rx.box(

                            rx.icon(
                                tag="cloud-upload",
                                size=50,

                                color="#000000",
                            ),

                            padding="18px",

                            border_radius="18px",

                            background=LIGHT_PINK,
                        ),

                        rx.text(
                            "Drop your dataset here",

                            size="5",

                            weight="bold",

                            color=rx.cond(
                                State.dark_mode,
                                DARK_TEXT,
                                LIGHT_TEXT,
                            ),
                        ),

                        rx.text(
                            "or click to browse files",

                            size="3",

                            color=rx.cond(
                                State.dark_mode,
                                DARK_TEXT_SECONDARY,
                                LIGHT_TEXT,
                            ),
                        ),

                        rx.text(
                            "CSV • XLSX • XLS",

                            size="2",

                            color=rx.cond(
                                State.dark_mode,
                                DARK_TEXT_SECONDARY,
                                LIGHT_TEXT_SECONDARY,
                            ),
                        ),

                        align="center",

                        spacing="3",
                    ),

                    id="dataset_upload",

                    accept={
                        "text/csv": [".csv"],

                        (
                            "application/"
                            "vnd.openxmlformats-"
                            "officedocument."
                            "spreadsheetml.sheet"
                        ): [".xlsx"],

                        "application/vnd.ms-excel": [
                            ".xls"
                        ],
                    },

                    max_files=1,

                    border=rx.cond(
                        State.dark_mode,
                        f"2px dashed {DARK_BORDER}",
                        f"2px dashed {LIGHT_TAUPE}",
                    ),

                    border_radius="20px",

                    padding="70px 40px",

                    width="100%",

                    background=rx.cond(
                        State.dark_mode,
                        DARK_SURFACE_ALT,
                        LIGHT_SURFACE_ALT,
                    ),
                ),

                # =================================================
                # ANALYZE BUTTON
                # =================================================

                rx.button(

                    rx.icon(
                        tag="upload",
                        size=20,
                    ),

                    "Analyze Dataset",

                    on_click=State.handle_upload(
                        rx.upload_files(
                            upload_id="dataset_upload"
                        )
                    ),

                    size="4",

                    width="100%",

                    padding="25px",

                    border_radius="14px",

                    background=rx.cond(
                        State.dark_mode,
                        DARK_PINK,
                        LIGHT_PINK,
                    ),

                    color="#000000",

                    _hover={
                        "background": rx.cond(
                            State.dark_mode,
                            "#E5AAAA",
                            "#EAB5B5",
                        ),
                    },
                ),

                # =================================================
                # ERROR
                # =================================================

                rx.cond(

                    State.error_message != "",

                    rx.box(

                        rx.text(
                            State.error_message,

                            color=rx.cond(
                                State.dark_mode,
                                DARK_TEXT,
                                LIGHT_TEXT,
                            ),
                        ),

                        padding="15px",

                        border_radius="10px",

                        background=rx.cond(
                            State.dark_mode,
                            DARK_PINK,
                            LIGHT_PINK,
                        ),

                        width="100%",
                    ),
                ),

                width="650px",

                max_width="90vw",

                spacing="5",
            ),

            min_height="100vh",

            padding="40px",
        ),

        width="100%",

        min_height="100vh",

        background=rx.cond(
            State.dark_mode,
            DARK_BG,
            LIGHT_BG,
        ),
    )


# ============================================================
# MAIN DASHBOARD
# ============================================================

def dashboard():

    return rx.box(

        rx.vstack(

            # =================================================
            # HEADER
            # =================================================

            rx.hstack(

                rx.vstack(

                    rx.heading(
                        "AutoVizIQ",

                        size="8",

                        weight="bold",

                        color=rx.cond(
                            State.dark_mode,
                            DARK_TEXT,
                            LIGHT_TEXT,
                        ),
                    ),

                    rx.text(
                        (
                            "Automatic Data Intelligence "
                            "Dashboard"
                        ),

                        size="3",

                        color=rx.cond(
                            State.dark_mode,
                            DARK_TEXT_SECONDARY,
                            LIGHT_TEXT_SECONDARY,
                        ),
                    ),

                    align="start",

                    spacing="1",
                ),

                rx.spacer(),

                rx.hstack(

                    # Theme button
                    theme_toggle(),

                    # Dataset filename
                    rx.box(

                        rx.icon(
                            tag="file-spreadsheet",
                            size=18,

                            color=rx.cond(
                                State.dark_mode,
                                DARK_TEXT,
                                LIGHT_TEXT,
                            ),
                        ),

                        rx.text(
                            State.filename,

                            size="2",

                            weight="medium",

                            color=rx.cond(
                                State.dark_mode,
                                DARK_TEXT,
                                LIGHT_TEXT,
                            ),
                        ),

                        padding="10px 14px",

                        border_radius="10px",

                        background=rx.cond(
                            State.dark_mode,
                            DARK_SURFACE_ALT,
                            LIGHT_SURFACE_ALT,
                        ),
                    ),

                    # New dataset
                    rx.button(

                        rx.icon(
                            tag="upload",
                            size=18,
                        ),

                        "New Dataset",

                        on_click=State.reset_dashboard,

                        variant="outline",

                        color=rx.cond(
                            State.dark_mode,
                            DARK_TEXT,
                            LIGHT_TEXT,
                        ),

                        border_color=rx.cond(
                            State.dark_mode,
                            DARK_BORDER,
                            LIGHT_BORDER,
                        ),
                    ),

                    spacing="3",
                ),

                width="100%",

                align="center",
            ),

            # =================================================
            # DATASET PROFILE
            # =================================================

            rx.box(

                rx.vstack(

                    rx.hstack(

                        rx.vstack(

                            rx.text(
                                "DATASET PROFILE",

                                size="2",

                                weight="bold",

                                color=rx.cond(
                                    State.dark_mode,
                                    DARK_MINT,
                                    "#5D6B6B",
                                ),
                            ),

                            rx.text(
                                (
                                    "Auto-generated analysis "
                                    "of your dataset"
                                ),

                                size="2",

                                color=rx.cond(
                                    State.dark_mode,
                                    DARK_TEXT_SECONDARY,
                                    LIGHT_TEXT_SECONDARY,
                                ),
                            ),

                            align="start",
                        ),

                        rx.spacer(),

                        rx.badge(

                            "ANALYZED",

                            variant="soft",

                            background=rx.cond(
                                State.dark_mode,
                                DARK_MINT,
                                LIGHT_MINT,
                            ),

                            color="#000000",
                        ),

                        width="100%",

                        align="center",
                    ),

                    rx.hstack(

                        rx.text(
                            "Rows: ",

                            weight="bold",

                            color=rx.cond(
                                State.dark_mode,
                                DARK_TEXT,
                                LIGHT_TEXT,
                            ),
                        ),

                        rx.text(
                            State.total_rows,

                            color=rx.cond(
                                State.dark_mode,
                                DARK_TEXT,
                                LIGHT_TEXT,
                            ),
                        ),

                        rx.text(
                            " • Columns: ",

                            weight="bold",

                            color=rx.cond(
                                State.dark_mode,
                                DARK_TEXT,
                                LIGHT_TEXT,
                            ),
                        ),

                        rx.text(
                            State.total_columns,

                            color=rx.cond(
                                State.dark_mode,
                                DARK_TEXT,
                                LIGHT_TEXT,
                            ),
                        ),

                        rx.text(
                            " • Numeric: ",

                            weight="bold",

                            color=rx.cond(
                                State.dark_mode,
                                DARK_TEXT,
                                LIGHT_TEXT,
                            ),
                        ),

                        rx.text(
                            State.numeric_count,

                            color=rx.cond(
                                State.dark_mode,
                                DARK_TEXT,
                                LIGHT_TEXT,
                            ),
                        ),

                        width="100%",

                        wrap="wrap",
                    ),

                    spacing="3",

                    width="100%",
                ),

                padding="20px",

                border_radius="16px",

                background=rx.cond(
                    State.dark_mode,
                    DARK_SURFACE_ALT,
                    LIGHT_SURFACE_ALT,
                ),

                border=rx.cond(
                    State.dark_mode,
                    f"1px solid {DARK_BORDER}",
                    f"1px solid {LIGHT_BORDER}",
                ),

                width="100%",
            ),

            # =================================================
            # KPI CARDS
            # =================================================

            rx.grid(

                kpi_card(
                    "Total Rows",
                    State.total_rows,
                    "database",
                ),

                kpi_card(
                    "Total Columns",
                    State.total_columns,
                    "columns-3",
                ),

                kpi_card(
                    "Numeric Variables",
                    State.numeric_count,
                    "hash",
                ),

                kpi_card(
                    "Missing Values",
                    State.missing_values,
                    "circle-alert",
                ),

                kpi_card(
                    "Duplicate Rows",
                    State.duplicate_rows,
                    "copy",
                ),

                kpi_card(
                    "Data Completeness",
                    State.completeness,
                    "circle-check",
                ),

                columns=(
                    "repeat(6, minmax(150px, 1fr))"
                ),

                spacing="4",

                width="100%",
            ),

            # =================================================
            # AUTOMATIC INSIGHTS
            # =================================================

            rx.box(

                rx.vstack(

                    rx.hstack(

                        rx.icon(
                            tag="sparkles",
                            size=24,

                            color=rx.cond(
                                State.dark_mode,
                                DARK_PINK,
                                "#000000",
                            ),
                        ),

                        rx.text(
                            "Auto Insights",

                            size="5",

                            weight="bold",

                            color=rx.cond(
                                State.dark_mode,
                                DARK_TEXT,
                                LIGHT_TEXT,
                            ),
                        ),

                        spacing="3",

                        align="center",
                    ),

                    rx.grid(

                        rx.box(

                            rx.text(
                                State.insight_1,

                                size="3",

                                color=rx.cond(
                                    State.dark_mode,
                                    DARK_TEXT,
                                    LIGHT_TEXT,
                                ),
                            ),

                            padding="18px",

                            border_radius="12px",

                            background=rx.cond(
                                State.dark_mode,
                                DARK_SURFACE_ALT,
                                LIGHT_SURFACE_ALT,
                            ),
                        ),

                        rx.box(

                            rx.text(
                                State.insight_2,

                                size="3",

                                color=rx.cond(
                                    State.dark_mode,
                                    DARK_TEXT,
                                    LIGHT_TEXT,
                                ),
                            ),

                            padding="18px",

                            border_radius="12px",

                            background=rx.cond(
                                State.dark_mode,
                                DARK_SURFACE_ALT,
                                LIGHT_SURFACE_ALT,
                            ),
                        ),

                        rx.box(

                            rx.text(
                                State.insight_3,

                                size="3",

                                color=rx.cond(
                                    State.dark_mode,
                                    DARK_TEXT,
                                    LIGHT_TEXT,
                                ),
                            ),

                            padding="18px",

                            border_radius="12px",

                            background=rx.cond(
                                State.dark_mode,
                                DARK_SURFACE_ALT,
                                LIGHT_SURFACE_ALT,
                            ),
                        ),

                        rx.box(

                            rx.text(
                                State.insight_4,

                                size="3",

                                color=rx.cond(
                                    State.dark_mode,
                                    DARK_TEXT,
                                    LIGHT_TEXT,
                                ),
                            ),

                            padding="18px",

                            border_radius="12px",

                            background=rx.cond(
                                State.dark_mode,
                                DARK_SURFACE_ALT,
                                LIGHT_SURFACE_ALT,
                            ),
                        ),

                        columns=(
                            "repeat(4, minmax(0, 1fr))"
                        ),

                        spacing="4",

                        width="100%",
                    ),

                    width="100%",

                    spacing="4",
                ),

                padding="24px",

                border_radius="18px",

                border=rx.cond(
                    State.dark_mode,
                    f"1px solid {DARK_BORDER}",
                    f"1px solid {LIGHT_BORDER}",
                ),

                background=rx.cond(
                    State.dark_mode,
                    DARK_SURFACE,
                    LIGHT_SURFACE,
                ),

                width="100%",
            ),

            # =================================================
            # VISUAL ANALYTICS HEADER
            # =================================================

            rx.hstack(

                rx.vstack(

                    rx.text(
                        "Visual Analytics",

                        size="6",

                        weight="bold",

                        color=rx.cond(
                            State.dark_mode,
                            DARK_TEXT,
                            LIGHT_TEXT,
                        ),
                    ),

                    rx.text(
                        (
                            "Interactive charts automatically "
                            "selected from your dataset"
                        ),

                        size="3",

                        color=rx.cond(
                            State.dark_mode,
                            DARK_TEXT_SECONDARY,
                            LIGHT_TEXT_SECONDARY,
                        ),
                    ),

                    align="start",

                    spacing="1",
                ),

                rx.spacer(),

                rx.badge(

                    "7 INTERACTIVE VISUALS",

                    variant="soft",

                    background=rx.cond(
                        State.dark_mode,
                        DARK_PINK,
                        LIGHT_PINK,
                    ),

                    color="#000000",
                ),

                width="100%",

                align="center",
            ),

            # =================================================
            # CHART ROW 1
            # =================================================

            rx.grid(

                chart_card(
                    "Distribution",
                    State.chart_1,
                ),

                chart_card(
                    "Category Breakdown",
                    State.chart_2,
                ),

                columns=(
                    "repeat(2, minmax(0, 1fr))"
                ),

                spacing="5",

                width="100%",
            ),

            # =================================================
            # CHART ROW 2
            # =================================================

            rx.grid(

                chart_card(
                    "Relationship Analysis",
                    State.chart_3,
                ),

                chart_card(
                    "Outlier Analysis",
                    State.chart_4,
                ),

                columns=(
                    "repeat(2, minmax(0, 1fr))"
                ),

                spacing="5",

                width="100%",
            ),

            # =================================================
            # CHART ROW 3
            # =================================================

            rx.grid(

                chart_card(
                    "Correlation Analysis",
                    State.chart_5,
                ),

                chart_card(
                    "Trend / Performance Analysis",
                    State.chart_6,
                ),

                columns=(
                    "repeat(2, minmax(0, 1fr))"
                ),

                spacing="5",

                width="100%",
            ),

            # =================================================
            # CHART 7
            # =================================================

            chart_card(
                "Composition Analysis",
                State.chart_7,
            ),

            # =================================================
            # DETECTED DATASET STRUCTURE
            # =================================================

            rx.box(

                rx.vstack(

                    rx.text(
                        "Detected Dataset Structure",

                        size="5",

                        weight="bold",

                        color=rx.cond(
                            State.dark_mode,
                            DARK_TEXT,
                            LIGHT_TEXT,
                        ),
                    ),

                    rx.grid(

                        # -----------------------------------------
                        # Numeric columns
                        # -----------------------------------------

                        rx.box(

                            rx.vstack(

                                rx.text(
                                    "Numeric Columns",

                                    weight="bold",

                                    color=rx.cond(
                                        State.dark_mode,
                                        DARK_TEXT,
                                        LIGHT_TEXT,
                                    ),
                                ),

                                rx.foreach(
                                    State.numeric_columns,

                                    lambda column: rx.badge(

                                        column,

                                        variant="soft",

                                        background=rx.cond(
                                            State.dark_mode,
                                            DARK_MINT,
                                            LIGHT_MINT,
                                        ),

                                        color="#000000",
                                    ),
                                ),

                                align="start",

                                spacing="2",
                            ),

                            padding="20px",

                            border_radius="14px",

                            background=rx.cond(
                                State.dark_mode,
                                DARK_SURFACE_ALT,
                                LIGHT_SURFACE_ALT,
                            ),
                        ),

                        # -----------------------------------------
                        # Categorical columns
                        # -----------------------------------------

                        rx.box(

                            rx.vstack(

                                rx.text(
                                    "Categorical Columns",

                                    weight="bold",

                                    color=rx.cond(
                                        State.dark_mode,
                                        DARK_TEXT,
                                        LIGHT_TEXT,
                                    ),
                                ),

                                rx.foreach(
                                    State.categorical_columns,

                                    lambda column: rx.badge(

                                        column,

                                        variant="soft",

                                        background=rx.cond(
                                            State.dark_mode,
                                            DARK_PINK,
                                            LIGHT_PINK,
                                        ),

                                        color="#000000",
                                    ),
                                ),

                                align="start",

                                spacing="2",
                            ),

                            padding="20px",

                            border_radius="14px",

                            background=rx.cond(
                                State.dark_mode,
                                DARK_SURFACE_ALT,
                                LIGHT_SURFACE_ALT,
                            ),
                        ),

                        # -----------------------------------------
                        # Date columns
                        # -----------------------------------------

                        rx.box(

                            rx.vstack(

                                rx.text(
                                    "Date Columns",

                                    weight="bold",

                                    color=rx.cond(
                                        State.dark_mode,
                                        DARK_TEXT,
                                        LIGHT_TEXT,
                                    ),
                                ),

                                rx.foreach(
                                    State.date_columns,

                                    lambda column: rx.badge(

                                        column,

                                        variant="soft",

                                        background=rx.cond(
                                            State.dark_mode,
                                            DARK_TAUPE,
                                            LIGHT_TAUPE,
                                        ),

                                        color="#000000",
                                    ),
                                ),

                                align="start",

                                spacing="2",
                            ),

                            padding="20px",

                            border_radius="14px",

                            background=rx.cond(
                                State.dark_mode,
                                DARK_SURFACE_ALT,
                                LIGHT_SURFACE_ALT,
                            ),
                        ),

                        columns=(
                            "repeat(3, minmax(0, 1fr))"
                        ),

                        spacing="4",

                        width="100%",
                    ),

                    width="100%",

                    spacing="4",
                ),

                padding="24px",

                border_radius="18px",

                border=rx.cond(
                    State.dark_mode,
                    f"1px solid {DARK_BORDER}",
                    f"1px solid {LIGHT_BORDER}",
                ),

                background=rx.cond(
                    State.dark_mode,
                    DARK_SURFACE,
                    LIGHT_SURFACE,
                ),

                width="100%",
            ),

            # =================================================
            # FOOTER
            # =================================================

            rx.center(

                rx.text(
                    (
                        "AutoVizIQ • "
                        "Automated Data Intelligence"
                    ),

                    size="2",

                    color=rx.cond(
                        State.dark_mode,
                        DARK_TEXT_SECONDARY,
                        LIGHT_TEXT_SECONDARY,
                    ),
                ),

                padding="20px",
            ),

            # =================================================
            # GLOBAL DASHBOARD WIDTH
            # =================================================

            width="100%",

            max_width="1600px",

            margin="0 auto",

            padding="35px",

            spacing="6",
        ),

        width="100%",

        min_height="100vh",

        background=rx.cond(
            State.dark_mode,
            DARK_BG,
            LIGHT_BG,
        ),

        color=rx.cond(
            State.dark_mode,
            DARK_TEXT,
            LIGHT_TEXT,
        ),

        transition="background 0.3s ease",
    )


# ============================================================
# MAIN PAGE
# ============================================================

def index():

    return rx.cond(
        State.is_uploaded,
        dashboard(),
        upload_screen(),
    )


# ============================================================
# APP
# ============================================================

app = rx.App()

app.add_page(
    index,
    title=(
        "AutoVizIQ | "
        "Automated Data Intelligence"
    ),
)