# autoviziq_engine/data_cleaner.py

import re
import numpy as np
import pandas as pd


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Trim unnecessary whitespace in text columns."""
    df = df.copy()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

        df[col] = df[col].replace("", np.nan)

    return df


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    """Identify columns that appear to contain dates."""
    date_columns = []

    for col in df.columns:
        series = df[col].dropna()

        if series.empty:
            continue

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue

        name = str(col).lower()
        date_name_hint = any(
            word in name
            for word in ["date", "time", "timestamp", "datetime"]
        )

        if (
            not pd.api.types.is_object_dtype(df[col])
            and not pd.api.types.is_string_dtype(df[col])
        ):
            continue

        sample = series.astype(str).head(500)

        # Avoid treating ordinary numeric strings as dates.
        date_pattern = sample.str.contains(
            r"[-/:]|[A-Za-z]{3,}",
            regex=True,
        ).mean()

        if not date_name_hint and date_pattern < 0.5:
            continue

        try:
            parsed = pd.to_datetime(
                sample,
                errors="coerce",
                format="mixed",
            )
            success_rate = parsed.notna().mean()

            if success_rate >= 0.80:
                date_columns.append(col)
        except (ValueError, TypeError, OverflowError):
            continue

    return date_columns


def detect_numeric_like_columns(
    df: pd.DataFrame,
    date_columns: list[str] | None = None,
) -> list[str]:
    """Find text columns that mostly contain numeric values."""
    date_columns = date_columns or []
    numeric_columns = []

    for col in df.columns:
        if col in date_columns:
            continue

        if not (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            continue

        series = df[col].dropna()

        if series.empty:
            continue

        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )

        converted = pd.to_numeric(cleaned, errors="coerce")

        if converted.notna().mean() >= 0.90:
            numeric_columns.append(col)

    return numeric_columns


def detect_identifier_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns that look like identifiers rather than measurements."""
    identifier_columns = []

    for col in df.columns:
        name = str(col).lower()
        series = df[col].dropna()

        name_hint = any(
            token in name
            for token in [
                "id",
                "uuid",
                "identifier",
                "customer_code",
                "order_code",
                "account_number",
            ]
        )

        unique_ratio = (
            series.nunique(dropna=True) / len(series)
            if len(series)
            else 0
        )

        if name_hint or (
            unique_ratio >= 0.95
            and (
                pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])
            )
        ):
            identifier_columns.append(col)

    return identifier_columns


def detect_constant_columns(df: pd.DataFrame) -> list[str]:
    """Find columns with one or zero distinct non-missing values."""
    return [
        col
        for col in df.columns
        if df[col].nunique(dropna=True) <= 1
    ]


def detect_outliers(df: pd.DataFrame) -> dict:
    """Flag possible numeric outliers using the IQR method."""
    results = {}

    identifier_columns = set(detect_identifier_columns(df))

    for col in df.select_dtypes(include=np.number).columns:
        if col in identifier_columns:
            continue

        series = df[col].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            results[col] = 0
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        results[col] = int(
            ((series < lower_bound) | (series > upper_bound)).sum()
        )

    return results


def analyze_data_quality(df: pd.DataFrame) -> dict:
    """Return a quality summary for the supplied dataset."""
    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "empty_rows": int(df.isna().all(axis=1).sum()),
        "empty_columns": int(df.isna().all(axis=0).sum()),
        "missing_values": missing_cells,
        "completeness": (
            round((1 - missing_cells / total_cells) * 100, 2)
            if total_cells
            else 100.0
        ),
        "column_details": {
            str(col): {
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "unique": int(df[col].nunique(dropna=True)),
            }
            for col in df.columns
        },
    }


def clean_dataframe(
    df: pd.DataFrame,
    remove_high_missing: bool = True,
    high_missing_threshold: float = 80.0,
    remove_constant_columns: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Clean a DataFrame and return (cleaned_dataframe, cleaning_report).

    High-missingness columns can be removed when enabled.
    Outliers and repeated identifiers are reported, not deleted.
    Missing dates are preserved rather than guessed.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The uploaded dataset contains no data.")

    if not 0 <= high_missing_threshold <= 100:
        raise ValueError("high_missing_threshold must be between 0 and 100.")

    original_rows = len(df)
    original_columns = len(df.columns)
    log = []
    cleaned = df.copy()

    # Remove columns with blank names and make duplicate names unique.
    renamed_columns = []
    seen_names = {}

    for index, col in enumerate(cleaned.columns):
        name = str(col).strip() or f"Unnamed_{index + 1}"

        if name in seen_names:
            seen_names[name] += 1
            name = f"{name}_{seen_names[name]}"
        else:
            seen_names[name] = 1

        renamed_columns.append(name)

    if list(cleaned.columns) != renamed_columns:
        cleaned.columns = renamed_columns
        log.append("Standardized blank or duplicate column names.")

    # Standardize text.
    cleaned = normalize_text(cleaned)
    log.append("Trimmed whitespace and standardized blank text values.")

    # Remove fully empty rows and columns.
    empty_rows = int(cleaned.isna().all(axis=1).sum())

    if empty_rows:
        cleaned = cleaned.loc[~cleaned.isna().all(axis=1)].copy()
        log.append(f"Removed {empty_rows} completely empty row(s).")

    empty_columns = [
        col for col in cleaned.columns if cleaned[col].isna().all()
    ]

    if empty_columns:
        cleaned = cleaned.drop(columns=empty_columns)
        log.append(
            f"Removed {len(empty_columns)} completely empty column(s)."
        )

    if cleaned.empty or len(cleaned.columns) == 0:
        raise ValueError(
            "No usable data remains after removing completely empty rows "
            "and columns."
        )

    # Convert likely date columns.
    date_columns = detect_date_columns(cleaned)

    for col in date_columns:
        try:
            cleaned[col] = pd.to_datetime(
                cleaned[col],
                errors="coerce",
                format="mixed",
            )
        except (ValueError, TypeError, OverflowError):
            continue

    if date_columns:
        log.append(
            f"Detected date/time column(s): {', '.join(map(str, date_columns))}."
        )

    # Convert numeric-like text columns.
    numeric_like_columns = detect_numeric_like_columns(
        cleaned,
        date_columns=date_columns,
    )

    for col in numeric_like_columns:
        cleaned[col] = (
            cleaned[col]
            .astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if numeric_like_columns:
        log.append(
            "Converted numeric-looking text column(s): "
            + ", ".join(map(str, numeric_like_columns))
            + "."
        )

    # Optionally remove columns with excessive missing values.
    high_missing_columns = []

    if remove_high_missing and len(cleaned) > 0:
        missing_percent = cleaned.isna().mean() * 100
        high_missing_columns = [
            col
            for col, percent in missing_percent.items()
            if percent > high_missing_threshold
        ]

        if high_missing_columns:
            cleaned = cleaned.drop(columns=high_missing_columns)
            log.append(
                f"Removed {len(high_missing_columns)} column(s) with more "
                f"than {high_missing_threshold:g}% missing values."
            )

    if cleaned.empty or len(cleaned.columns) == 0:
        raise ValueError(
            "No usable columns remain after applying the missing-value rule."
        )

    # Remove exact duplicate rows.
    duplicates_before = int(cleaned.duplicated().sum())

    if duplicates_before:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        log.append(f"Removed {duplicates_before} duplicate row(s).")
    else:
        cleaned = cleaned.reset_index(drop=True)

    # Detect IDs before filling missing values.
    identifier_columns = set(detect_identifier_columns(cleaned))
    missing_values_filled = 0

    # Fill numeric missing values with the median, excluding ID columns.
    numeric_columns = cleaned.select_dtypes(include=np.number).columns

    for col in numeric_columns:
        if col in identifier_columns:
            continue

        missing_count = int(cleaned[col].isna().sum())

        if missing_count == 0:
            continue

        median_value = cleaned[col].median()

        if pd.notna(median_value):
            cleaned[col] = cleaned[col].fillna(median_value)
            missing_values_filled += missing_count

    # Fill categorical missing values with the mode.
    categorical_columns = cleaned.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns

    for col in categorical_columns:
        if col in identifier_columns:
            continue

        missing_count = int(cleaned[col].isna().sum())

        if missing_count == 0:
            continue

        modes = cleaned[col].mode(dropna=True)

        if not modes.empty:
            cleaned[col] = cleaned[col].fillna(modes.iloc[0])
            missing_values_filled += missing_count

    if missing_values_filled:
        log.append(
            f"Filled {missing_values_filled} missing value(s) using "
            "numeric medians or categorical modes."
        )

    # Optionally remove constant columns.
    constant_columns = detect_constant_columns(cleaned)
    removed_constant_columns = []

    if remove_constant_columns and constant_columns:
        removed_constant_columns = constant_columns
        cleaned = cleaned.drop(columns=removed_constant_columns)
        log.append(
            f"Removed {len(removed_constant_columns)} constant column(s)."
        )
    elif constant_columns:
        log.append(
            f"Detected {len(constant_columns)} constant column(s); "
            "kept them by default."
        )

    # Recalculate date, ID, and outlier information after cleaning.
    remaining_date_columns = [
        col
        for col in cleaned.columns
        if pd.api.types.is_datetime64_any_dtype(cleaned[col])
    ]
    remaining_identifier_columns = [
        col for col in detect_identifier_columns(cleaned)
        if col in cleaned.columns
    ]

    outlier_report = detect_outliers(cleaned)
    repeated_identifier_rows = 0

    for col in remaining_identifier_columns:
        series = cleaned[col].dropna()
        repeated_identifier_rows += int(series.duplicated().sum())

    final_missing = int(cleaned.isna().sum().sum())
    total_cells = int(cleaned.shape[0] * cleaned.shape[1])

    completeness = (
        round((1 - final_missing / total_cells) * 100, 2)
        if total_cells
        else 100.0
    )

    rows_removed = max(0, original_rows - len(cleaned))
    columns_removed = max(0, original_columns - len(cleaned.columns))

    if not log:
        log.append("No cleaning changes were required.")

    report = {
        "cleaning_log": log,
        "rows_removed": int(rows_removed),
        "columns_removed": int(columns_removed),
        "missing_values_filled": int(missing_values_filled),
        "completeness": float(completeness),
        "duplicate_rows_removed": int(duplicates_before),
        "remaining_missing_values": final_missing,
        "date_columns": [str(col) for col in remaining_date_columns],
        "numeric_columns": [
            str(col)
            for col in cleaned.select_dtypes(include=np.number).columns
        ],
        "categorical_columns": [
            str(col)
            for col in cleaned.select_dtypes(
                include=["object", "string", "category", "bool"]
            ).columns
        ],
        "identifier_columns": [
            str(col) for col in remaining_identifier_columns
        ],
        "constant_columns": [
            str(col)
            for col in constant_columns
            if col in cleaned.columns
        ],
        "outliers_detected": outlier_report,
        "repeated_identifier_values_detected": int(
            repeated_identifier_rows
        ),
    }

    return cleaned, report