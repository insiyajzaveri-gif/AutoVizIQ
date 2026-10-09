import pandas as pd
import numpy as np

from autoviziq_engine.data_cleaner import clean_dataframe


data = {
    "Customer_ID": [101, 102, 103, 104, 105, 105],
    "Age": [21, 25, np.nan, 32, 29, 29],
    "Gender": ["Female", "Male", "Female", np.nan, "Female", "Female"],
    "Revenue": [1200, 1500, 1800, np.nan, 2200, 2200],
    "Order_Date": [
        "2026-01-05",
        "2026-01-10",
        "2026-02-15",
        "INVALID DATE",
        "2026-03-01",
        "2026-03-01"
    ],
    "Constant_Column": [
        "Same", "Same", "Same", "Same", "Same", "Same"
    ],
    "Numeric_Text": ["10", "20", "30", "40", "50", "50"],
    "Mostly_Empty": [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        "Only value",
        np.nan
    ]
}


df = pd.DataFrame(data)

# Add completely empty row
df.loc[len(df)] = [np.nan] * 8


print("\n" + "=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print(df)

print("\nOriginal shape:")
print(df.shape)


# Run AutoVizIQ cleaner
cleaned_df, report = clean_dataframe(df)


print("\n" + "=" * 60)
print("CLEANED DATASET")
print("=" * 60)
print(cleaned_df)

print("\nCleaned shape:")
print(cleaned_df.shape)


print("\n" + "=" * 60)
print("AUTOVIZIQ CLEANING REPORT")
print("=" * 60)

print(f"\nOriginal rows: {report['original_rows']}")
print(f"Final rows: {report['final_rows']}")

print(f"\nOriginal columns: {report['original_columns']}")
print(f"Final columns: {report['final_columns']}")

print(
    f"\nDuplicate rows removed: "
    f"{report['duplicate_rows_removed']}"
)

print(
    f"Empty rows removed: "
    f"{report['empty_rows_removed']}"
)

print(
    f"Empty columns removed: "
    f"{report['empty_columns_removed']}"
)

print(
    f"\nFinal missing values: "
    f"{report['final_missing_values']}"
)

print(
    f"Final completeness: "
    f"{report['completeness']}%"
)


print("\n" + "=" * 60)
print("DETECTED DATA FEATURES")
print("=" * 60)

print(
    "\nNumeric-like columns:",
    report["numeric_like_columns"]
)

print(
    "\nDate columns:",
    report["date_columns"]
)

print(
    "\nConstant columns:",
    report["constant_columns"]
)

print(
    "\nIdentifier columns:",
    report["identifier_columns"]
)

print(
    "\nTotal outliers:",
    report["total_outliers"]
)


print("\n" + "=" * 60)
print("WHAT AUTOVIZIQ DID")
print("=" * 60)

for action in report["cleaning_log"]:
    print("✓", action)