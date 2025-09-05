"""
Seattle Housing Affordability & Rent Trends Pipeline
----------------------------------------------------
This script automates the ETL (Extract → Transform → Load) process:
1. Reads raw datasets from `raw_data/`
2. Cleans and merges them into analysis-ready tables
3. Saves outputs to `data_work/` (intermediate) and `data_out/` (final for Tableau/SQL)
4. Exports to SQLite (or Postgres if configured)
5. Runs regression forecasting to project rent trends

Usage:
    python pipeline.py
"""

import os
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Folder setup ---
RAW_DIR = "raw_data"
WORK_DIR = "data_work"
OUT_DIR = "data_out"

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load raw datasets ---
def load_zillow_rent():
    """Load Zillow Observed Rent Index (ZORI) data for Seattle ZIP codes."""
    file_path = os.path.join(RAW_DIR, "zillow_rent.csv")
    df = pd.read_csv(file_path)

    df.rename(columns={
        "RegionName": "ZIP"
    }, inplace=True)
    seattle_df = df[df["Metro"].str.contains("Seattle", case=False, na=False)]
    
    return seattle_df

# df = load_zillow_rent()
# print(df.head())
# print(df["Metro"].unique())

def load_census_income():
    """Load Median household income data."""
    file_path = os.path.join(RAW_DIR, "Income_Breakdown_by_ZIP_Code.csv")
    df = pd.read_csv(file_path)
    df = df[["ZIP","Metro", "Households - Median income (dollars)"]]
    df.rename(columns={
        "Households - Median income (dollars)": "AnnualMedianIncome"
    }, inplace=True)
    df["AnnualMedianIncome"] = pd.to_numeric(df["AnnualMedianIncome"], errors="coerce")
    df["MonthlyMedianIncome"] = df["AnnualMedianIncome"] / 12

    seattle_df = df[df["Metro"].str.contains("Seattle", case=False, na=False)]
    return seattle_df[["ZIP", "Metro", "MonthlyMedianIncome"]]

# df = load_census_income()
# print(df.head())
# print(df["Metro"].unique())

def load_census_rent_burden():
    """Load ACS gross rent as % of household income."""
    file_path = os.path.join(RAW_DIR, "acs_rent_burden.csv")
    df = pd.read_csv(file_path)

    df.rename(columns={
        "NAME": "ZIP"
    }, inplace=True)

    df["ZIP"] = df["ZIP"].str.extract(r"ZCTA5 (\d{5})") 
    df = df.dropna(subset=["ZIP"])
    df["ZIP"] = df["ZIP"].astype(int)

    RENT_BURDEN_RENAME = {
        "B25070_001E": "TotalHouseholds",
        "B25070_002E": "HH_RentLT10Pct",
        "B25070_003E": "HH_Rent10to14Pct",
        "B25070_004E": "HH_Rent15to19Pct",
        "B25070_005E": "HH_Rent20to24Pct",
        "B25070_006E": "HH_Rent25to29Pct",
        "B25070_007E": "HH_Rent30to34Pct",
        "B25070_008E": "HH_Rent35to39Pct",
        "B25070_009E": "HH_Rent40to49Pct",
        "B25070_010E": "HH_Rent50PlusPct",
        "B25070_011E": "HH_RentNotComputed"
    }

    df.rename(columns=RENT_BURDEN_RENAME, inplace=True)
    for col in RENT_BURDEN_RENAME.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.filter(regex="^(ZIP|TotalHouseholds|HH_|Pct_)")
    df = df.drop("HH_RentNotComputed", axis = 1)
    return df

df = load_census_rent_burden()
# print(df.head())

def load_neighborhoods():
    """Load Seattle neighborhood boundaries shapefile or GeoJSON."""
    file_path = os.path.join(RAW_DIR, "Neighborhood_Map_Atlas_Neighborhoods.geojson")
    gdf = gpd.read_file(file_path)
    return gdf

# --- Clean & merge ---
def clean_and_merge(zillow, income, rent_burden):
    """Merge rent + income + burden datasets into a single DataFrame."""
    
    date_columns = zillow.columns[9:]
    zillow["MedianRent"] = zillow[date_columns].median(axis=1)
    zillow = zillow[["ZIP", "MedianRent"]]

    # merge on ZIP code
    df = income.merge(zillow, on="ZIP", how="left")
    df = df.merge(rent_burden, on="ZIP", how="left")

    # SEATTLE_ZIPS = [
    #     98101, 98102, 98103, 98104, 98105, 98106, 98107, 98108, 98109, 98112,
    #     98115, 98116, 98117, 98118, 98119, 98121, 98122, 98125, 98126, 98133,
    #     98134, 98136, 98144, 98146, 98148, 98154, 98155, 98158, 98164, 98166,
    #     98168, 98174, 98177, 98178, 98188, 98195, 98198, 98199
    # ]
    # df = df[df["ZIP"].isin(SEATTLE_ZIPS)]

    # Calculate affordability ratio (rent / income)
    df["rent_to_income"] = df["MedianRent"] / df["MonthlyMedianIncome"]

    return df

# --- Save intermediate + outputs ---
def save_outputs(df):
    """Save merged dataset for analysis + Tableau."""
    work_file = os.path.join(WORK_DIR, "merged_dataset.csv")
    out_file = os.path.join(OUT_DIR, "affordability_dashboard.csv")

    df.to_csv(work_file, index=False)
    df.to_csv(out_file, index=False)

# --- Export to SQL ---
def export_to_sql(df, db_type="sqlite", db_name="housing_affordability.db"):
    if db_type == "sqlite":
        engine = create_engine(f"sqlite:///{db_name}")
    elif db_type == "postgres":
        # Replace with your credentials
        engine = create_engine("postgresql://user:password@localhost:5432/housing")
    else:
        raise ValueError("db_type must be 'sqlite' or 'postgres'")

    df.to_sql("housing_data", engine, if_exists="replace", index=False)
    print(f"✅ Exported to {db_type} database: {db_name}")

# --- Regression Forecasting ---
def prepare_zillow_timeseries(zillow_df, min_months=12):
    """
    Convert wide Zillow rent dataset to long format:
    ZIP | date | rent
    """
    date_cols = [c for c in zillow_df.columns if any(x in c for x in ["/", "-", "20"])]

    long_df = zillow_df.melt(
        id_vars = ["ZIP"], 
        value_vars = date_cols,
        var_name = "date", 
        value_name = "rent"  
    )

    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    long_df = long_df.dropna(subset=["date"]) 
    long_df["year"] = long_df["date"].dt.year
    long_df["month"] = long_df["date"].dt.month

    # Require at least `min_months` valid rent observations
    long_df = long_df.groupby("ZIP").filter(lambda g: g["rent"].notna().sum() >= min_months)

    # Fill gaps
    long_df["rent"] = long_df.groupby("ZIP")["rent"].transform(lambda s: s.ffill().bfill())

    long_df = long_df.sort_values(["ZIP", "date"]).reset_index(drop=True)

    return long_df
def run_forecast(df, years=5):
    """
    Simple linear regression forecast of rents for each ZIP code.
    """
    forecasts = []
    for zip_code, group in df.groupby("ZIP"):
        if group.shape[0] < 24:  # require at least 2 years of data
            continue

        # Use time index (months since start) as predictor
        group["t"] = np.arange(len(group))
        X = group[["t"]]
        y = group["rent"].values

        model = LinearRegression()
        model.fit(X, y)

        # Predict future rents
        future_t = np.arange(len(group), len(group) + years * 12).reshape(-1, 1)
        future_dates = pd.date_range(start=group["date"].max() + pd.offsets.MonthBegin(),
                                     periods=years * 12, freq="MS")
        preds = model.predict(future_t)

        for dt, pred in zip(future_dates, preds):
            forecasts.append({
                "ZIP": zip_code,
                "date": dt,
                "forecast_rent": pred
            })

    forecast_df = pd.DataFrame(forecasts)
    forecast_df.to_csv(os.path.join(OUT_DIR, "rent_forecast.csv"), index=False)
    print("✅ Forecasting complete! Results saved to data_out/rent_forecast.csv")

    return forecast_df

# --- Main pipeline ---
def main():
    print("Loading raw datasets...")
    zillow = load_zillow_rent()
    income = load_census_income()
    rent_burden = load_census_rent_burden()

    print("Cleaning & merging...")
    merged = clean_and_merge(zillow, income, rent_burden)

    print("Saving outputs...")
    save_outputs(merged)

    print("Exporting to SQL...")
    export_to_sql(merged, db_type="sqlite") 

    print("Running regression forecasting...")
    run_forecast(zillow)

    print("🎉 Pipeline complete!")

if __name__ == "__main__":
    main()


