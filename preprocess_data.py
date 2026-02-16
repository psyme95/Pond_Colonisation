# Import libraries
import pandas as pd
import geopandas as gpd
import numpy as np
from OSGridConverter import grid2latlong

# Load the data
Pond_Surveys = pd.read_excel('./Data/Pond_Surveys.xls')
Pond_Agreements = pd.read_excel('./Data/Pond_Agreements.xls')
EDP_Boundaries = gpd.read_file('./Data/EDP_Boundaries_111225.gdb', layer='EDP_Boundaries_111225')

# Process & Clean Pond Agreements ----
Pond_Ag = Pond_Agreements[["GlobalID", "Site Grid Reference", "Pond Status",
                           "Creation or Restoration?", "Within Core/Fringe Area?"]].copy()

Pond_Ag = Pond_Ag.rename(columns={
    "Site Grid Reference": "Grid_Ref",
    "Pond Status": "Pond_Status",
    "Creation or Restoration?": "Type",
    "Within Core/Fringe Area?": "Area"
})

# Clean the Pond Status column
Pond_Ag["Pond_Status"] = Pond_Ag["Pond_Status"].replace({
    "Pond Complete": "Complete",
    "Pond Complete/Under Review": "Complete",
    "Pond Failed": "Failed"
})

# Collapse restoration types into a single category
Pond_Ag["Type"] = Pond_Ag["Type"].map({
    "Creation": "Creation",
    "Restoration (existing pond)": "Restoration",
    "Restoration (ghost pond)": "Restoration"
})

# Keep only core/fringe
Pond_Ag["Area"] = Pond_Ag["Area"].map({
    "Core": "Core",
    "Fringe": "Fringe"
})

# Remove spaces from grid references
Pond_Ag["Grid_Ref"] = Pond_Ag["Grid_Ref"].fillna("").str.replace(r"[^A-Za-z0-9]", "", regex=True)

# Convert grid references to lat/long
def safe_grid2latlong(grid_ref):
    if pd.isna(grid_ref) or grid_ref == "" or len(grid_ref) < 4:
        return pd.Series([np.nan, np.nan])
    try:
        coords = grid2latlong(grid_ref)
        return pd.Series([coords.latitude, coords.longitude])
    except:
        return pd.Series([np.nan, np.nan])

Pond_Ag[["Latitude", "Longitude"]] = Pond_Ag["Grid_Ref"].apply(safe_grid2latlong)

# Filter to only include complete or failed ponds
Pond_Ag = Pond_Ag[Pond_Ag["Pond_Status"].isin(["Complete", "Failed"])]

# Process & Clean Pond Surveys ----
Pond_Surv = Pond_Surveys[["Pond_GUID", "Monitoring Year", "eDNA Score", "GCN Status"]].copy()

Pond_Surv = Pond_Surv.rename(columns={
    "Pond_GUID": "Pond_GUID",
    "Monitoring Year": "Year",
    "eDNA Score": "eDNA_Score",
    "GCN Status": "GCN_Status"
})

# Convert Year to numeric
Pond_Surv["Year"] = Pond_Surv["Year"].replace({
    "Year 1": 1,
    "Year 2": 2,
    "Year 3": 3,
    "Year 4": 4,
    "Year 5": 5,
    "Contingency Survey": 6
})

# Extract eDNA Score as numeric
Pond_Surv["eDNA_Score"] = pd.to_numeric(
    Pond_Surv["eDNA_Score"].astype(str).str.extract(r'(\d+)')[0],
    errors='coerce'
)

# Convert GCN Status to binary
Pond_Surv["GCN_Status"] = Pond_Surv["GCN_Status"].map({
    "Present": 1,
    "Absent": 0
})

# Join surveys to agreements
Pond_Data = pd.merge(
    Pond_Surv,
    Pond_Ag,
    left_on="Pond_GUID",
    right_on="GlobalID",
    how="left"
)

# Remove surveys without valid GCN status, Year, or Pond Status
Pond_Data = Pond_Data.dropna(subset=["GCN_Status", "Year", "Pond_Status"])

# Collapse duplicate surveys within the same pond-year ----
# Take maximum GCN status and eDNA score
Pond_Data = Pond_Data.groupby(["Pond_GUID", "Year"], as_index=False).agg({
    "GCN_Status": "max",
    "eDNA_Score": lambda x: np.nan if x.isna().all() else x.max(),
    "Type": "first",
    "Area": "first",
    "Pond_Status": "first",
    "Latitude": "first",
    "Longitude": "first"
})

# Add pond colonisation column ----
# GCN_Colonised = 1 if GCN detected in this year or any previous year
Pond_Data = Pond_Data.sort_values(["Pond_GUID", "Year"])
Pond_Data["GCN_Colonised"] = Pond_Data.groupby("Pond_GUID")["GCN_Status"].cummax()

# Create spatial points and join to EDP boundaries ----
# Filter to only ponds with valid coordinates
Pond_Data_With_Coords = Pond_Data.dropna(subset=["Latitude", "Longitude"])

# Create GeoDataFrame
Pond_Geo = gpd.GeoDataFrame(
    Pond_Data_With_Coords,
    geometry=gpd.points_from_xy(
        Pond_Data_With_Coords["Longitude"],
        Pond_Data_With_Coords["Latitude"]
    ),
    crs="EPSG:4326"
)

# Transform to match EDP_Boundaries CRS
Pond_Geo = Pond_Geo.to_crs(EDP_Boundaries.crs)

# Spatial join
Pond_Final = gpd.sjoin(Pond_Geo, EDP_Boundaries, how="left", predicate="within")

# Drop geometry and spatial columns, keep only EDP
Pond_Final = Pond_Final.drop(columns=["geometry", "index_right"])

# Select relevant columns (matching R output structure)
Pond_Final = Pond_Final[["Pond_GUID", "Year", "GCN_Status", "GCN_Colonised",
                         "eDNA_Score", "Type", "Area", "Pond_Status",
                         "Latitude", "Longitude", "EDP"]]

# Save cleaned datasets ----
Pond_Final.to_csv('./Data/Pond_Surveys_Clean.csv', index=False)
Pond_Ag.to_csv('./Data/Pond_Agreements_Clean.csv', index=False)

print("Data preprocessing complete!")
print(f"Pond surveys: {len(Pond_Final)} rows")
print(Pond_Final.head())