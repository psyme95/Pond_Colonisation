# Import libraries
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

# Load and filter data
Pond_Surveys = pd.read_csv('./Data/Pond_Surveys_Clean.csv')
Pond_Surveys = Pond_Surveys[Pond_Surveys['Year'] <= 5]

# Calculate first colonised year for each pond
first_colonised = Pond_Surveys.groupby('Pond_GUID').apply(
    lambda group: group.loc[group['GCN_Status'] == 1, 'Year'].min()
                  if (group['GCN_Status'] == 1).any()
                  else np.nan
)

# Merge back to original dataframe
Pond_Surveys['Year_First_Colonised'] = Pond_Surveys['Pond_GUID'].map(first_colonised)

# Drop rows with NA values for EDP
Pond_Surveys = Pond_Surveys.dropna(subset=["EDP"])

# Add "New_Presence" column to pond surveys
Pond_Surveys["New_Presence"] = (
    Pond_Surveys["Year_First_Colonised"].notna() &
    (Pond_Surveys["Year_First_Colonised"] == Pond_Surveys["Year"])
).astype(int)

# Add previously colonised column
Pond_Surveys["Colonised_Previous"] = (
    Pond_Surveys["Year_First_Colonised"].notna() &
    (Pond_Surveys["Year_First_Colonised"] < Pond_Surveys["Year"])
).astype(int)

# Add never colonised column
Pond_Surveys["Absence"] = (
    Pond_Surveys["Year_First_Colonised"].isna() |
    (Pond_Surveys["Year_First_Colonised"] > Pond_Surveys["Year"])
).astype(int)

# Aggregate and calculate naive occupancy
naive_results = Pond_Surveys.groupby(["EDP", "Year"]).agg({
    'Pond_GUID': 'count',
    'New_Presence': 'sum',
    'Colonised_Previous': 'sum',
    'Absence': 'sum'
}).reset_index()

# Rename the Pond_GUID column to Ponds_Surveyed
naive_results = naive_results.rename(columns={'Pond_GUID': 'Ponds_Surveyed'})

# Calculate Total_Colonised
naive_results['Total_Colonised'] = naive_results['New_Presence'] + naive_results['Colonised_Previous']

# Calculate Yearly_Colonisation and Naive_Occupancy
naive_results['Yearly_Colonisation'] = (
    naive_results['New_Presence'] /
    (naive_results['Ponds_Surveyed'] - naive_results['Colonised_Previous'])
).round(3)

naive_results['Naive_Occupancy'] = (
    naive_results['Total_Colonised'] /
    naive_results['Ponds_Surveyed']
).round(3)

# Calculate confidence intervals for each row
def calculate_ci(row):
    lower, upper = proportion_confint(
        count=row['Total_Colonised'],
        nobs=row['Ponds_Surveyed'],
        method='beta'  # Same as R's binom.test
    )
    return pd.Series({'Lower_CI': round(lower, 3), 'Upper_CI': round(upper, 3)})

naive_results[['Lower_CI', 'Upper_CI']] = naive_results.apply(calculate_ci, axis=1)

# Remove Total_Colonised column and arrange by EDP and Year
naive_results = naive_results.drop("Total_Colonised", axis = 1).sort_values(by = ["EDP", "Year"])

# Export to .csv
naive_results.to_csv('./Output/Naive_Occupancy_Python.csv', index = False)