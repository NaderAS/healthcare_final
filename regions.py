import pandas as pd

# Load dataset
df = pd.read_csv("regions.csv")

# ----------------------------
# PART 1: Mean % Deaths due to Nutritional Deficiencies
# ----------------------------

# Filter for relevant deaths
df_deaths = df[
    (df["metric_name"] == "Percent") &
    (df["measure_name"] == "Deaths") &
    (df["cause_name"] == "Nutritional deficiencies")
]

# Compute mean % deaths per region
mean_deaths_by_region = df_deaths.groupby("location_name")["val"].mean().sort_values(ascending=False)
print("Top regions by average % of deaths due to nutritional deficiencies:")
print(mean_deaths_by_region.head(10))

# ----------------------------
# PART 2: Mean % Prevalence of Sub-Causes of Malnutrition
# ----------------------------

# Filter out the parent cause and keep only prevalence
df_prevalence = df[
    (df["metric_name"] == "Percent") &
    (df["measure_name"] == "Prevalence") &
    (df["cause_name"] != "Nutritional deficiencies")
]

# Compute mean prevalence per region per cause
mean_prevalence_by_region_cause = df_prevalence.groupby(["location_name", "cause_name"])["val"].mean().reset_index()

# Keep only top 10 death regions from Part 1
top_region_names = mean_deaths_by_region.head(10).index.tolist()
filtered_prevalence = mean_prevalence_by_region_cause[
    mean_prevalence_by_region_cause["location_name"].isin(top_region_names)
]

# Sort by region then descending prevalence
ranked_prevalence = filtered_prevalence.sort_values(by=["location_name", "val"], ascending=[True, False])

print("\nTop malnutrition causes by average % prevalence in high-death regions:")
print(ranked_prevalence.head(20))
