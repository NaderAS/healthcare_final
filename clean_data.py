import pandas as pd

# Step 1: Load the Excel file
df = pd.read_excel("data.xlsx")

# Step 2: Keep only the most recent year for each country
latest_dates = df.groupby("setting")["date"].transform("max")
df_latest = df[df["date"] == latest_dates].copy()

# Step 3: Map dimension names to clean column names
dimension_map = {
    "Sex": "sex",
    "Child's age (6 groups) (0-59m)": "age_category",
    "Education (3 groups)": "education",
    "Place of residence": "residence",
    "Economic status (wealth quintile)": "wealth_quintile",
    "Subnational region": "subnational_region"
}

# Step 4: Initialize a new list of rows with dimensions as columns
rows = []

# Step 5: For each row, keep estimate and place the subgroup under the correct dimension column
for _, row in df_latest.iterrows():
    dim_name = dimension_map.get(row["dimension"])
    if dim_name:
        new_row = {
            "setting": row["setting"],
            "indicator_name": row["indicator_name"],
            "wbincome2024": row["wbincome2024"],
            "estimate": row["estimate"],
            dim_name: row["subgroup"]
        }
        rows.append(new_row)

# Step 6: Create DataFrame without grouping or changing estimates
df_final = pd.DataFrame(rows)

# Step 7: Export the result to Excel
df_final.to_excel("final_cleaned_latest_dimensions.xlsx", index=False)

print("✅ File saved as: final_cleaned_latest_dimensions.xlsx")
