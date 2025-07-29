import pandas as pd

# Load the Excel file
df = pd.read_excel("data.xlsx")

# Print column names to verify
print(df.columns)

# Filter rows where the 'dimension' column contains the word "Education"
education_present = df[df['dimension'].str.contains("Education", case=False, na=False)]

# Get unique list of countries (from the 'setting' column)
countries_with_education = education_present['setting'].unique()

# Output the result
print(f"Number of countries with education data: {len(countries_with_education)}")
print("Countries:", countries_with_education)




# Load full dataset
df = pd.read_excel("data.xlsx")

# Create income-to-education estimate mapping
education_mapping = {
    'Low income': 'Primary',
    'Lower-middle income': 'Lower Secondary',
    'Upper-middle income': 'Upper Secondary',
    'High income': 'Tertiary'
}

# Assign estimated education level
df['estimated_education'] = df['wbincome2024'].map(education_mapping)

# Optional: overwrite missing education values only
# Assuming a column 'dimension' where some rows contain 'Education'
education_rows = df['dimension'].str.contains('Education', case=False, na=False)

# Keep the original if it's Education, otherwise assign estimated
df['final_education'] = df.apply(
    lambda row: row['subgroup'] if education_rows.loc[row.name] else row['estimated_education'],
    axis=1
)

# Export to Excel for Power BI
df.to_excel("data_with_estimated_education.xlsx", index=False)
