import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# -----------------------------
# Step 1: Load the cleaned dataset
# -----------------------------
df = pd.read_excel("final_cleaned_latest_dimensions.xlsx")

# -----------------------------
# Step 2: Define the base indicators
# -----------------------------
base_indicators = [
    "Overweight prevalence in children aged < 5 years (%)",
    "Underweight prevalence in children aged < 5 years (%)",
    "Stunting prevalence in children aged < 5 years (%)",
    "Wasting prevalence in children aged < 5 years (%)",
    "Severe wasting prevalence in children aged < 5 years (%)"
]

# -----------------------------
# Step 3: User input
# -----------------------------
user_input = {
    "setting": "Lebanon",
    "sex": "Male",
    "age_category": "24-35 months",
    "education": "Secondary or higher",
    "residence": "Urban",
    "wealth_quintile": "Q3 (middle)",
    "subnational_region": "Mount Lebanon"
}

# -----------------------------
# Step 4: Predict for each indicator
# -----------------------------
features = ["setting", "sex", "age_category", "education", "residence", "wealth_quintile", "subnational_region"]
results = {}

for base in base_indicators:
    # Try sex-specific version first
    sexed_indicator = f"{base} - {user_input['sex']}"
    available_indicators = df["indicator_name"].unique()

    # Use sex-specific if available, otherwise general
    chosen = sexed_indicator if sexed_indicator in available_indicators else base

    # Filter data
    df_subset = df[df["indicator_name"] == chosen].dropna(subset=["estimate"])
    if df_subset.empty:
        results[base] = "No data"
        continue

    X = df_subset[features]
    y = df_subset["estimate"]

    # Preprocess
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), features)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train/test split only if enough rows
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
    else:
        model.fit(X, y)

    # Predict for the user input
    input_df = pd.DataFrame([user_input])
    try:
        prediction = model.predict(input_df)[0]
        results[base] = round(prediction, 2)
    except:
        results[base] = "Prediction error"

# -----------------------------
# Step 5: Output predictions
# -----------------------------
print("\n🔮 Predicted Malnutrition Estimates for Input (Sex-specific when available):\n")
for indicator, pred in results.items():
    name = indicator.replace(" prevalence in children aged < 5 years (%)", "")
    print(f"- {name}: {pred}%")
