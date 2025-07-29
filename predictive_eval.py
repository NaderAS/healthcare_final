import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
# Step 3: User input for prediction
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
# Step 4: Initialize storage
# -----------------------------
features = ["setting", "sex", "age_category", "education", "residence", "wealth_quintile", "subnational_region"]
results = {}
metrics_table = []

# -----------------------------
# Step 5: Loop through indicators
# -----------------------------
for base in base_indicators:
    # Try sex-specific version first
    sexed_indicator = f"{base} - {user_input['sex']}"
    available_indicators = df["indicator_name"].unique()

    # Use sex-specific if available, otherwise fallback to general
    chosen = sexed_indicator if sexed_indicator in available_indicators else base

    # Filter data
    df_subset = df[df["indicator_name"] == chosen].dropna(subset=["estimate"])
    if df_subset.empty:
        results[base] = "No data"
        metrics_table.append({
            "Indicator": base,
            "Used": chosen,
            "MSE": "N/A",
            "MAE": "N/A",
            "R²": "N/A"
        })
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

    # Train/test split if enough data
    if len(X) > 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics_table.append({
            "Indicator": base,
            "Used": chosen,
            "MSE": round(mse, 2),
            "MAE": round(mae, 2),
            "R²": round(r2, 2)
        })

    else:
        model.fit(X, y)
        metrics_table.append({
            "Indicator": base,
            "Used": chosen,
            "MSE": "Insufficient data",
            "MAE": "Insufficient data",
            "R²": "Insufficient data"
        })

    # Predict for user
    input_df = pd.DataFrame([user_input])
    try:
        prediction = model.predict(input_df)[0]
        results[base] = round(prediction, 2)
    except:
        results[base] = "Prediction error"

# -----------------------------
# Step 6: Output predictions
# -----------------------------
print("\n🔮 Predicted Malnutrition Estimates for Input:\n")
for indicator, pred in results.items():
    name = indicator.replace(" prevalence in children aged < 5 years (%)", "")
    print(f"- {name}: {pred}%")

# -----------------------------
# Step 7: Output Evaluation Table
# -----------------------------
metrics_df = pd.DataFrame(metrics_table)
print("\n📊 Model Evaluation Summary:\n")
print(metrics_df)

# Optional: Export metrics to Excel
metrics_df.to_excel("model_evaluation_summary.xlsx", index=False)
