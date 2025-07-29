import pandas as pd
from itertools import product
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor

# -----------------------------
# Load and filter data for Lebanon
# -----------------------------
df = pd.read_excel("final_cleaned_latest_dimensions.xlsx")
df = df[df["setting"] == "Lebanon"]
df = df.dropna(subset=["estimate", "indicator_name"])

# -----------------------------
# Define indicators and input features
# -----------------------------
base_indicators = [
    "Overweight prevalence in children aged < 5 years (%)",
    "Underweight prevalence in children aged < 5 years (%)",
    "Stunting prevalence in children aged < 5 years (%)",
    "Wasting prevalence in children aged < 5 years (%)",
    "Severe wasting prevalence in children aged < 5 years (%)"
]

features = [
    "setting", "sex", "age_category", "education",
    "residence", "wealth_quintile", "subnational_region"
]

# -----------------------------
# Create all possible combinations for Lebanon
# -----------------------------
# Only keep features that have non-empty value lists
value_lists = {col: df[col].dropna().unique().tolist() for col in features}
value_lists = {k: v for k, v in value_lists.items() if len(v) > 0}

# Ensure all required features are present
required_features = list(value_lists.keys())
combinations = list(product(*value_lists.values()))
input_df = pd.DataFrame(combinations, columns=required_features)

# -----------------------------
# Predict for each indicator
# -----------------------------
results = []

for indicator in base_indicators:
    subset = df[df["indicator_name"] == indicator]

    if subset.empty or len(subset) < 5:
        print(f"⚠️ Skipping {indicator} due to insufficient data")
        continue

    # Check all required columns are present
    X_train = subset[required_features]
    y_train = subset["estimate"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), required_features)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
        ))
    ])

    try:
        model.fit(X_train, y_train)
        preds = model.predict(input_df)

        output = input_df.copy()
        output["indicator_name"] = indicator
        output["predicted_estimate"] = preds.round(2)
        results.append(output)
    except Exception as e:
        print(f"❌ Failed for {indicator}: {e}")

# -----------------------------
# Export results
# -----------------------------
if results:
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_excel("lebanon_combinations_predictions.xlsx", index=False)
    print("✅ Exported to lebanon_combinations_predictions.xlsx")
else:
    print("❌ No predictions generated.")
