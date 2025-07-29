import pandas as pd
from itertools import product, combinations
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
# Base indicators and features
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

adjustment_factors = {
    "Dietary Iron Deficiency": {"Stunting": 1.3, "Underweight": 1.2, "Overweight": 1.1},
    "Vitamin A Deficiency": {"Stunting": 1.4, "Underweight": 1.3},
    "Iodine Deficiency": {"Stunting": 1.2},
    "Protein Energy Malnutrition": {"Underweight": 1.5, "Stunting": 1.4, "Wasting": 1.4}
}

# -----------------------------
# Generate all combinations
# -----------------------------
value_lists = {col: df[col].dropna().unique().tolist() for col in features}
value_lists = {k: v for k, v in value_lists.items() if len(v) > 0}

required_features = list(value_lists.keys())
base_input_combinations = list(product(*value_lists.values()))
base_df = pd.DataFrame(base_input_combinations, columns=required_features)

# -----------------------------
# Generate all deficiency combinations (16 total)
# -----------------------------
deficiency_names = list(adjustment_factors.keys())
all_def_combos = []
for i in range(0, len(deficiency_names)+1):
    all_def_combos.extend(combinations(deficiency_names, i))

# -----------------------------
# Prediction loop
# -----------------------------
results = []

for indicator in base_indicators:
    subset = df[df["indicator_name"] == indicator]
    if subset.empty or len(subset) < 5:
        print(f"⚠️ Skipping {indicator}")
        continue

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

    model.fit(X_train, y_train)
    preds = model.predict(base_df)

    # -----------------------------
    # Apply all deficiency combinations
    # -----------------------------
    for combo in all_def_combos:
        adjusted = preds.copy()
        for defi in combo:
            effect_dict = adjustment_factors.get(defi, {})
            for i, row in enumerate(base_df.itertuples()):
                simple = indicator.split()[0]  # e.g., "Stunting"
                if simple in effect_dict:
                    adjusted[i] *= effect_dict[simple]

        combo_df = base_df.copy()
        combo_df["indicator_name"] = indicator
        combo_df["deficiencies"] = ", ".join(combo) if combo else "None"
        combo_df["predicted_estimate"] = adjusted.round(2)
        results.append(combo_df)

# -----------------------------
# Final export
# -----------------------------
if results:
    final = pd.concat(results, ignore_index=True)
    final.to_excel("lebanon_combinations_with_deficiencies.xlsx", index=False)
    print("✅ Exported: lebanon_combinations_with_deficiencies.xlsx")
else:
    print("❌ No predictions were made.")
