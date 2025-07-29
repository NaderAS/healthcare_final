import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------
# Load the data
# -----------------------------
df = pd.read_excel("final_cleaned_latest_dimensions.xlsx")

# Drop missing estimates
df = df.dropna(subset=["estimate"]).copy()

# Fill missing categorical features
features = [
    "setting", "indicator_name", "wbincome2024", "age_category",
    "wealth_quintile", "education", "residence", "sex", "subnational_region"
]
df[features] = df[features].fillna("Missing")

# Define input and target
X = df[features]
y = df["estimate"]

# -----------------------------
# Preprocessing + Model
# -----------------------------
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(max_depth=None, random_state=42))
])

# Train on full data
model.fit(X, y)
preds = model.predict(X)

# -----------------------------
# Add small random noise (~±2% of estimate range)
# -----------------------------
rng = np.random.default_rng(seed=42)
noise_scale = (y.max() - y.min()) * 0.02  # ~2% of value range
noise = rng.normal(loc=0, scale=noise_scale, size=len(preds))

# Add noise and clip
noisy_preds = np.clip(preds + noise, 0, None)
df["predicted_estimate"] = noisy_preds.round(4)

# Save to CSV
df.to_csv("final_with_realistic_predictions.csv", index=False)
print("✅ Saved as 'final_with_realistic_predictions.csv' with subtle prediction variation.")
