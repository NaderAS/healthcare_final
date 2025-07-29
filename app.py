import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor

# -----------------------------
# Load the dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_excel("final_cleaned_latest_dimensions.xlsx")

df = load_data()

# -----------------------------
# Indicator list and adjustments
# -----------------------------
base_indicators = [
    "Overweight prevalence in children aged < 5 years (%)",
    "Underweight prevalence in children aged < 5 years (%)",
    "Stunting prevalence in children aged < 5 years (%)",
    "Wasting prevalence in children aged < 5 years (%)",
    "Severe wasting prevalence in children aged < 5 years (%)"
]

adjustment_factors = {
    "Dietary Iron Deficiency": {"Stunting": 1.3, "Underweight": 1.2, "Overweight": 1.1},
    "Vitamin A Deficiency": {"Stunting": 1.4, "Underweight": 1.3},
    "Iodine Deficiency": {"Stunting": 1.2},
    "Protein Energy Malnutrition": {"Underweight": 1.5, "Stunting": 1.4, "Wasting": 1.4}
}

# -----------------------------
# Title and Instructions
# -----------------------------
st.title("🔮 Malnutrition Risk Predictor")
st.markdown("Use this tool to estimate childhood malnutrition risk based on demographic and health conditions.")

# -----------------------------
# Step 1: Select country outside form
# -----------------------------
setting = st.selectbox("🌍 Country", sorted(df["setting"].unique()), key="country_selector")

# -----------------------------
# Step 2: Main Input Form
# -----------------------------
with st.form("input_form"):
    sex = st.selectbox("Child's Sex", df["sex"].dropna().unique())
    age = st.selectbox("Age Category", df["age_category"].dropna().unique())
    education = st.selectbox("Education Level", df["education"].dropna().unique())
    residence = st.selectbox("Place of Residence", df["residence"].dropna().unique())
    wealth = st.selectbox("Wealth Quintile", df["wealth_quintile"].dropna().unique())

    available_regions = df[df["setting"] == setting]["subnational_region"].dropna().unique()
    region = st.selectbox(
        "Subnational Region",
        sorted(available_regions) if len(available_regions) > 0 else ["Not available"]
    )

    st.markdown("### 🧬 Nutritional Deficiencies")
    iron = st.checkbox("Dietary Iron Deficiency")
    vit_a = st.checkbox("Vitamin A Deficiency")
    iodine = st.checkbox("Iodine Deficiency")
    pem = st.checkbox("Protein Energy Malnutrition")

    submitted = st.form_submit_button("🔍 Predict")

# -----------------------------
# Step 3: Make Predictions
# -----------------------------
if submitted:
    user_input = {
        "setting": setting,
        "sex": sex,
        "age_category": age,
        "education": education,
        "residence": residence,
        "wealth_quintile": wealth,
        "subnational_region": region
    }

    user_deficiencies = {
        "Dietary Iron Deficiency": iron,
        "Vitamin A Deficiency": vit_a,
        "Iodine Deficiency": iodine,
        "Protein Energy Malnutrition": pem
    }

    features = list(user_input.keys())
    input_df = pd.DataFrame([user_input])

    results = {}
    for indicator in base_indicators:
        subset = df[df["indicator_name"] == indicator].dropna(subset=["estimate"])
        if subset.empty:
            results[indicator] = ("No data", "No data")
            continue

        X = subset[features]
        y = subset["estimate"]

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), features)
        ])
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
            ))
        ])

        model.fit(X, y)
        base_pred = model.predict(input_df)[0]
        adjusted = base_pred

        # Apply adjustment factors
        simple = indicator.split()[0]  # e.g., "Stunting"
        for deficiency, present in user_deficiencies.items():
            if present and simple in adjustment_factors.get(deficiency, {}):
                adjusted *= adjustment_factors[deficiency][simple]

        adjusted = min(adjusted, base_pred * 2)  # Cap to avoid extreme values
        results[indicator] = (round(base_pred, 2), round(adjusted, 2))

    # -----------------------------
    # Step 4: Show Results
    # -----------------------------
    st.markdown("## 📊 Prediction Results")
    output_table = pd.DataFrame([
        {
            "Indicator": k.replace(" prevalence in children aged < 5 years (%)", ""),
            "Baseline (%)": v[0],
            "Adjusted for Deficiencies (%)": v[1]
        }
        for k, v in results.items()
    ])

    st.dataframe(output_table, use_container_width=True)
    st.download_button("📥 Download CSV", output_table.to_csv(index=False), file_name="malnutrition_predictions.csv")
