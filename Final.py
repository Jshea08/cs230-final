"""
CS-230 Final Project
Cardiovascular Disease Prediction
Author: Jared Shea

This script:
- Loads the cardiovascular dataset
- Cleans & engineers features (age in years, realistic blood pressure)
- Fits a generalized linear model (logistic regression)
- Evaluates performance (accuracy, precision, recall, confusion matrix)
- Saves the trained model & feature list for use in a Streamlit app
- Builds a Streamlit app with:
    * Home page (project overview + about me + model explanation)
    * Exploratory Data Analysis page with dropdown for 13+ variables
    * Risk Estimator page using the trained model
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    confusion_matrix,)
import joblib

import streamlit as st
import altair as alt

# ============================================================
# HELPER FUNCTIONS ADDED
# ============================================================

def compute_bmi(weight_kg, height_cm, round_to=1):                                                                      # A function with two or more parameters
    """Compute BMI from weight (kg) and height (cm)."""
    height_m = height_cm / 100.0
    if height_m <= 0:
        return None
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, round_to)


def basic_numeric_summary(series):                                                                                      # A function that returns two or more values
    """Return the mean and standard deviation for a numeric Series."""
    return series.mean(), series.std()


def add_age_years_column(df):
    """Add age_years column (age in days → years as int) and print summary."""
    df["age_years"] = (df["age"] / 365.25).astype(int)                                                                  # Add a new column to the DataFrame
    print(df["age_years"].describe())
    return df


def create_bp_mask(df):
    """Return boolean mask for rows with realistic blood pressure and print counts."""
    bp_mask = (                                                                                                         # Filter data by multiple conditions with AND
        (df["ap_hi"] >= 80)
        & (df["ap_hi"] <= 240)
        & (df["ap_lo"] >= 40)
        & (df["ap_lo"] <= 150)
        & (df["ap_hi"] > df["ap_lo"]))

    print("Original number of rows:", len(df))
    print("Rows with realistic BP:", bp_mask.sum())
    return bp_mask


def split_features_target(df, feature_cols, target_col="cardio"):
    """Split a DataFrame into X (features) and y (target)."""
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def evaluate_model(model, X_test, y_test):
    """Print and return common evaluation metrics for a classifier."""
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print("Accuracy:", acc)
    print("\nClassification report:")
    print(classification_report(y_test, preds))
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(cm)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,
    }


def compute_risk_bucket(prob):
    """Turn a predicted probability into a simple risk label."""
    if prob < 0.20:
        return "Low estimated risk"
    elif prob < 0.50:
        return "Moderate estimated risk"
    else:
        return "Higher estimated risk"


# ============================================================
# 1–9. DATA LOADING, CLEANING, MODEL FITTING, SAVING                                                                    #Thank professor Schirmacher for this part
# ============================================================

df = pd.read_csv("cardio.csv", sep=";")                                                                   # 1. Load data
print(df.head())
print(df.info())

df = add_age_years_column(df)                                                                                            # 2. Feature engineering: age in years  (call 1)

bp_mask = create_bp_mask(df)                                                                                             # 3. Clean unrealistic blood pressure values
df_clean = df[bp_mask].copy()

print(df_clean.describe())
print("Cardio distribution (proportion):")
print(df_clean["cardio"].value_counts(normalize=True))

features = [                                                                                                            # 4. Select features and target
    "age_years",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "weight",]

feature_labels = [f"Feature: {col}" for col in features]                                                                # List comprehension example

X, y = split_features_target(df_clean, features, target_col="cardio")

                                                                                                                        # 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,)

model = LogisticRegression(max_iter=500)                                                                                # 6. Fit logistic regression GLM
model.fit(X_train, y_train)

metrics = evaluate_model(model, X_test, y_test)                                                                         # 7. Predictions & evaluation

coef_df = pd.DataFrame(                                                                                                 # 8. Coefficients
    {
        "feature": features,
        "coefficient": model.coef_[0],}
)

coef_df = coef_df.sort_values(by="coefficient", key=np.abs, ascending=False)                                            #Sort by absolute value of coefficient

print("\nLogistic Regression Coefficients (sorted by |coefficient|):")
print(coef_df)

for fname in features:                                                                                                  # Loop through items in a list
    print("Trained feature:", fname)

joblib.dump(model, "cardio_log_reg_model.joblib")                                                               # 9. Save model & feature list for Streamlit app
joblib.dump(features, "cardio_features.joblib")

print(
    "\nSaved model to 'cardio_log_reg_model.joblib' and feature list to 'cardio_features.joblib'.")

# ============================================================
# STREAMLIT APP
# ============================================================

model = joblib.load("cardio_log_reg_model.joblib")                                                                      # Thank you professor Chervany for this section
features = joblib.load("cardio_features.joblib")

df_app = pd.read_csv("cardio.csv", sep=";")
df_app = add_age_years_column(df_app)

df_app["bmi"] = df_app.apply(                                                                                           # Add BMI column using compute_bmi
    lambda row: compute_bmi(row["weight"], row["height"]),
    axis=1,)

# ---------- Streamlit page config ----------
st.set_page_config(
    layout="wide",
    page_title="Cardiovascular Disease Analysis",
    page_icon="❤️",)

# ---------- Custom CSS for dark red header & cleaner layout ----------
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #4a0000, #8b0000);
        color: white;
        padding: 1.6rem 2.2rem;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        letter-spacing: 0.03em;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.9;
    }
    /* Tighten default top whitespace under the header */
    .block-container {
        padding-top: 1.2rem;
    }
    .prof-list {
        font-size: 0.85rem;
        line-height: 1.2rem;
        margin-left: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,)

# ---------- Custom header banner ----------
st.markdown(
    """
    <div class="main-header">
        <h1>❤️ Cardiovascular Disease Analysis & Risk Estimator</h1>
        <p>CS-230 Final Project • Bentley University • Jared Shea</p>
    </div>
    """,
    unsafe_allow_html=True,)

st.write(
    "Use the **sidebar** to switch between pages:\n\n"
    "- **Home**: Overview of the project, why it matters to me, and how the model works.\n"
    "- **Exploratory Analysis**: Explore the variables with interactive charts and key takeaways.\n"
    "- **Risk Estimator**: Enter information and get an estimated probability.\n\n")

st.sidebar.title("Navigation")                                                                                          # Sidebar navigation
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Exploratory Analysis", "Risk Estimator"],)

pretty_names = {                                                                                                        # Helper: nicer names for some variables in text
    "age_years": "Age (in years)",
    "age": "Age (in days)",
    "ap_hi": "Systolic blood pressure",
    "ap_lo": "Diastolic blood pressure",
    "cholesterol": "Cholesterol category",
    "gluc": "Glucose category",
    "smoke": "Smoking status",
    "alco": "Alcohol use",
    "active": "Physical activity",
    "weight": "Body weight",
    "height": "Height",
    "gender": "Gender",
    "cardio": "Cardiovascular disease indicator",
    "bmi": "Body Mass Index",}

_ = list(pretty_names.keys())
_ = pretty_names.get("age_years", "Age (in years)")

# ============================================================
# PAGE 0 — HOME  (unchanged content)
# ============================================================

if page == "Home":
    col_left, col_right = st.columns([1, 2])

    # ----- LEFT COLUMN: images -----
    with col_left:
        st.subheader("Jared Shea - Author")
        if os.path.exists("jared_photo.jpg"):
            st.image(
                "jared_photo.jpg",
                caption="Jared Shea",
                use_container_width=True,)
        else:
            st.info(
                "Add a file named **`jared_photo.jpg`** to this folder to show your picture here.")

        if os.path.exists("heart.png"):
            st.image(
                "heart.png",
                caption="Cardiovascular health",
                use_container_width=True,)
        elif os.path.exists("heart.jpg"):
            st.image(
                "heart.jpg",
                caption="Cardiovascular health",
                use_container_width=True,)
        else:
            st.info(
                "Add a file named **`heart.png`** or **`heart.jpg`** (for example, a heart or ECG image) to show it here.")

        # -------- PROFESSOR ACKNOWLEDGMENT SECTION --------
        st.subheader("Dedication")

        st.markdown("""
            <style>
                .prof-list {
                    font-size: 0.85rem;      /* slightly smaller so lines don't wrap */
                    line-height: 1.2rem;     /* tighter spacing */
                    margin-left: 0.2rem;
                }
            </style>
            """,
                    unsafe_allow_html=True)

        st.markdown("""
            <div class="prof-list">
            <strong>This project is dedicated to these professors who provided me with the foundational knowledge to carry out the inner workings of this project:</strong><br><br>
            • Professor Vaughan — MA 214 Applied Statistics<br>
            • Professor Schirmacher — MA 380 Generalized Linear Models<br>
            • Professor Carter — MA 705 Statistical Modeling<br>
            • Professor Cherveny — ST 625 Quantitative Analysis<br>
            • Professor Masloff — CS 230 Programming Essentials
            </div>
            """,
                    unsafe_allow_html=True)

    # ----- RIGHT COLUMN: text sections -----
    with col_right:
        st.subheader("About This Project")
        st.write(
            "This final project for CS-230 intro to Python uses a real cardiovascular health "
            "dataset with **70,000 patient records**. Each row contains medical measurements like age, "
            "blood pressure, cholesterol, glucose, height and weight, plus lifestyle factors such as "
            "smoking, alcohol use, and physical activity.\n\n"
            "My goals are to:\n"
            "1. **Understand the data** – Which health and lifestyle factors are most associated with cardiovascular disease?\n"
            "2. **Build a predictive model** – Use a generalized linear model (logistic regression) to estimate the probability "
            "that an individual has cardiovascular disease based on their characteristics.\n")

        st.subheader("Why This Matters to Me")
        st.write(
            "This project is personal. Recently, I’ve been seeing cardiologists for several issues related to "
            "my own heart. That experience has motivated me to learn as much as I can about cardiovascular disease — "
            "not just from a medical perspective, but also from a data-driven one.\n\n"
            "By working with this dataset, I’m trying to better understand risk factors that might affect me and other "
            "people going through similar things. As a senior triple major in **Business Economics, Professional Sales, "
            "and Data Analytics**, this project combines strengths of all three: real-world data, quantitative modeling, "
            "and communication. It’s meaningful to me that I can bring analytics skills into a space that directly "
            "impacts health and quality of life.\n")

        st.subheader("How This Model Works")
        st.write(
            "This project uses a generalized linear model (GLM), specifically a logistic regression model, to estimate "
            "the probability that an individual has cardiovascular disease. The model evaluates how predictors such as age, "
            "blood pressure, cholesterol, glucose, lifestyle habits, and overall physical health contribute to changes in "
            "disease risk. Logistic regression models the log-odds of cardiovascular disease as a linear combination of these "
            "inputs, allowing us to quantify how each factor influences the likelihood of disease. In addition, this approach "
            "provides interpretable coefficients that help clarify which variables play the most meaningful role in shaping "
            "someone’s predicted outcome.\n\n"
            "Once trained on the cleaned dataset, the model can take new inputs and convert them into a predicted probability "
            "between 0 and 1. This probability reflects the model’s assessment of how likely it is that a person with a "
            "particular profile has cardiovascular disease. While simplified, this GLM approach is widely used in medicine "
            "and epidemiology because it balances interpretability with predictive power, making it a valuable tool for "
            "understanding real-world health risk patterns. By structuring the model this way, we can not only produce accurate "
            "predictions but also better understand the underlying relationships that drive cardiovascular risk across "
            "different groups of individuals.")

# ============================================================
# PAGE 1 — EXPLORATORY ANALYSIS                                                                                         #Thank you professor Carter for this part
# ============================================================

elif page == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")

    variable = st.selectbox(                                                                                            # Streamlit dropdown list
        "Choose a variable to explore:",
        [
            "age_years",
            "age",
            "gender",
            "height",
            "weight",
            "bmi",
            "ap_hi",
            "ap_lo",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "cardio",
        ],)

    nice_var = pretty_names.get(variable, variable)

    st.subheader(f"Summary statistics for `{variable}`")
    st.write(df_app[variable].describe())

    if pd.api.types.is_numeric_dtype(df_app[variable]):                                                                 # Find min/max/mean/std for a numeric column
        min_val = df_app[variable].min()
        max_val = df_app[variable].max()
        mean_val, std_val = basic_numeric_summary(df_app[variable])
        st.caption(
            f"Min: {min_val:.2f} | Max: {max_val:.2f} | Mean: {mean_val:.2f} | Std: {std_val:.2f}"
        )

    continuous_vars = ["age_years", "age", "height", "weight", "bmi", "ap_hi", "ap_lo"]                                 # Decide continuous vs categorical
    is_continuous = variable in continuous_vars

    st.subheader("Distribution")

    if is_continuous:
        hist = (                                                                                                         # Histogram  #CHART1
            alt.Chart(df_app)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X(variable, bin=alt.Bin(maxbins=30), title=nice_var),
                y=alt.Y("count()", title="Count"),)
            .properties(height=250, title=f"Distribution of {nice_var}")
        )
        st.altair_chart(hist, use_container_width=True)

        hist_by_cardio = (                                                                                              # Histogram by cardio
            alt.Chart(df_app)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X(variable, bin=alt.Bin(maxbins=30), title=nice_var),
                y=alt.Y("count()", title="Count"),
                color=alt.Color(
                    "cardio:N",
                    title="Cardio (0 = no disease, 1 = disease)",),)
            .properties(
                height=250,
                title=f"Distribution of {nice_var} by cardio status",)
        )
        st.altair_chart(hist_by_cardio, use_container_width=True)

        st.subheader(f"{nice_var} by cardio status (boxplot)")                                            #              Boxplot by cardio  #CHART2
        box = (
            alt.Chart(df_app)
            .mark_boxplot()
            .encode(
                x=alt.X("cardio:N", title="Cardio (0 = no disease, 1 = disease)"),
                y=alt.Y(variable, title=nice_var),)
            .properties(height=250)
        )
        st.altair_chart(box, use_container_width=True)

    else:
        st.caption("Showing proportion of each category by cardio status.")                                             # Categorical: stacked proportions by cardio
        cat_df = (
            df_app.groupby([variable, "cardio"])
            .size()
            .rename("count")
            .reset_index())

        total_per_level = cat_df.groupby(variable)["count"].transform("sum")
        cat_df["proportion"] = cat_df["count"] / total_per_level

        cat_chart = (
            alt.Chart(cat_df)
            .mark_bar()
            .encode(
                x=alt.X(f"{variable}:N", title=nice_var),
                y=alt.Y("proportion:Q", title="Proportion"),
                color=alt.Color("cardio:N", title="Cardio"),
                tooltip=[variable, "cardio", "proportion"],)
            .properties(
                height=300,
                title=f"{nice_var} distribution by cardio status (proportions)",)
        )
        st.altair_chart(cat_chart, use_container_width=True)

    means_by_cardio = None                                                                                              # Relationship with cardio
    corr_value = None

    if variable != "cardio":
        st.subheader(f"Mean `{variable}` by cardio")
        means_by_cardio = df_app.groupby("cardio")[variable].mean().rename(
            "mean_" + variable)
        st.write(means_by_cardio)

        if is_continuous or df_app[variable].dtype != "object":                                                         # correlation for numeric vars
            st.subheader(f"Correlation between `{variable}` and `cardio`")
            corr_df = df_app[["cardio", variable]].corr()
            st.write(corr_df)
            corr_value = corr_df.loc[variable, "cardio"]
    else:
        st.subheader("Cardio distribution (0 = no disease, 1 = disease)")
        st.write(
            df_app["cardio"]
            .value_counts(normalize=True)
            .rename("proportion"))

    st.subheader("Example pivot table: mean cardio by age_years and cholesterol")                                        # Pivot table example
    pivot_table = df_app.pivot_table(
        index="age_years",
        columns="cholesterol",
        values="cardio",
        aggfunc="mean",)
    st.write(pivot_table.head())

    # ========================================================
    # KEY TAKEAWAYS FOR THIS VARIABLE
    # ========================================================
    st.subheader(f"Key Takeaways for {nice_var}")

    overall_rate = df_app["cardio"].mean()
    overall_pct = round(overall_rate * 100, 1)

    if variable == "cardio":
        st.write(
            f"- Overall, about **{overall_pct}%** of patients in this dataset are labeled as having cardiovascular disease (`cardio = 1`).\n"
            "- The split between 0 and 1 is fairly balanced, which is helpful for training a predictive model.\n"
            "- This baseline rate is useful context when you compare disease rates across different subgroups."
        )

    elif is_continuous and means_by_cardio is not None:
        mean0 = means_by_cardio.loc[0]
        mean1 = means_by_cardio.loc[1]
        diff = mean1 - mean0

        mean0_r = round(mean0, 2)
        mean1_r = round(mean1, 2)
        diff_r = round(diff, 2)

        if diff > 0:
            direction_phrase = (
                "Patients **with** cardiovascular disease tend to have **higher** values "
                f"of {nice_var} compared to those without disease."
            )
        else:
            direction_phrase = (
                "Patients **without** cardiovascular disease tend to have **higher** values "
                f"of {nice_var} compared to those with disease."
            )

        st.write(
            f"- Among patients **without** cardio (0), average `{variable}` ≈ **{mean0_r}**.\n"
            f"- Among patients **with** cardio (1), average `{variable}` ≈ **{mean1_r}**.\n"
            f"- The difference between the two groups is about **{abs(diff_r)} units**.\n\n"
            f"{direction_phrase}"
        )

        if corr_value is not None:
            corr_r = round(corr_value, 3)
            if abs(corr_value) < 0.05:
                corr_text = (
                    f"The correlation between `{variable}` and `cardio` is about **{corr_r}**, "
                    "which suggests only a very weak linear relationship in this dataset."
                )
            elif corr_value > 0:
                corr_text = (
                    f"The correlation between `{variable}` and `cardio` is about **{corr_r}**, "
                    "indicating that higher values of this variable are associated with a "
                    "**higher likelihood** of cardiovascular disease."
                )
            else:
                corr_text = (
                    f"The correlation between `{variable}` and `cardio` is about **{corr_r}**, "
                    "indicating that higher values of this variable are associated with a "
                    "**lower likelihood** of cardiovascular disease."
                )
            st.write(corr_text)

        st.write(
            "Overall, this variable helps highlight how typical levels differ between the cardio and "
            "non-cardio groups, which supports its role as a predictor in the logistic regression model."
        )

    else:
        rates_by_level = df_app.groupby(variable)["cardio"].mean()                                                      # Categorical: disease rate by category
        level_high = rates_by_level.idxmax()
        level_low = rates_by_level.idxmin()
        high_rate = round(rates_by_level.max() * 100, 1)
        low_rate = round(rates_by_level.min() * 100, 1)

        st.write(
            f"- Across all patients, about **{overall_pct}%** have cardiovascular disease.\n"
            f"- For `{variable}`, the lowest disease rate is in category **`{level_low}`**, "
            f"with about **{low_rate}%** of patients having cardio.\n"
            f"- The highest disease rate is in category **`{level_high}`**, "
            f"with about **{high_rate}%** of patients having cardio.\n\n"
            "This means that different levels of this variable line up with noticeably different risk levels. "
            "Categories that sit well above the overall disease rate look like higher-risk groups, while "
            "those well below it look more protected in this dataset."
        )

# ============================================================
# PAGE 2 — RISK ESTIMATOR                                                                                               #Shout out professor Vaughan
# ============================================================

elif page == "Risk Estimator":
    st.header("Cardiovascular Disease Risk Estimator")

    col1, col2 = st.columns(2)

    with col1:
        age_years_input = st.slider("Age (years)", 29, 64, 50)                                                          # Slider
        ap_hi_input = st.slider("Systolic BP (ap_hi)", 80, 240, 130)
        ap_lo_input = st.slider("Diastolic BP (ap_lo)", 40, 150, 80)
        weight_input = st.slider("Weight (kg)", 40, 150, 75)
        height_input = st.slider("Height (cm)", 140, 210, 170)

    with col2:
        cholesterol_input = st.selectbox("Cholesterol", [1, 2, 3])
        gluc_input = st.selectbox("Glucose", [1, 2, 3])
        smoke_input = st.selectbox("Do you smoke?", [0, 1])
        alco_input = st.selectbox("Drink alcohol?", [0, 1])
        active_input = st.selectbox("Physically active?", [0, 1])

    if st.button("Estimate Risk"):
        input_dict = {
            "age_years": age_years_input,
            "ap_hi": ap_hi_input,
            "ap_lo": ap_lo_input,
            "cholesterol": cholesterol_input,
            "gluc": gluc_input,
            "smoke": smoke_input,
            "alco": alco_input,
            "active": active_input,
            "weight": weight_input,
        }

        input_df = pd.DataFrame(
            [[input_dict[col] for col in features]],
            columns=features,)

        prob = model.predict_proba(input_df)[0, 1]
        pred = model.predict(input_df)[0]

        st.subheader("Estimated Probability of Cardiovascular Disease")
        st.write(f"### **{prob:.2%}**")

        risk_bucket = compute_risk_bucket(prob)
        st.write(f"**Risk Category:** {risk_bucket}")

        user_bmi = compute_bmi(weight_input, height_input, round_to=1)
        st.write(f"Estimated BMI based on your inputs: **{user_bmi}**")

        if pred == 1:
            st.error("Model Prediction: **Higher Risk (cardio = 1)**")
        else:
            st.success("Model Prediction: **Lower Risk (cardio = 0)**")


