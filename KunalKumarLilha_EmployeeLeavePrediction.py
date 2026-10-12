# Employee Leave (Attrition) Prediction & HR Decision Dashboard
# Single-file submission: frontend + data pipeline + ML model + analytics

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Employee Leave Risk Analytics", page_icon="👥", layout="wide")
DATASET_NAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset"
KAGGLE_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/pavansubhasht/ibm-hr-analytics-attrition-dataset"

def find_local_dataset():
    candidates = [
        Path(DATASET_NAME),
        Path("data") / DATASET_NAME,
        Path("/mnt/data") / DATASET_NAME,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

@st.cache_data(show_spinner=False)
def download_kaggle_dataset():
    import urllib.request
    with urllib.request.urlopen(KAGGLE_DOWNLOAD_URL, timeout=30) as response:
        content = response.read()
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        preferred = [n for n in csv_names if DATASET_NAME.lower() in n.lower()]
        selected = preferred[0] if preferred else csv_names[0]
        return pd.read_csv(zf.open(selected))

def load_dataset(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    local = find_local_dataset()
    if local:
        return pd.read_csv(local)
    try:
        return download_kaggle_dataset()
    except Exception:
        return None

def prepare_data(df):
    required = {"Attrition"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "The uploaded file is missing the required column(s): "
            + ", ".join(sorted(missing))
        )

    data = df.copy()
    data["Attrition"] = data["Attrition"].astype(str).str.strip().str.title()
    data = data.drop_duplicates().reset_index(drop=True)

    drop_cols = [c for c in ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"] if c in data.columns]
    data = data.drop(columns=drop_cols, errors="ignore")

    if "Age" in data.columns:
        data["AgeBand"] = pd.cut(
            data["Age"], bins=[17, 25, 35, 45, 60, np.inf],
            labels=["18-25", "26-35", "36-45", "46-60", "60+"],
            include_lowest=True,
        )

    if "YearsAtCompany" in data.columns:
        data["TenureBand"] = pd.cut(
            data["YearsAtCompany"], bins=[-1, 1, 3, 5, 10, np.inf],
            labels=["0-1", "2-3", "4-5", "6-10", "10+"],
            include_lowest=True,
        )

    if "MonthlyIncome" in data.columns:
        data["IncomeBand"] = pd.qcut(
            data["MonthlyIncome"], q=4,
            labels=["Low", "Lower-Mid", "Upper-Mid", "High"],
            duplicates="drop",
        )

    satisfaction_cols = [
        c for c in [
            "EnvironmentSatisfaction", "JobSatisfaction",
            "RelationshipSatisfaction", "WorkLifeBalance"
        ] if c in data.columns
    ]
    if satisfaction_cols:
        data["SatisfactionScore"] = data[satisfaction_cols].mean(axis=1).round(2)

    data["AttritionFlag"] = (data["Attrition"] == "Yes").astype(int)
    if data["AttritionFlag"].nunique() < 2:
        raise ValueError("The Attrition column must contain both Yes and No values.")
    return data

def build_model(data):
    X = data.drop(columns=["AttritionFlag", "Attrition"], errors="ignore")
    y = data["AttritionFlag"]
    X = X.drop(columns=["AgeBand", "TenureBand", "IncomeBand"], errors="ignore")

    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = X.select_dtypes(include=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }
    cm = confusion_matrix(y_test, predictions)
    return pipeline, metrics, cm

def rate_by(data, column):
    if column not in data.columns:
        return pd.DataFrame()
    out = (
        data.groupby(column, observed=True)["AttritionFlag"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "Employees", "sum": "Leavers", "mean": "AttritionRate"})
    )
    out["AttritionRate"] *= 100
    return out.sort_values("AttritionRate", ascending=False)

def score_all(model, data):
    X = data.drop(columns=["AttritionFlag", "Attrition"], errors="ignore")
    X = X.drop(columns=["AgeBand", "TenureBand", "IncomeBand"], errors="ignore")
    result = data.copy()
    result["RiskProbability"] = model.predict_proba(X)[:, 1]
    result["RiskBand"] = pd.cut(
        result["RiskProbability"],
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low", "Watch", "High"]
    )
    return result

def report_markdown(data, metrics, total, leavers, attrition_rate, high_risk):
    dept = rate_by(data, "Department")
    top_department = (
        f"{dept.iloc[0]['Department']} ({dept.iloc[0]['AttritionRate']:.1f}%)"
        if not dept.empty else "Not available"
    )
    return f"""# Employee Leave (Attrition) Risk Analytics

## Executive summary

This dashboard analyses the IBM HR Analytics Employee Attrition & Performance
practice dataset and estimates employee attrition risk for decision support.
The dataset contains **{total:,} records**, including **{leavers:,} labelled
leavers**, for an observed attrition rate of **{attrition_rate:.1f}%**.

The highest observed department attrition rate is **{top_department}**.
The model identifies **{high_risk:,} active employee profiles** in the high-risk
band (probability at least 60%). These profiles should receive human review,
not automatic action.

## Dashboard design

1. **Executive Overview** — KPI cards, attrition mix, department comparison,
   and the Fact → Insight → Opportunity → Action decision narrative.
2. **Drivers & Trends** — overtime, department, age-band, and tenure-band
   comparisons with targeted retention recommendations.
3. **Risk & Prediction** — active-employee risk distribution, highest-risk
   profiles, and a single-employee what-if prediction form.
4. **Model Evaluation** — Accuracy, Precision, Recall, F1 Score, ROC-AUC,
   and the confusion matrix.

## Model results

| Metric | Value |
|---|---:|
| Accuracy | {metrics["Accuracy"]:.1%} |
| Precision | {metrics["Precision"]:.1%} |
| Recall | {metrics["Recall"]:.1%} |
| F1 Score | {metrics["F1 Score"]:.1%} |
| ROC-AUC | {metrics["ROC-AUC"]:.1%} |

## Methodology

The workflow removes duplicate rows and source identifier/constant columns,
creates age, tenure, income, and satisfaction features, and uses a stratified
80/20 train-test split. Numeric values are median-imputed and standardised.
Categorical values are imputed and one-hot encoded. A class-balanced Logistic
Regression model produces the risk probability.

## Recommended actions

- Review excessive overtime in high-risk departments and roles.
- Add early-tenure check-ins during the first few years of employment.
- Strengthen engagement and work-life-balance interventions.
- Review compensation and career progression in high-attrition roles.
- Use model output as a review signal, never as the sole employment decision.

## Limitations and ethical use

The dataset is fictional and has no exit-date field, so this project does not
claim monthly or weekly forecasting. Observational associations do not prove
causation, and metrics can change with the data or split. Human review,
fairness checks, relevant context, and privacy safeguards are required before
using any risk signal in an HR process.
"""

st.title("👥 Employee Leave (Attrition) Risk Analytics")
st.caption("Business Intelligence workflow: Data → Information → Insight → Decision → Action")

with st.sidebar:
    st.header("Project Controls")
    uploaded = st.file_uploader("Upload the Kaggle CSV (optional)", type=["csv"])
    st.markdown(f"**Dataset:** [IBM HR Analytics on Kaggle]({KAGGLE_DATASET_URL})")
    st.divider()
    st.markdown(
        "**Problem statement**\n\n"
        "Identify factors associated with employee attrition, estimate attrition risk, "
        "and convert the findings into targeted HR retention actions."
    )

raw = load_dataset(uploaded)
if raw is None:
    st.error(
        "Dataset could not be loaded automatically. Download the CSV from Kaggle "
        "and upload it using the sidebar."
    )
    st.stop()

data = prepare_data(raw)
try:
    model, metrics, cm = build_model(data)
    scored = score_all(model, data)
except Exception as exc:
    st.error(f"Model could not be trained: {exc}")
    st.stop()

total = len(data)
leavers = int(data["AttritionFlag"].sum())
stayers = total - leavers
attrition_rate = leavers / total * 100
high_risk = int(((scored["RiskBand"] == "High") & (scored["AttritionFlag"] == 0)).sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Employees", f"{total:,}")
k2.metric("Employees Who Left", f"{leavers:,}")
k3.metric("Attrition Rate", f"{attrition_rate:.1f}%")
k4.metric("High-Risk Active Profiles", f"{high_risk:,}")

tabs = st.tabs([
    "Executive Overview", "Drivers & Trends", "Risk & Prediction",
    "Model Evaluation", "Project Report"
])

with tabs[0]:
    st.subheader("Executive Overview")
    left, right = st.columns(2)

    with left:
        st.markdown("### Attrition mix")
        st.bar_chart(pd.Series({"Stayed": stayers, "Left": leavers}, name="Employees"))

    with right:
        st.markdown("### Attrition by department")
        dept = rate_by(data, "Department")
        st.dataframe(
            dept[["Department", "Employees", "Leavers", "AttritionRate"]]
            .style.format({"AttritionRate": "{:.1f}%"}),
            use_container_width=True, hide_index=True
        )

    st.markdown("### Fact → Insight → Opportunity → Action")
    facts = [
        ("FACT", f"{attrition_rate:.1f}% of the {total:,} employee records are labelled as attrition."),
        ("INSIGHT", "Attrition is concentrated in identifiable employee segments rather than being uniform."),
        ("OPPORTUNITY", "Use department, overtime, satisfaction, tenure and compensation patterns to target retention resources."),
        ("ACTION", "Prioritise high-risk segments for human HR review and targeted retention interventions."),
    ]
    cols = st.columns(4)
    for col, (title, text) in zip(cols, facts):
        with col:
            st.markdown(f"**{title}**")
            st.info(text)

    st.warning(
        "The dataset has no employee exit-date field. The project therefore does not invent monthly or weekly attrition trends."
    )

with tabs[1]:
    st.subheader("Drivers, Segment Trends & Opportunities")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Overtime impact")
        ot = rate_by(data, "OverTime")
        st.dataframe(
            ot[["OverTime", "Employees", "Leavers", "AttritionRate"]]
            .style.format({"AttritionRate": "{:.1f}%"}),
            use_container_width=True, hide_index=True
        )
        st.bar_chart(ot.set_index("OverTime")["AttritionRate"])

    with c2:
        st.markdown("### Department attrition rate")
        st.bar_chart(rate_by(data, "Department").set_index("Department")["AttritionRate"])

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### Attrition by age band")
        st.bar_chart(rate_by(data, "AgeBand").set_index("AgeBand")["AttritionRate"])
    with c4:
        st.markdown("### Attrition by tenure band")
        st.bar_chart(rate_by(data, "TenureBand").set_index("TenureBand")["AttritionRate"])

    st.markdown("### Recommended HR actions")
    for action in [
        "Review excessive overtime in high-risk departments and roles.",
        "Create early-tenure retention check-ins for employees in the first few years.",
        "Strengthen engagement and work-life-balance interventions.",
        "Review compensation and career-progression issues in high-attrition roles.",
        "Use the prediction score as decision support, not as an automatic employment decision.",
    ]:
        st.write("• " + action)

with tabs[2]:
    st.subheader("Employee Risk Segmentation & Prediction")
    active = scored[scored["AttritionFlag"] == 0].copy()

    risk_counts = active["RiskBand"].value_counts().reindex(["Low", "Watch", "High"], fill_value=0)
    st.markdown("### Current active-employee risk distribution")
    st.bar_chart(risk_counts)

    display_cols = [
        c for c in [
            "Age", "Department", "JobRole", "OverTime", "JobSatisfaction",
            "WorkLifeBalance", "MonthlyIncome", "YearsAtCompany",
            "RiskProbability", "RiskBand"
        ] if c in active.columns
    ]
    st.markdown("### Highest-risk active profiles")
    st.dataframe(
        active.sort_values("RiskProbability", ascending=False).head(15)[display_cols]
        .style.format({"RiskProbability": "{:.1%}"}),
        use_container_width=True, hide_index=True
    )

    st.divider()
    st.markdown("### Single-employee prediction")
    cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
    editable = [
        c for c in [
            "Age", "BusinessTravel", "Department", "DistanceFromHome", "Education",
            "EducationField", "EnvironmentSatisfaction", "Gender", "JobInvolvement",
            "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus",
            "MonthlyIncome", "NumCompaniesWorked", "OverTime", "PercentSalaryHike",
            "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel",
            "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance",
            "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
            "YearsWithCurrManager"
        ] if c in data.columns
    ]

    values = {}
    form_cols = st.columns(3)
    with st.form("prediction_form"):
        for i, col in enumerate(editable):
            with form_cols[i % 3]:
                if col in cat_cols:
                    options = sorted(data[col].dropna().astype(str).unique().tolist())
                    values[col] = st.selectbox(col, options)
                else:
                    med = float(data[col].median())
                    integer_like = col in {
                        "Age", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
                        "JobInvolvement", "JobLevel", "JobSatisfaction",
                        "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
                        "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
                        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
                        "NumCompaniesWorked"
                    }
                    values[col] = st.number_input(
                        col, value=int(round(med)) if integer_like else med,
                        step=1 if integer_like else (100.0 if col == "MonthlyIncome" else 1.0)
                    )
        submitted = st.form_submit_button("Predict Leave / Attrition Risk")

    if submitted:
        # Start from a complete valid feature row so model columns not shown in
        # the compact UI (for example rate fields) are filled with dataset values.
        one = data.drop(
            columns=["AttritionFlag", "Attrition", "AgeBand", "TenureBand", "IncomeBand"],
            errors="ignore",
        ).iloc[[0]].copy()
        for field, value in values.items():
            if field in one.columns:
                one.loc[one.index[0], field] = value

        probability = float(model.predict_proba(one)[:, 1][0])
        band = "High" if probability >= 0.60 else ("Watch" if probability >= 0.30 else "Low")
        st.metric("Predicted Attrition Probability", f"{probability:.1%}")
        if band == "High":
            st.error("HIGH RISK — prioritise a human HR review.")
        elif band == "Watch":
            st.warning("WATCH — consider a retention check-in.")
        else:
            st.success("LOW RISK — continue normal employee engagement.")
        st.caption(
            "This prediction is a decision-support signal only and should not be the sole basis for employment decisions."
        )

with tabs[3]:
    st.subheader("Model Evaluation")
    metric_cols = st.columns(5)
    for col, (name, value) in zip(metric_cols, metrics.items()):
        col.metric(name, f"{value:.1%}")

    st.markdown("### Confusion matrix")
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual Stay", "Actual Leave"],
            columns=["Predicted Stay", "Predicted Leave"]
        ),
        use_container_width=True
    )

    st.markdown("### Methodology")
    st.write(
        "The model uses a stratified 80/20 train-test split. Numeric features are median-imputed "
        "and standardised. Categorical features are imputed and one-hot encoded. Logistic Regression "
        "with class balancing is used to give the minority attrition class greater weight."
    )

with tabs[4]:
    st.subheader("Project Report & Dashboard Blueprint")
    report = report_markdown(data, metrics, total, leavers, attrition_rate, high_risk)
    st.markdown(report)
    st.download_button(
        "Download this report as Markdown",
        data=report,
        file_name="employee_attrition_dashboard_report.md",
        mime="text/markdown",
    )
    st.info(
        "This report is generated from the currently loaded dataset. Upload a "
        "different compatible CSV to refresh the analysis and report."
    )

st.divider()
st.caption(
    "Educational project based on the IBM HR Analytics Employee Attrition & Performance dataset. "
    "The dataset is fictional and should not be presented as real IBM workforce data."
)
