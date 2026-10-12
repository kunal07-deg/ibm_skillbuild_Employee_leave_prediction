# Employee Leave (Attrition) Risk Analytics

## Project overview

This project is an interactive Streamlit dashboard that analyses employee
attrition and estimates risk using the IBM HR Analytics Employee Attrition &
Performance practice dataset. It follows the workflow:

**Data → Information → Insight → Decision → Action**

The dashboard provides executive KPIs, department and segment analysis,
employee risk scoring, single-employee prediction, model evaluation, and a
generated project report.

> “Employee leave” means employee attrition/exit because the dataset target is
> named `Attrition`.

## Dataset

IBM HR Analytics Employee Attrition & Performance:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

The dataset is fictional and intended for analytics practice. It should not be
presented as real IBM workforce data. The application can load the CSV from
the project folder, `data/`, a public Kaggle download, or the sidebar uploader.

## Technologies

- Python 3.11+
- Pandas and NumPy
- Scikit-learn
- Streamlit

## Machine-learning workflow

1. Load and validate the dataset.
2. Remove duplicate, identifier, and constant fields.
3. Create age, tenure, income, and satisfaction features.
4. Split data with a stratified 80/20 train-test split.
5. Impute and scale numeric features.
6. Impute and one-hot encode categorical features.
7. Train class-balanced Logistic Regression.
8. Evaluate Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
9. Generate employee-level risk probabilities and risk bands.

## Dashboard sections

- **Executive Overview:** KPI cards, attrition mix, department comparison,
  and Fact → Insight → Opportunity → Action.
- **Drivers & Trends:** overtime, department, age-band, and tenure-band
  comparisons with retention recommendations.
- **Risk & Prediction:** active-employee risk distribution, high-risk profiles,
  and an individual prediction form.
- **Model Evaluation:** model metrics, confusion matrix, and methodology.
- **Project Report:** current-data report and downloadable Markdown blueprint.

## Setup and run

```bash
python -m pip install -r requirements.txt
# Recommended when the Streamlit command is not on PATH
python -m streamlit run KunalKumarLilha_EmployeeLeavePrediction.py
# Alternative when Streamlit is on PATH
streamlit run KunalKumarLilha_EmployeeLeavePrediction.py
```

If automatic dataset download is unavailable, download the Kaggle CSV and
upload it in the dashboard sidebar.

## Limitations and ethical use

The data has no exit-date field, so the project does not claim monthly or
weekly forecasting. Observational relationships do not prove causation and
model metrics can vary by dataset and split. Risk scores are decision-support
signals only; they must not be the sole basis for termination, hiring,
promotion, compensation, or disciplinary decisions. Human review, fairness,
privacy, and relevant context are required.

## Submission files (AICTE | IBM SkillsBuild)

| Required name | File | Purpose |
|---|---|---|
| `YourName_ProjectName.py` | `KunalKumarLilha_EmployeeLeavePrediction.py` | Complete project code |
| `requirements.txt` | `requirements.txt` | Python dependency list |
| `YourName_ProjectReport.docx` | `KunalKumarLilha_ProjectReport.docx` | Full project report with methodology, UI walkthrough, and findings |
| `README.md` | `README.md` | Project overview and setup instructions |
