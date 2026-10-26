# FraudShield AI 🛡️

**An AI-powered fraud monitoring platform for screening transactions, prioritizing suspicious cases, and tracking investigation decisions.**

FraudShield AI helps users review transaction risk through an interactive dashboard. It combines machine learning-based fraud scoring with a review queue, investigation notes, reporting, and an audit trail.

## ✨ Features

- **Transaction Screening:** Upload transaction data in CSV format and review risk scores.
- **Risk Prioritization:** Identify potentially suspicious transactions for further investigation.
- **Review Queue:** Search, filter, and manage flagged cases.
- **Decision Tracking:** Record review outcomes and investigation notes.
- **Audit Trail:** Track recorded review activities and decisions.
- **Reports & Exports:** Review screening results and export data.
- **Model Explainability:** Explore information that helps interpret model predictions.
- **False Positive Analysis:** Review cases that may have been incorrectly flagged.

## 🧰 Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Joblib
- SQLite
- Machine Learning

## 🤖 Machine Learning

The project explores classification models for credit card fraud detection, including Logistic Regression and Random Forest.

Model evaluation uses metrics such as:

- Precision
- Recall
- F1-score
- Average Precision
- ROC-AUC

The classification threshold is configurable to help balance fraud detection against false alarms.

## 🏗️ Application Workflow

1. Upload transaction data.
2. Generate transaction risk scores.
3. Identify and prioritize flagged cases.
4. Review cases and record decisions.
5. Track actions through the audit trail.
6. Analyze results and export reports.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/fraudshield-ai.git
cd fraudshield-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

The application should open in your browser.

## 📁 Project Structure

```text
fraudshield-ai/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
├── data/
└── supporting scripts
```

The actual folder structure may differ depending on the project files included in the repository.

## 🔒 Data & Responsible Use

The application is intended to support fraud review, not replace human investigation. A flagged transaction is a risk indicator, not proof of fraud.

Do not upload private customer data, credentials, API keys, or sensitive transaction records to a public repository.

## 🔮 Future Improvements

- Deploy the dashboard for public demonstration.
- Add an AI-assisted investigation summary.
- Improve monitoring of model performance and false positives.
- Add role-based access and stronger production security controls.

---

**FraudShield AI** — Machine learning for smarter transaction review.
