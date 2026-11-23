# Hospital Readmission Prediction — Data Mining Mini Project

## What this project does
A Flask web application with an HTML/CSS/JavaScript front end and a Python/scikit-learn back end. It trains a Random Forest classifier on the UCI Diabetes 130-US Hospitals dataset and estimates the likelihood of readmission within 30 days.

**Important:** This is an academic prototype using historical data, not a medical tool. Do not enter real patient-identifying information or use the output for care decisions.

## 1. Requirements
- Python 3.10 or newer
- VS Code (recommended) or another code editor
- A browser
- Internet connection to download the dataset and install Python packages

## 2. Dataset
Download `diabetic_data.csv` from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008

Place the CSV at:
`data/diabetic_data.csv`

The target column is `readmitted`: `<30` means readmission within 30 days; `>30` means after 30 days; `NO` means no recorded readmission. This project turns it into a binary classification target: `<30` = 1, other values = 0.

## 3. Setup (Windows)
Open this project folder in VS Code. Open Terminal and run:

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

## 4. Setup (macOS)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open the local address printed by Flask, usually http://127.0.0.1:5000

## 5. Files and responsibilities
- `train_model.py`: loads and cleans the dataset, prepares features, splits patients into train/test groups, trains/evaluates the classifier, saves the model and metrics.
- `app.py`: Flask back end. Receives the web form, converts values to a DataFrame, calls the saved model and sends the result to the page.
- `templates/index.html`: front-end structure and patient input form.
- `static/css/style.css`: page layout, colours, spacing and responsive styling.
- `static/js/script.js`: browser-side input validation.
- `data/`: put the downloaded CSV here.
- `models/`: generated model and evaluation metrics are saved here.

## 6. Data mining / machine learning concepts
1. Data collection — historical hospital encounter dataset.
2. Data cleaning — missing values are imputed; irrelevant identifiers are excluded.
3. Feature selection — a small set of admission, treatment and prior-visit fields is used.
4. Encoding — One-Hot Encoding converts categorical fields into numeric columns.
5. Classification — Random Forest predicts the binary target.
6. Train/test split — patients are grouped so the same patient is less likely to appear in both sets.
7. Evaluation — accuracy, precision, recall, F1-score and ROC-AUC are printed and saved.
8. Prediction — the trained pipeline processes one new form submission.

## 7. Important academic notes
- The data is from 130 US hospitals, 1999–2008, and consists of diabetic inpatient encounters. State this limitation in your report.
- This is a simplified student model. The 0.5 probability threshold is a demonstration choice, not a clinically validated threshold.
- Because readmission is relatively uncommon, do not report accuracy alone. Discuss precision, recall, F1 and ROC-AUC.
- Do not claim the model diagnoses disease or proves that a patient will be readmitted.
- The form uses a subset of dataset fields. Admission type options are simplified numeric codes for demonstration; the model itself learns from the original codes.
