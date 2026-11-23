from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "readmission_model.joblib"

FEATURES = [
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "insulin",
    "change",
    "diabetesMed",
    "admission_type_id"
]


def load_model():
    if not MODEL_PATH.exists():
        return None

    bundle = joblib.load(MODEL_PATH)

    return bundle


@app.route("/")
def home():
    return render_template(
        "index.html",
        form_data={},
        submitted=False
    )


@app.route("/predict", methods=["POST"])
def predict():

    form_data = request.form.to_dict()

    try:
        bundle = load_model()

        if bundle is None:
            raise FileNotFoundError(
                "Trained model not found. Please run train_model.py first."
            )

        pipeline = bundle["pipeline"]
        threshold = float(bundle["threshold"])

        row = {
            "age": form_data["age"],
            "time_in_hospital": int(form_data["time_in_hospital"]),
            "num_lab_procedures": int(form_data["num_lab_procedures"]),
            "num_procedures": int(form_data["num_procedures"]),
            "num_medications": int(form_data["num_medications"]),
            "number_outpatient": int(form_data["number_outpatient"]),
            "number_emergency": int(form_data["number_emergency"]),
            "number_inpatient": int(form_data["number_inpatient"]),
            "number_diagnoses": int(form_data["number_diagnoses"]),
            "insulin": form_data["insulin"],
            "change": form_data["change"],
            "diabetesMed": form_data["diabetesMed"],
            "admission_type_id": int(form_data["admission_type_id"])
        }

        X = pd.DataFrame([row], columns=FEATURES)

        probabilities = pipeline.predict_proba(X)[0]

        probability = float(probabilities[1])

        prediction_value = int(probability >= threshold)

        if prediction_value == 1:
            prediction = "Higher Readmission Likelihood"
            result_type = "high"
        else:
            prediction = "Lower Readmission Likelihood"
            result_type = "low"

        return render_template(
            "index.html",
            form_data=form_data,
            submitted=True,
            prediction=prediction,
            probability=probability,
            result_type=result_type
        )

    except Exception as e:

        return render_template(
            "index.html",
            form_data=form_data,
            submitted=False,
            error=str(e)
        )


if __name__ == "__main__":
   app.run(host="127.0.0.1", port=8000, debug=True)