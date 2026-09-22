import os
import uuid
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier, XGBRegressor

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Persistent state files. Do not rely on in-memory globals because hosted
# Gunicorn/Flask processes can restart between requests.
DATASET_STATE_PATH = os.path.join(UPLOAD_DIR, "current_training_dataset.pkl")
TEST_DATASET_STATE_PATH = os.path.join(UPLOAD_DIR, "current_test_dataset.pkl")
MODEL_STATE_PATH = os.path.join(MODEL_DIR, "sih_ess_complete_model.joblib")

RANDOM_STATE = 42
HOLDOUT_SIZE = 0.20
FAILURE_THRESHOLD = 0.50
ANOMALY_WARNING_THRESHOLD = 0.70
ANOMALY_REJECT_THRESHOLD = 0.85

DATASET = None
DATASET_NAME = None
MODEL_PACKAGE = None
LAST_PREDICTIONS = None
TEST_DATASET = None
TEST_DATASET_NAME = None
LAST_TEST_PREDICTIONS = None

def save_training_dataset(df, name):
    df.to_pickle(DATASET_STATE_PATH)
    meta_path = os.path.join(UPLOAD_DIR, "current_training_dataset_name.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(name or "")


def load_training_dataset():
    global DATASET, DATASET_NAME
    if DATASET is not None:
        return DATASET
    if not os.path.exists(DATASET_STATE_PATH):
        return None
    try:
        DATASET = pd.read_pickle(DATASET_STATE_PATH)
        meta_path = os.path.join(UPLOAD_DIR, "current_training_dataset_name.txt")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                DATASET_NAME = f.read().strip() or None
        return DATASET
    except Exception:
        DATASET = None
        DATASET_NAME = None
        return None


def save_test_dataset(df, name):
    df.to_pickle(TEST_DATASET_STATE_PATH)
    meta_path = os.path.join(UPLOAD_DIR, "current_test_dataset_name.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(name or "")


def load_test_dataset():
    global TEST_DATASET, TEST_DATASET_NAME
    if TEST_DATASET is not None:
        return TEST_DATASET
    if not os.path.exists(TEST_DATASET_STATE_PATH):
        return None
    try:
        TEST_DATASET = pd.read_pickle(TEST_DATASET_STATE_PATH)
        meta_path = os.path.join(UPLOAD_DIR, "current_test_dataset_name.txt")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                TEST_DATASET_NAME = f.read().strip() or None
        return TEST_DATASET
    except Exception:
        TEST_DATASET = None
        TEST_DATASET_NAME = None
        return None


def load_model_package():
    global MODEL_PACKAGE
    if MODEL_PACKAGE is not None:
        return MODEL_PACKAGE
    if not os.path.exists(MODEL_STATE_PATH):
        return None
    try:
        MODEL_PACKAGE = joblib.load(MODEL_STATE_PATH)
        return MODEL_PACKAGE
    except Exception:
        MODEL_PACKAGE = None
        return None


def restore_persistent_state():
    load_training_dataset()
    load_test_dataset()
    load_model_package()


FAILURE_FEATURES = [
    "burn_in_temperature_c", "burn_in_duration_hr", "thermal_acceleration_factor",
    "measured_vdd_v_initial", "measured_vdd_v_early_mean",
    "measured_idd_ma_initial", "measured_idd_ma_early_mean",
    "leakage_current_na_initial", "leakage_current_na_early_mean",
    "junction_temp_rise_c_initial", "junction_temp_rise_c_early_mean"
]
DRIFT_FEATURES = FAILURE_FEATURES.copy()


def read_dataset(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".xls", ".xlsx"):
        try:
            return pd.read_excel(path)
        except Exception:
            return pd.read_csv(path, sep="\t")
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_csv(path)


def prepare_target(df):
    df = df.copy()
    if "cumulative_fail_flag" not in df.columns:
        raise ValueError("Dataset must contain cumulative_fail_flag.")
    target = df["cumulative_fail_flag"]
    if target.dtype == bool:
        out = target.astype(int)
    else:
        out = target.astype(str).str.strip().str.lower().map({
            "true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0,
            "pass": 0, "fail": 1
        })
        out = out.fillna(pd.to_numeric(target, errors="coerce"))
    if out.isna().any():
        raise ValueError("Could not convert cumulative_fail_flag to 0/1.")
    df["target"] = out.astype(int)
    return df


def available_features(df, candidates):
    features = [c for c in candidates if c in df.columns]
    if len(features) < 5:
        raise ValueError("At least 5 leakage-safe numeric model features are required.")
    return features


def numeric_matrix(df, features):
    X = df[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def build_classifier(scale_pos_weight):
    return XGBClassifier(
        n_estimators=700, max_depth=4, learning_rate=0.03,
        subsample=0.90, colsample_bytree=0.90, min_child_weight=4,
        gamma=0.05, reg_alpha=0.05, reg_lambda=3.0,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
        n_jobs=-1, tree_method="hist"
    )


def clean_json_value(value):
    if isinstance(value, dict):
        return {str(k): clean_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json_value(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def classification_metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "threshold": float(threshold)
    }


def train_failure_model(df, features):
    X, y = numeric_matrix(df, features), df["target"]
    if y.nunique() < 2:
        raise ValueError("Failure target must contain both PASS (0) and FAIL (1).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=HOLDOUT_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)
    negative = int(np.sum(y_train == 0))
    positive = int(np.sum(y_train == 1))
    model = build_classifier(negative / max(positive, 1))
    model.fit(X_train_i, y_train, eval_set=[(X_test_i, y_test)], verbose=False)

    train_prob = model.predict_proba(X_train_i)[:, 1]
    test_prob = model.predict_proba(X_test_i)[:, 1]
    train_metrics = classification_metrics(y_train, train_prob, FAILURE_THRESHOLD)
    holdout_metrics = classification_metrics(y_test, test_prob, FAILURE_THRESHOLD)

    return {
        "model": model, "imputer": imputer, "features": features,
        "threshold": FAILURE_THRESHOLD, "model_type": "XGBoost Classifier",
        "target": "cumulative_fail_flag",
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "train_rows": int(len(y_train)), "holdout_rows": int(len(y_test))
    }


def normalize_anomaly_scores(raw_scores, low=None, high=None):
    raw_scores = np.asarray(raw_scores, dtype=float)
    low = float(np.percentile(raw_scores, 1)) if low is None else float(low)
    high = float(np.percentile(raw_scores, 99)) if high is None else float(high)
    clipped = np.clip(raw_scores, low, high)
    normalized = 1 - ((clipped - low) / max(high - low, 1e-9))
    return np.clip(normalized, 0, 1), low, high


def train_isolation_forest(df, features):
    X, _ = train_test_split(df, test_size=HOLDOUT_SIZE, stratify=df["target"], random_state=RANDOM_STATE)
    X = numeric_matrix(X, features)
    imputer = SimpleImputer(strategy="median")
    X_i = imputer.fit_transform(X)
    model = IsolationForest(
        n_estimators=350, contamination="auto", max_samples="auto",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_i)
    raw = model.score_samples(X_i)
    _, low, high = normalize_anomaly_scores(raw)
    return {
        "model": model, "imputer": imputer, "features": features,
        "score_low": low, "score_high": high,
        "model_type": "Isolation Forest"
    }


def train_drift_model(df, features):
    required = ["measured_vdd_v_initial", "measured_vdd_v_final"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    d = df.copy()
    d["vdd_drift_168h"] = (
        pd.to_numeric(d["measured_vdd_v_final"], errors="coerce")
        - pd.to_numeric(d["measured_vdd_v_initial"], errors="coerce")
    )
    d = d.dropna(subset=["vdd_drift_168h"])
    if len(d) < 10:
        raise ValueError("Not enough rows with valid VDD initial/final values.")
    X, y = numeric_matrix(d, features), d["vdd_drift_168h"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=HOLDOUT_SIZE, random_state=RANDOM_STATE
    )
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)
    model = XGBRegressor(
        n_estimators=600, max_depth=4, learning_rate=0.03,
        subsample=0.90, colsample_bytree=0.90, min_child_weight=4,
        gamma=0.05, reg_alpha=0.05, reg_lambda=3.0,
        objective="reg:squarederror", eval_metric="mae",
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"
    )
    model.fit(X_train_i, y_train, eval_set=[(X_test_i, y_test)], verbose=False)
    train_pred = model.predict(X_train_i)
    test_pred = model.predict(X_test_i)
    train_metrics = {
        "mae": float(mean_absolute_error(y_train, train_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
        "r2": float(r2_score(y_train, train_pred))
    }
    holdout_metrics = {
        "mae": float(mean_absolute_error(y_test, test_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "r2": float(r2_score(y_test, test_pred))
    }
    return {
        "model": model, "imputer": imputer, "features": features,
        "target": "vdd_drift_168h", "model_type": "XGBoost Regressor",
        "train_metrics": train_metrics, "holdout_metrics": holdout_metrics,
        "train_rows": int(len(y_train)), "holdout_rows": int(len(y_test))
    }


def train_all(df):
    df = prepare_target(df)
    if "component_id" not in df.columns:
        raise ValueError("Column 'component_id' was not found.")
    failure_features = available_features(df, FAILURE_FEATURES)
    drift_features = available_features(df, DRIFT_FEATURES)
    failure = train_failure_model(df, failure_features)
    anomaly = train_isolation_forest(df, failure_features)
    drift = train_drift_model(df, drift_features)
    package = {
        "failure_model": failure, "anomaly_model": anomaly, "drift_model": drift,
        "version": "SIH-ESS-ML-2.0-FINAL", "random_state": RANDOM_STATE,
        "holdout_fraction": HOLDOUT_SIZE, "dataset_name": DATASET_NAME
    }
    joblib.dump(package, MODEL_STATE_PATH)
    return package, df


def shap_explanation(package, row):
    if not SHAP_AVAILABLE:
        return []
    artifact = package["failure_model"]
    try:
        X = numeric_matrix(pd.DataFrame([row]), artifact["features"])
        X_i = artifact["imputer"].transform(X)
        values = np.asarray(shap.TreeExplainer(artifact["model"]).shap_values(X_i))
        if values.ndim == 2:
            values = values[0]
        out = []
        for feature, value, sv in zip(artifact["features"], X_i[0], values):
            out.append({"feature": feature, "value": float(value), "shap_value": float(sv),
                        "impact": "INCREASES RISK" if sv > 0 else "DECREASES RISK"})
        return sorted(out, key=lambda x: abs(x["shap_value"]), reverse=True)[:8]
    except Exception:
        return []


def predict_row(row, package, include_actual=True, include_shap=True):
    failure, anomaly, drift = package["failure_model"], package["anomaly_model"], package["drift_model"]
    Xf = failure["imputer"].transform(numeric_matrix(pd.DataFrame([row]), failure["features"]))
    fp = float(failure["model"].predict_proba(Xf)[0, 1])
    failure_prediction = int(fp >= failure["threshold"])

    Xa = anomaly["imputer"].transform(numeric_matrix(pd.DataFrame([row]), anomaly["features"]))
    raw = float(anomaly["model"].score_samples(Xa)[0])
    anomaly_score = float(normalize_anomaly_scores(np.array([raw]), anomaly["score_low"], anomaly["score_high"])[0][0])

    Xd = drift["imputer"].transform(numeric_matrix(pd.DataFrame([row]), drift["features"]))
    drift_value = float(drift["model"].predict(Xd)[0])
    initial_vdd = pd.to_numeric(pd.Series([row.get("measured_vdd_v_initial")]), errors="coerce").iloc[0]
    predicted_vdd = float(initial_vdd + drift_value) if pd.notna(initial_vdd) else None

    # Risk status is driven by the supervised failure model. Anomaly severity is a
    # separate diagnostic signal and no longer creates a contradictory failure label.
    if fp >= 0.75:
        status = "REJECT"
    elif fp >= 0.50:
        status = "WARNING"
    else:
        status = "PASS"

    result = {
        "component_id": str(row["component_id"]),
        "failure_probability": fp,
        "failure_prediction": failure_prediction,
        "anomaly_score": anomaly_score,
        "anomaly_level": "HIGH" if anomaly_score >= ANOMALY_REJECT_THRESHOLD else ("MODERATE" if anomaly_score >= ANOMALY_WARNING_THRESHOLD else "LOW"),
        "predicted_vdd_drift": drift_value,
        "predicted_vdd_168h": predicted_vdd,
        "status": status,
        "vdd_series": [
            {"label": "Initial", "value": float(pd.to_numeric(pd.Series([row.get("measured_vdd_v_initial")]), errors="coerce").iloc[0]) if pd.notna(pd.to_numeric(pd.Series([row.get("measured_vdd_v_initial")]), errors="coerce").iloc[0]) else None},
            {"label": "Early", "value": float(pd.to_numeric(pd.Series([row.get("measured_vdd_v_early_mean")]), errors="coerce").iloc[0]) if pd.notna(pd.to_numeric(pd.Series([row.get("measured_vdd_v_early_mean")]), errors="coerce").iloc[0]) else None},
            {"label": "Late", "value": float(pd.to_numeric(pd.Series([row.get("measured_vdd_v_late_mean")]), errors="coerce").iloc[0]) if pd.notna(pd.to_numeric(pd.Series([row.get("measured_vdd_v_late_mean")]), errors="coerce").iloc[0]) else None},
            {"label": "Final", "value": float(pd.to_numeric(pd.Series([row.get("measured_vdd_v_final")]), errors="coerce").iloc[0]) if pd.notna(pd.to_numeric(pd.Series([row.get("measured_vdd_v_final")]), errors="coerce").iloc[0]) else None},
            {"label": "Predicted 168h", "value": predicted_vdd}
        ]
    }
    if include_shap:
        result["shap"] = shap_explanation(package, row)
    if include_actual and "target" in row:
        result["actual_failure"] = int(row["target"])
        if int(row["target"]) == failure_prediction:
            result["classification_result"] = "TRUE POSITIVE" if failure_prediction else "TRUE NEGATIVE"
        else:
            result["classification_result"] = "FALSE POSITIVE" if failure_prediction else "FALSE NEGATIVE"
    return clean_json_value(result)


def batch_test_predict(df, package):
    failure, anomaly, drift = package["failure_model"], package["anomaly_model"], package["drift_model"]
    Xf = failure["imputer"].transform(numeric_matrix(df, failure["features"]))
    probs = failure["model"].predict_proba(Xf)[:, 1]
    fpred = (probs >= failure["threshold"]).astype(int)
    Xa = anomaly["imputer"].transform(numeric_matrix(df, anomaly["features"]))
    raw = anomaly["model"].score_samples(Xa)
    ascore = normalize_anomaly_scores(raw, anomaly["score_low"], anomaly["score_high"])[0]
    Xd = drift["imputer"].transform(numeric_matrix(df, drift["features"]))
    dpred = drift["model"].predict(Xd)
    initial = pd.to_numeric(df["measured_vdd_v_initial"], errors="coerce").to_numpy()
    predicted_vdd = initial + dpred
    out = pd.DataFrame({
        "component_id": df["component_id"].astype(str),
        "failure_probability": probs,
        "failure_prediction": fpred,
        "predicted_result": np.where(fpred == 1, "FAIL", "PASS"),
        "anomaly_severity": ascore,
        "anomaly_level": np.where(ascore >= ANOMALY_REJECT_THRESHOLD, "HIGH", np.where(ascore >= ANOMALY_WARNING_THRESHOLD, "MODERATE", "LOW")),
        "predicted_vdd_drift_168h": dpred,
        "predicted_vdd_168h": predicted_vdd,
        "actual_failure": df["target"].astype(int).to_numpy()
    })
    out["classification_result"] = np.select(
        [out.failure_prediction.eq(1) & out.actual_failure.eq(1),
         out.failure_prediction.eq(0) & out.actual_failure.eq(0),
         out.failure_prediction.eq(1) & out.actual_failure.eq(0)],
        ["TRUE POSITIVE", "TRUE NEGATIVE", "FALSE POSITIVE"], default="FALSE NEGATIVE"
    )
    out["status"] = np.where(out.failure_probability >= .75, "REJECT", np.where(out.failure_probability >= .50, "WARNING", "PASS"))
    return out


@app.route("/")
def index():
    return render_template("index.html", shap_available=SHAP_AVAILABLE)


@app.route("/api/status")
def api_status():
    dataset = load_training_dataset()
    test_dataset = load_test_dataset()
    package = load_model_package()
    return jsonify(clean_json_value({
        "trained": package is not None,
        "dataset": DATASET_NAME,
        "rows": int(len(dataset)) if dataset is not None else 0,
        "test_dataset": TEST_DATASET_NAME,
        "test_rows": int(len(test_dataset)) if test_dataset is not None else 0,
        "shap_available": SHAP_AVAILABLE
    }))


@app.route("/api/upload", methods=["POST"])
def upload():
    global DATASET, DATASET_NAME
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "Select a training dataset file first."}), 400
    file = request.files["file"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    file.save(path)
    try:
        df = read_dataset(path)
        if "component_id" not in df.columns:
            raise ValueError("Dataset must contain component_id.")
        df = prepare_target(df)
        DATASET, DATASET_NAME = df, filename
        save_training_dataset(DATASET, DATASET_NAME)
        return jsonify(clean_json_value({
            "message": "Training dataset loaded.", "filename": filename,
            "rows": len(df), "columns": list(df.columns),
            "preview": df.head(8).drop(columns=["target"], errors="ignore").to_dict(orient="records")
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/train", methods=["POST"])
def train():
    global MODEL_PACKAGE, DATASET, LAST_PREDICTIONS
    dataset = load_training_dataset()
    if dataset is None:
        return jsonify({"error": "Upload a training dataset first."}), 400
    try:
        package, DATASET = train_all(dataset)
        MODEL_PACKAGE = package
        joblib.dump(MODEL_PACKAGE, MODEL_STATE_PATH)
        LAST_PREDICTIONS = None
        return jsonify(clean_json_value({
            "message": "Final model trained and saved.",
            "failure_train_metrics": package["failure_model"]["train_metrics"],
            "failure_holdout_metrics": package["failure_model"]["holdout_metrics"],
            "drift_train_metrics": package["drift_model"]["train_metrics"],
            "drift_holdout_metrics": package["drift_model"]["holdout_metrics"],
            "split": {"train_fraction": 0.80, "holdout_fraction": 0.20,
                      "train_rows": package["failure_model"]["train_rows"],
                      "holdout_rows": package["failure_model"]["holdout_rows"]},
            "threshold": package["failure_model"]["threshold"],
            "features": package["failure_model"]["features"]
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/components")
def components():
    dataset = load_training_dataset()
    return jsonify({"components": dataset["component_id"].astype(str).tolist() if dataset is not None else []})


@app.route("/api/predict/<component_id>")
def predict(component_id):
    package = load_model_package()
    dataset = load_training_dataset()
    if package is None or dataset is None:
        return jsonify({"error": "Train the model first."}), 400
    rows = dataset[dataset["component_id"].astype(str) == str(component_id)]
    if rows.empty:
        return jsonify({"error": "Component not found."}), 404
    return jsonify(clean_json_value(predict_row(rows.iloc[0], package, True, True)))


@app.route("/api/predict-all", methods=["POST"])
def predict_all():
    global LAST_PREDICTIONS
    package = load_model_package()
    dataset = load_training_dataset()
    if package is None or dataset is None:
        return jsonify({"error": "Train the model first."}), 400
    LAST_PREDICTIONS = batch_test_predict(dataset, package)
    path = os.path.join(UPLOAD_DIR, "sih_component_predictions.csv")
    LAST_PREDICTIONS.to_csv(path, index=False)
    return jsonify(clean_json_value({
        "message": "Training-set batch prediction complete.", "count": len(LAST_PREDICTIONS),
        "status_distribution": LAST_PREDICTIONS["status"].value_counts().to_dict()
    }))


@app.route("/api/download-predictions")
def download_predictions():
    path = os.path.join(UPLOAD_DIR, "sih_component_predictions.csv")
    if LAST_PREDICTIONS is None or not os.path.exists(path):
        return jsonify({"error": "Run training-set batch prediction first."}), 400
    return send_file(path, as_attachment=True, download_name="sih_component_predictions.csv")


@app.route("/api/test-upload", methods=["POST"])
def test_upload():
    global TEST_DATASET, TEST_DATASET_NAME, LAST_TEST_PREDICTIONS
    if load_model_package() is None:
        return jsonify({"error": "Train the final model before uploading custom test data."}), 400
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "Select a custom test dataset first."}), 400
    file = request.files["file"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, f"test_{uuid.uuid4().hex}_{filename}")
    file.save(path)
    try:
        df = prepare_target(read_dataset(path))
        if "component_id" not in df.columns:
            raise ValueError("Test dataset must contain component_id.")
        TEST_DATASET, TEST_DATASET_NAME, LAST_TEST_PREDICTIONS = df, filename, None
        save_test_dataset(TEST_DATASET, TEST_DATASET_NAME)
        return jsonify(clean_json_value({
            "message": "Custom test dataset loaded.", "filename": filename,
            "rows": len(df), "columns": list(df.columns),
            "preview": df.head(8).drop(columns=["target"], errors="ignore").to_dict(orient="records")
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/test-components")
def test_components():
    dataset = load_test_dataset()
    return jsonify({"components": dataset["component_id"].astype(str).tolist() if dataset is not None else []})


@app.route("/api/test-evaluate", methods=["POST"])
def test_evaluate():
    global LAST_TEST_PREDICTIONS
    package = load_model_package()
    test_dataset = load_test_dataset()
    if package is None:
        return jsonify({"error": "Train the final model first."}), 400
    if test_dataset is None:
        return jsonify({"error": "Upload a custom test dataset first."}), 400
    try:
        # Vectorized inference: no per-row SHAP, no huge JSON payload, and no NaN JSON failure.
        LAST_TEST_PREDICTIONS = batch_test_predict(test_dataset, package)
        y = test_dataset["target"].astype(int).to_numpy()
        p = LAST_TEST_PREDICTIONS["failure_probability"].to_numpy()
        c = classification_metrics(y, p, package["failure_model"]["threshold"])

        actual_drift = pd.to_numeric(test_dataset["measured_vdd_v_final"], errors="coerce") - pd.to_numeric(test_dataset["measured_vdd_v_initial"], errors="coerce")
        pred_drift = LAST_TEST_PREDICTIONS["predicted_vdd_drift_168h"].to_numpy()
        valid = actual_drift.notna().to_numpy() & np.isfinite(pred_drift)
        drift_metrics = None
        if valid.sum() >= 2:
            drift_metrics = {
                "mae": float(mean_absolute_error(actual_drift.to_numpy()[valid], pred_drift[valid])),
                "rmse": float(np.sqrt(mean_squared_error(actual_drift.to_numpy()[valid], pred_drift[valid]))),
                "r2": float(r2_score(actual_drift.to_numpy()[valid], pred_drift[valid]))
            }

        path = os.path.join(UPLOAD_DIR, "sih_ess_custom_test_predictions.csv")
        LAST_TEST_PREDICTIONS.to_csv(path, index=False)
        result_counts = LAST_TEST_PREDICTIONS["classification_result"].value_counts().to_dict()
        return jsonify(clean_json_value({
            "message": "Custom test evaluation complete.", "dataset": TEST_DATASET_NAME,
            "count": len(LAST_TEST_PREDICTIONS), "classification_metrics": c,
            "drift_metrics": drift_metrics, "classification_results": result_counts,
            "status_distribution": LAST_TEST_PREDICTIONS["status"].value_counts().to_dict(),
            "preview": LAST_TEST_PREDICTIONS.head(50).to_dict(orient="records")
        }))
    except Exception as e:
        app.logger.exception("Custom test evaluation failed")
        return jsonify({"error": f"Test evaluation failed: {e}"}), 500


@app.route("/api/test-predict/<component_id>")
def test_predict(component_id):
    package = load_model_package()
    test_dataset = load_test_dataset()
    if package is None or test_dataset is None:
        return jsonify({"error": "Train the model and upload custom test data first."}), 400
    rows = test_dataset[test_dataset["component_id"].astype(str) == str(component_id)]
    if rows.empty:
        return jsonify({"error": "Test component not found."}), 404
    result = predict_row(rows.iloc[0], package, True, True)
    row = rows.iloc[0]
    initial = pd.to_numeric(pd.Series([row.get("measured_vdd_v_initial")]), errors="coerce").iloc[0]
    final = pd.to_numeric(pd.Series([row.get("measured_vdd_v_final")]), errors="coerce").iloc[0]
    if pd.notna(initial) and pd.notna(final):
        result["actual_vdd_drift"] = float(final - initial)
        result["drift_error"] = float(result["predicted_vdd_drift"] - result["actual_vdd_drift"])
    result["evaluation_dataset"] = TEST_DATASET_NAME
    return jsonify(clean_json_value(result))


@app.route("/api/download-test-predictions")
def download_test_predictions():
    path = os.path.join(UPLOAD_DIR, "sih_ess_custom_test_predictions.csv")
    if LAST_TEST_PREDICTIONS is None or not os.path.exists(path):
        return jsonify({"error": "Evaluate the custom test set first."}), 400
    return send_file(path, as_attachment=True, download_name="sih_ess_custom_test_predictions.csv")


# Restore any state that was persisted by an earlier request.
# This runs when a Gunicorn worker starts/restarts.
restore_persistent_state()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
