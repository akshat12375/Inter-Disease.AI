import os
import io
import base64
import pickle
from pathlib import Path
from flask import Flask, render_template, request, flash
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
app.secret_key = "replace_this_with_a_random_secret"



diabetes_predict = joblib.load("diabetes_model.pkl")
heart_predict=joblib.load("heart_model.pkl")
kidney_predict=joblib.load("kidney_model.pkl")
liver_predict=joblib.load("liver_model_final.pkl")

BASE_DIR = Path(__file__).resolve().parent

# -------------------------
# Load model bundle
# -------------------------
BUNDLE_PATH = BASE_DIR / "model_bundle1.pkl"
if not BUNDLE_PATH.exists():
    raise FileNotFoundError(f"Bundle file not found at {BUNDLE_PATH}")

with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)

model_path = BASE_DIR / bundle.get("model", "multi_disease_model.h5")
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at {model_path}")

scaler = bundle["scaler"]
features = bundle["features"]
targets = bundle["targets"]
feature_means = bundle.get("feature_means", {feat: 0.0 for feat in features})

# -------------------------
# Load Keras model
# -------------------------
model = load_model(str(model_path))

# -------------------------
# Prepare background for SHAP
# -------------------------
mean_vec_raw = np.array([feature_means[f] for f in features], dtype=float).reshape(1, -1)
try:
    background_scaled = scaler.transform(mean_vec_raw)
except Exception:
    background_scaled = np.zeros_like(mean_vec_raw)

background_for_shap = np.tile(background_scaled, (10, 1))

# -------------------------
# Initialize SHAP explainer
# -------------------------
explainer = None
explainer_type = None
try:
    explainer = shap.Explainer(model.predict, background_for_shap)
    explainer_type = "unified"
    print("SHAP Explainer initialized (unified).")
except Exception as e:
    try:
        explainer = shap.DeepExplainer(model, background_for_shap)
        explainer_type = "deep"
        print("SHAP DeepExplainer initialized as fallback.")
    except Exception as e2:
        explainer = None
        explainer_type = None
        print("SHAP initialization failed, will use permutation fallback.", e, e2)

defaults = {feat: "" for feat in features}

# -------------------------
# Permutation importance fallback
# -------------------------
def permutation_importance_for_instance(X_scaled: np.ndarray):
    base_pred = model.predict(X_scaled, verbose=0)[0]
    n_features = X_scaled.shape[1]
    importances = np.zeros(n_features, dtype=float)
    mean_scaled = background_scaled.ravel()
    for i in range(n_features):
        Xp = X_scaled.copy()
        Xp[0, i] = mean_scaled[i]
        try:
            new_pred = model.predict(Xp, verbose=0)[0]
            importances[i] = np.sum(np.abs(base_pred - new_pred))
        except Exception:
            importances[i] = 0.0
    return importances

# -------------------------
# Flask routes
# -------------------------



@app.route('/') # instancing one page (homepage)
def home():
    return render_template("home.html")
# ^^ open home.html, then see that it extends layout.
# render home page.

@app.route('/diabetes') # instancing child page
def diabetes():
    return render_template("diabetes.html")



@app.route('/heartdisease/') # instancing child page
def heartdisease():
    return render_template("heartdisease.html")

@app.route('/inter_disease/',methods=["GET"]) # instancing child page
def inter_disease():
    return render_template("temporary.html", features=features, defaults=defaults,
                           prediction=None, shap_results=None, shap_plot=None,
                           shap_summary=None, shap_insight_text=None)


@app.route('/kidneydisease/') # instancing child page
def kidneydisease():
    return render_template("kidneydisease.html")


@app.route('/liverdisease/') # instancing child page
def liverdisease():
    return render_template("liverdisease.html")

@app.route('/predictheart/', methods=['POST'])
def predictheart():

    # ----------------------------
    # Original prediction
    # ----------------------------
    int_features = [x for x in request.form.values()]
    processed_feature_heart = [np.array(int_features, dtype=float)]

    prediction = heart_predict.predict(processed_feature_heart)
    display_text = "This person has Heart Disease" if prediction[0] == 1 else "This person doesn't have Heart Disease"

    # ----------------------------
    # SHAP computation
    # ----------------------------
    shap_plot = None
    shap_summary = {"positive": [], "negative": []}
    shap_insight_text = None

    try:
        X_input = np.array(int_features, dtype=float).reshape(1, -1)
        background = np.zeros_like(X_input)

        explainer = shap.KernelExplainer(heart_predict.predict, background)
        shap_values = explainer.shap_values(X_input)
        abs_vals = np.abs(shap_values).ravel()

        # ---- Heart Feature List ----
        feature_names = [
            "age", "ane", "creatinine_phosphokinase", "y_diabetes",
            "ejection_fraction", "high_blood_pressure", "platelets",
            "serum_creatinine", "serum_sodium", "gender",
            "smoking", "time"
        ]

        # Ranking
        top_k = min(len(feature_names), len(abs_vals))
        ranking = sorted(zip(feature_names, abs_vals), key=lambda x: x[1], reverse=True)[:top_k]
        shap_results = [{"feature": f, "impact": float(round(v, 6))} for f, v in ranking]

        # Bar plot
        feat_names_plot = [f for f, _ in ranking][::-1]
        impacts = [v for _, v in ranking][::-1]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(feat_names_plot, impacts, color="#ff4d4d")
        ax.set_xlabel("Feature impact (|SHAP|)")
        ax.set_title("Top Influencing Features")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        shap_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)

        # Positive / negative summary
        for f, v in zip(feature_names, abs_vals):
            if v >= 0:
                shap_summary["positive"].append({"feature": f, "impact": float(round(v, 6))})
            else:
                shap_summary["negative"].append({"feature": f, "impact": float(round(abs(v), 6))})

        # Insight text
        if len(shap_summary["positive"]) > 0:
            top_pos = sorted(shap_summary["positive"], key=lambda x: x["impact"], reverse=True)[:3]
            shap_insight_text = "Key contributing factors: " + ", ".join(
                [t["feature"].replace("_", " ").title() for t in top_pos]) + "."

    except Exception as e:
        print("SHAP computation failed:", e)

    # ----------------------------
    # Render template
    # ----------------------------
    return render_template('heartdisease.html',
                           output_text="Result: {}".format(display_text),
                           shap_plot=shap_plot,
                           shap_summary=shap_summary,
                           shap_insight_text=shap_insight_text)



@app.route('/predictkidney/', methods=['POST'])
def predictkidney():

    # ----------------------------
    # Original Prediction
    # ----------------------------
    int_features = [x for x in request.form.values()]   # fetch form values
    processed_features_kidney = [np.array(int_features, dtype=float)]

    prediction = kidney_predict.predict(processed_features_kidney)
    display_text = (
        "This person has Chronic Kidney Disease" 
        if prediction[0] == 1 else 
        "This person does NOT have Chronic Kidney Disease"
    )

    # ----------------------------
    # SHAP computation
    # ----------------------------
    shap_plot = None
    shap_summary = {"positive": [], "negative": []}
    shap_insight_text = None

    try:
        X_input = np.array(int_features, dtype=float).reshape(1, -1)
        background = np.zeros_like(X_input)

        explainer = shap.KernelExplainer(kidney_predict.predict, background)
        shap_values = explainer.shap_values(X_input)
        abs_vals = np.abs(shap_values).ravel()

        # ---- Kidney Feature List ----
        feature_names = [
            "ane", "bp", "sg", "sod", "pot",
            "hemo", "pcv", "pc", "id", "age"
        ]

        # Ranking features
        top_k = min(len(feature_names), len(abs_vals))
        ranking = sorted(
            zip(feature_names, abs_vals), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]

        shap_results = [{"feature": f, "impact": float(round(v, 6))} for f, v in ranking]

        # Bar plot
        feat_names_plot = [f for f, _ in ranking][::-1]
        impacts = [v for _, v in ranking][::-1]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(feat_names_plot, impacts, color="#4d79ff")
        ax.set_xlabel("Feature impact (|SHAP|)")
        ax.set_title("Top Influencing Features (Kidney)")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        shap_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)

        # Categorize positive & negative impact
        for f, v in zip(feature_names, abs_vals):
            if v >= 0:
                shap_summary["positive"].append(
                    {"feature": f, "impact": float(round(v, 6))}
                )
            else:
                shap_summary["negative"].append(
                    {"feature": f, "impact": float(round(abs(v), 6))}
                )

        # Insight text
        if len(shap_summary["positive"]) > 0:
            top_pos = sorted(
                shap_summary["positive"], 
                key=lambda x: x["impact"], 
                reverse=True
            )[:3]

            shap_insight_text = (
                "Most influential kidney health indicators: " + 
                ", ".join([t["feature"].upper() for t in top_pos]) + "."
            )

    except Exception as e:
        print("SHAP computation failed:", e)

    # ----------------------------
    # Render template
    # ----------------------------
    return render_template(
        'kidneydisease.html',
        output_text=f"Result: {display_text}",
        shap_plot=shap_plot,
        shap_summary=shap_summary,
        shap_insight_text=shap_insight_text
    )




@app.route('/predictliver/', methods=['POST'])
def predictliver():

    # ----------------------------
    # Original Prediction
    # ----------------------------
    int_features = [x for x in request.form.values()]   # fetch form values
    processed_features_liver = [np.array(int_features, dtype=float)]

    prediction = liver_predict.predict(processed_features_liver)
    display_text = (
        "This person has Liver Disease"
        if prediction[0] == 1 else
        "This person does NOT have Liver Disease"
    )

    # ----------------------------
    # SHAP computation
    # ----------------------------
    shap_plot = None
    shap_summary = {"positive": [], "negative": []}
    shap_insight_text = None

    try:
        X_input = np.array(int_features, dtype=float).reshape(1, -1)
        background = np.zeros_like(X_input)

        explainer = shap.KernelExplainer(liver_predict.predict, background)
        shap_values = explainer.shap_values(X_input)
        abs_vals = np.abs(shap_values).ravel()

        # ---- Liver Feature List ----
        feature_names = [
            "age",
            "Aspartate_Aminotransferase",
            "Total_Protiens",
            "Albumin",
            "Albumin_and_Globulin_Ratio",
            "y_liver",
            "gender"
        ]

        # Ranking features
        top_k = min(len(feature_names), len(abs_vals))
        ranking = sorted(
            zip(feature_names, abs_vals),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        shap_results = [{"feature": f, "impact": float(round(v, 6))} for f, v in ranking]

        # Bar plot
        feat_names_plot = [f for f, _ in ranking][::-1]
        impacts = [v for _, v in ranking][::-1]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(feat_names_plot, impacts, color="#ff9933")
        ax.set_xlabel("Feature impact (|SHAP|)")
        ax.set_title("Top Influencing Features (Liver Disease)")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        shap_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)

        # Summaries
        for f, v in zip(feature_names, abs_vals):
            if v >= 0:
                shap_summary["positive"].append(
                    {"feature": f, "impact": float(round(v, 6))}
                )
            else:
                shap_summary["negative"].append(
                    {"feature": f, "impact": float(round(abs(v), 6))}
                )

        # Insight text
        if len(shap_summary["positive"]) > 0:
            top_pos = sorted(
                shap_summary["positive"],
                key=lambda x: x["impact"],
                reverse=True
            )[:3]

            shap_insight_text = (
                "Most influential liver health indicators: " +
                ", ".join([t["feature"].replace('_',' ').title() for t in top_pos]) +
                "."
            )

    except Exception as e:
        print("SHAP computation failed:", e)

    # ----------------------------
    # Render template
    # ----------------------------
    return render_template(
        'liverdisease.html',
        output_text=f"Result: {display_text}",
        shap_plot=shap_plot,
        shap_summary=shap_summary,
        shap_insight_text=shap_insight_text
    )



@app.route('/predictdiabetes/', methods=['POST'])
def predictdiabetes():      
    # ----------------------------
    # Original prediction
    # ----------------------------
    int_features = [x for x in request.form.values()]
    processed_feature_diabetes = [np.array(int_features, dtype=float)]
    
    prediction = diabetes_predict.predict(processed_feature_diabetes)
    display_text = "This person has Diabetes" if prediction[0] == 1 else "This person doesn't have Diabetes"

    # ----------------------------
    # SHAP computation
    # ----------------------------
    shap_plot = None
    shap_summary = {"positive": [], "negative": []}
    shap_insight_text = None

    try:
        X_input = np.array(int_features, dtype=float).reshape(1, -1)
        background = np.zeros_like(X_input)
        explainer = shap.KernelExplainer(diabetes_predict.predict, background)
        shap_values = explainer.shap_values(X_input)
        abs_vals = np.abs(shap_values).ravel()

        feature_names = ["Pregnancies", "Glucose", "bp", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "age"]

        # Build ranking˛
        top_k = min(8, len(feature_names))
        ranking = sorted(zip(feature_names, abs_vals), key=lambda x: x[1], reverse=True)[:top_k]
        shap_results = [{"feature": f, "impact": float(round(v, 6))} for f, v in ranking]
        # Bar plot
        feat_names_plot = [f for f, _ in ranking][::-1]
        impacts = [v for _, v in ranking][::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(feat_names_plot, impacts, color="#007bff")
        ax.set_xlabel("Feature impact (|SHAP|)")
        ax.set_title("Top Influencing Features")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        shap_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)

        # Positive / negative summary
        for f, v in zip(feature_names, abs_vals):
            if v >= 0:
                shap_summary["positive"].append({"feature": f, "impact": float(round(v, 6))})
            else:
                shap_summary["negative"].append({"feature": f, "impact": float(round(abs(v), 6))})

        # Insight text: top 3 positive contributors
        if len(shap_summary["positive"]) > 0:
            top_pos = sorted(shap_summary["positive"], key=lambda x: x["impact"], reverse=True)[:3]
            shap_insight_text = "Highest contributing factors: " + ", ".join(
                [t["feature"].replace("_", " ").title() for t in top_pos]) + "."

    except Exception as e:
        print("SHAP computation failed:", e)

    # ----------------------------
    # Render template
    # ----------------------------
    return render_template('diabetes.html',
                           output_text="Result: {}".format(display_text),
                           shap_plot=shap_plot,
                           shap_summary=shap_summary,
                           shap_insight_text=shap_insight_text)


@app.route("/predict", methods=["POST"])
def predict():
    # Collect inputs
    input_vals_raw = []
    provided = {}
    for feat in features:
        raw = request.form.get(feat)
        provided[feat] = raw if raw is not None else ""
        if feat in ["gender", "smoking", "high_blood_pressure"]:
            val = feature_means.get(feat, 0.0) if raw in [None, ""] else (1.0 if raw == "1" else 0.0)
        else:
            try:
                val = float(raw) if raw.strip() != "" else feature_means.get(feat, 0.0)
            except Exception:
                flash(f"Invalid value for {feat}. Using mean.", "warning")
                val = feature_means.get(feat, 0.0)
        input_vals_raw.append(val)

    X_new_raw = np.array(input_vals_raw, dtype=float).reshape(1, -1)

    try:
        X_scaled = scaler.transform(X_new_raw)
        probs = model.predict(X_scaled, verbose=0)[0]
    except Exception as e:
        flash(f"Error during prediction: {e}", "danger")
        return render_template("temporary.html",
                               features=features, defaults=provided,
                               prediction=None, shap_results=None,
                               shap_plot=None, shap_summary=None,
                               shap_insight_text=None)

    # Prediction results
    result = [{"target": t.replace("y_", "").replace("_", " ").title(), "prob": float(p)}
              for t, p in zip(targets, probs)]

    # ===== SHAP or fallback processing =====
    try:
        if explainer:
            ev = explainer(X_scaled)
            vals = np.array(ev.values)
            if vals.ndim == 3:
                abs_vals = np.mean(np.abs(vals[:, 0, :]), axis=0)
            elif vals.ndim == 2:
                abs_vals = np.abs(vals[0])
            else:
                abs_vals = np.abs(vals).ravel()
        else:
            raise RuntimeError("SHAP explainer not initialized")
    except Exception:
        abs_vals = permutation_importance_for_instance(X_scaled)

    top_k = min(8, len(features))
    ranking = sorted(zip(features, abs_vals), key=lambda x: x[1], reverse=True)[:top_k]
    shap_results = [{"feature": f, "impact": float(round(v, 6))} for f, v in ranking]

    # Plot
    feat_names = [f for f, _ in ranking][::-1]
    impacts = [v for _, v in ranking][::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(feat_names, impacts, color="#007bff")
    ax.set_xlabel("Feature impact (|SHAP|)")
    ax.set_title("Top Influencing Features")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    shap_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    shap_summary = {"positive": [], "negative": []}
    for f, v in zip(features, abs_vals):
        if v >= 0:
            shap_summary["positive"].append({"feature": f, "impact": float(round(v, 6))})
        else:
            shap_summary["negative"].append({"feature": f, "impact": float(abs(round(v, 6)))})

    if len(shap_summary["positive"]) > 0:
        top_pos = sorted(shap_summary["positive"], key=lambda x: x["impact"], reverse=True)[:3]
        shap_insight_text = "Highest contributing factors: " + ", ".join(
            [t["feature"].replace("_", " ").title() for t in top_pos]) + "."
    else:
        shap_insight_text = None

    return render_template("temporary.html",
                           features=features,
                           defaults=provided,
                           prediction=result,
                           shap_results=shap_results,
                           shap_plot=shap_plot,
                           shap_summary=shap_summary,
                           shap_insight_text=shap_insight_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
