from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# ============================================================
#  FILE PATHS
# ============================================================
MERGED_CROP_FILE = "merged_crop_price_enhanced_full.csv"
STATES_FILE = "states_maha.csv"
DISTRICTS_FILE = "districts_maha.csv"
DISTRICT_DEFAULTS_FILE = "district_defaults_maha.csv"
CROP_MASTER_FILE = "crop_master_maha.csv"

# ============================================================
#  LOAD DATASETS
# ============================================================
if os.path.exists(MERGED_CROP_FILE):
    final_df = pd.read_csv(MERGED_CROP_FILE)
elif os.path.exists("merged_crop_price.csv"):
    final_df = pd.read_csv("merged_crop_price.csv")
else:
    raise FileNotFoundError("Merged crop dataset NOT FOUND.")

states_df = pd.read_csv(STATES_FILE)
districts_df = pd.read_csv(DISTRICTS_FILE)
district_defaults_df = pd.read_csv(DISTRICT_DEFAULTS_FILE)

crop_master_df = pd.read_csv(CROP_MASTER_FILE) if os.path.exists(CROP_MASTER_FILE) else None

final_df.columns = [c.strip() for c in final_df.columns]
district_defaults_df.columns = [c.strip() for c in district_defaults_df.columns]

# Missing columns →
if "yield_per_hectare" not in final_df.columns:
    final_df["yield_per_hectare"] = 1200

if "avg_price" not in final_df.columns:
    if crop_master_df is not None:
        price_map = dict(zip(crop_master_df["crop"], crop_master_df["avg_price"]))
        final_df["avg_price"] = final_df["label"].map(price_map).fillna(30)
    else:
        final_df["avg_price"] = 30

# ============================================================
#  TRAIN MODEL
# ============================================================
feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'yield_per_hectare']

train_df = final_df.dropna(subset=feature_columns + ['label'])
X = train_df[feature_columns]
y = train_df['label']

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X, y)

# ============================================================
# SAFE FLOAT
# ============================================================
def safe_float(v, default):
    try:
        if v is None or str(v).strip() == "":
            return float(default)
        return float(v)
    except:
        return float(default)

# ============================================================
#  LANDING PAGE
# ============================================================
@app.route("/")
def landing():
    return render_template("landing.html")


# ============================================================
#  DASHBOARD
# ============================================================
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():

    # ---------------- STATES ----------------
    states = list(states_df["state_name"].unique())
    selected_state = request.form.get("state") or states[0]

    # ---------------- DISTRICTS ----------------
    if "state_id" in states_df.columns:
        state_id = states_df[states_df["state_name"] == selected_state]["state_id"].values[0]
        districts = list(districts_df[districts_df["state_id"] == state_id]["district_name"].unique())
    else:
        districts = list(districts_df[districts_df["state"] == selected_state]["district_name"].unique())

    selected_district = request.form.get("district") or districts[0]

    # ---------------- SEASON ----------------
    seasons = sorted(district_defaults_df["season"].unique())
    selected_season = request.form.get("season") or seasons[0]

    # ============================================================
    # IF PREDICT → DO NOT LOAD DEFAULTS
    # KEEP USER ENTERED VALUES
    # ============================================================
    if request.method == "POST" and request.form.get("action") == "predict":

        defaults = {
            "N": request.form.get("N"),
            "P": request.form.get("P"),
            "K": request.form.get("K"),
            "temperature": request.form.get("temperature"),
            "humidity": request.form.get("humidity"),
            "ph": request.form.get("ph"),
            "rainfall": request.form.get("rainfall"),
        }

    else:
        # ============================================================
        # WHEN STATE / DISTRICT / SEASON CHANGES → LOAD DEFAULTS
        # ============================================================
        try:
            defaults = district_defaults_df[
                (district_defaults_df["state"] == selected_state) &
                (district_defaults_df["district"] == selected_district) &
                (district_defaults_df["season"] == selected_season)
            ].iloc[0].to_dict()
        except:
            defaults = {"N": 80, "P": 40, "K": 40,
                        "temperature": 28, "humidity": 60,
                        "ph": 6.7, "rainfall": 500}

    recommended = None
    profit_list = None
    best_crop = None
    plot_data = None

    # ============================================================
    # PREDICT (BUTTON CLICKED)
    # ============================================================
    if request.method == "POST" and request.form.get("action") == "predict":

        N = safe_float(defaults["N"], defaults["N"])
        P = safe_float(defaults["P"], defaults["P"])
        K = safe_float(defaults["K"], defaults["K"])
        temperature = safe_float(defaults["temperature"], defaults["temperature"])
        humidity = safe_float(defaults["humidity"], defaults["humidity"])
        ph = safe_float(defaults["ph"], defaults["ph"])
        rainfall = safe_float(defaults["rainfall"], defaults["rainfall"])
        farm_size = safe_float(request.form.get("farm_size"), 1.0)

        input_df = pd.DataFrame([{
            "N": N, "P": P, "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
            "yield_per_hectare": final_df["yield_per_hectare"].mean()
        }])

        # Predict
        probs = model.predict_proba(input_df)[0]
        classes = model.classes_

        crop_prob_map = {c: probs[i] for i, c in enumerate(classes)}

        sorted_top = sorted(crop_prob_map.items(), key=lambda x: x[1], reverse=True)
        recommended = [c[0] for c in sorted_top[:3]]

        # Profit Calculation
        profit_list = []
        for crop in recommended:
            price = final_df[final_df["label"] == crop]["avg_price"].mean()
            yield_val = final_df[final_df["label"] == crop]["yield_per_hectare"].mean()
            expected_profit = yield_val * price * farm_size
            profit_list.append((crop, round(expected_profit, 2)))

        profit_list.sort(key=lambda x: x[1], reverse=True)
        best_crop = profit_list[0][0]

        # Chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([p[0] for p in profit_list], [p[1] for p in profit_list])
        ax.set_ylabel("Expected Profit (₹)")
        ax.set_title(f"Profit Comparison — {selected_district}")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode()
        plt.close(fig)

    return render_template(
        "index.html",
        states=states,
        districts=districts,
        seasons=seasons,
        selected_state=selected_state,
        selected_district=selected_district,
        selected_season=selected_season,
        defaults=defaults,
        recommended=recommended,
        profit_list=profit_list,
        best_crop=best_crop,
        plot_data=plot_data
    )


if __name__ == "__main__":
    app.run(debug=True)




