import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, render_template, request, flash
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-for-local-use")

LOOKBACK = 168
HORIZON = 24
# DATA_PATH = Path(os.environ.get("DATA_PATH", "PJME_hourly.csv"))
# MODEL_PATH = Path(os.environ.get("GRU_MODEL_PATH", "gru_checkpoints/best_model.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR.parent / "PJME_hourly.csv"

MODEL_PATH = (
    BASE_DIR.parent
    / "gru_checkpoints"
    / "best_model.pth"
)

# This architecture must match the one used to train your GRU checkpoint.
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 output_size=24, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def load_history():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Historical CSV not found at {DATA_PATH}. Put PJME_hourly.csv beside app.py "
            "or set the DATA_PATH environment variable."
        )

    df = pd.read_csv(DATA_PATH)
    columns = {str(c).strip().lower(): c for c in df.columns}
    dt_col = columns.get("datetime") or columns.get("date") or columns.get("timestamp")
    demand_col = columns.get("pjme_mw") or columns.get("demand") or columns.get("load") or columns.get("mw")
    if dt_col is None or demand_col is None:
        raise ValueError("CSV needs datetime and demand columns, such as Datetime and PJME_MW.")

    df = df[[dt_col, demand_col]].copy()
    df.columns = ["Datetime", "Demand_MW"]
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["Demand_MW"] = pd.to_numeric(df["Demand_MW"], errors="coerce")
    df = df.dropna().sort_values("Datetime").drop_duplicates("Datetime").reset_index(drop=True)
    if len(df) < LOOKBACK:
        raise ValueError(f"Need at least {LOOKBACK} hourly observations.")
    return df


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GRU checkpoint not found at {MODEL_PATH}. Check GRU_MODEL_PATH."
        )
    model = GRUModel().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def make_forecast(history, model):
    vals = history["Demand_MW"].to_numpy(dtype=np.float32)
    # Must mirror training preprocessing. If your GRU used a saved scaler,
    # replace this fit with loading that exact scaler.
    train_end = max(1, int(len(vals) * 0.70))
    scaler = StandardScaler().fit(vals[:train_end].reshape(-1, 1))
    scaled = scaler.transform(vals[-LOOKBACK:].reshape(-1, 1)).astype(np.float32)
    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(x).cpu().numpy().reshape(-1, 1)
    pred = scaler.inverse_transform(pred_scaled).flatten()
    last_time = pd.Timestamp(history["Datetime"].iloc[-1])
    times = pd.date_range(last_time + pd.Timedelta(hours=1), periods=HORIZON, freq="h")
    return pd.DataFrame({"Datetime": times, "Forecast_MW": pred})


@app.route("/", methods=["GET", "POST"])
def dashboard():
    error = None
    history = None
    forecast = None
    threshold = 40000.0
    supply = 40000
    shares = {"Residential": 50, "Industrial": 30, "Commercial": 20}
    try:
        history = load_history()
        model = load_model()
        forecast = make_forecast(history, model)
        threshold = float(request.form.get("threshold", history["Demand_MW"].quantile(0.90)))
        supply = int(request.form.get("supply", max(0, round(float(forecast["Forecast_MW"].max())))))
        shares = {
            "Residential": int(request.form.get("residential", 50)),
            "Industrial": int(request.form.get("industrial", 30)),
            "Commercial": int(request.form.get("commercial", 20)),
        }
    except Exception as exc:
        error = str(exc)

    if history is not None and forecast is not None:
        latest = history.iloc[-1]
        peak = history.loc[history["Demand_MW"].idxmax()]
        forecast_peak = forecast.loc[forecast["Forecast_MW"].idxmax()]
        peak_rows = forecast[forecast["Forecast_MW"] >= threshold]
        recent = history.tail(168)
        chart_labels = [x.isoformat() for x in recent["Datetime"]]
        chart_actual = [round(float(x), 2) for x in recent["Demand_MW"]]
        forecast_labels = [x.isoformat() for x in forecast["Datetime"]]
        forecast_values = [round(float(x), 2) for x in forecast["Forecast_MW"]]
        forecast_table = [
            {"time": t.strftime("%d %b %Y, %H:%M"), "demand": f"{v:,.0f}"}
            for t, v in zip(forecast["Datetime"], forecast["Forecast_MW"])
        ]
        allocation_total = sum(shares.values())
        allocation = []
        if allocation_total > 0:
            for sector, pct in shares.items():
                allocation.append({
                    "sector": sector,
                    "pct": pct,
                    "mw": supply * pct / allocation_total,
                })
        peak_table = [
            {"time": t.strftime("%d %b %Y, %H:%M"), "demand": f"{v:,.0f}"}
            for t, v in zip(peak_rows["Datetime"], peak_rows["Forecast_MW"])
        ]
        return render_template(
            "index.html",
            error=error,
            device=DEVICE.type.upper(),
            latest=f"{latest['Demand_MW']:,.0f}",
            latest_time=latest["Datetime"].strftime("%d %b %Y, %H:%M"),
            historical_peak=f"{peak['Demand_MW']:,.0f}",
            avg=f"{history['Demand_MW'].mean():,.0f}",
            forecast_peak=f"{forecast_peak['Forecast_MW']:,.0f}",
            forecast_peak_time=forecast_peak["Datetime"].strftime("%d %b, %H:%M"),
            chart_labels=chart_labels,
            chart_actual=chart_actual,
            forecast_labels=forecast_labels,
            forecast_values=forecast_values,
            forecast_table=forecast_table,
            threshold=threshold,
            peak_table=peak_table,
            supply=supply,
            shares=shares,
            allocation=allocation,
            allocation_total=allocation_total,
        )

    return render_template(
        "index.html", error=error, device=DEVICE.type.upper(),
        latest=None, historical_peak=None, avg=None, forecast_peak=None,
        forecast_peak_time=None, chart_labels=[], chart_actual=[],
        forecast_labels=[], forecast_values=[], forecast_table=[],
        threshold=threshold, peak_table=[], supply=supply, shares=shares,
        allocation=[], allocation_total=sum(shares.values()),
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
