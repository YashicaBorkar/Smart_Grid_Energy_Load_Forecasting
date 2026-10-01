SmartGrid AI Flask Dashboard

Files:
- app.py
- templates/index.html
- static/style.css
- requirements.txt

Expected project files:
- PJME_hourly.csv
- gru_checkpoints/best_model.pth

Run:
1) pip install -r requirements.txt
2) python app.py
3) Open http://127.0.0.1:5000

Important:
The GRU class in app.py assumes a 2-layer GRU, hidden_size=64, input_size=1,
output_size=24, dropout=0.2, and a linear head from 64 to 24. This must match
the training architecture exactly. Preprocessing assumes a StandardScaler fit
on the first 70% of the historical series. If the training notebook saved and
used a separate scaler, update make_forecast() to load that exact scaler.
