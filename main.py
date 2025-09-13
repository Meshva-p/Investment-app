import pandas as pd
import yfinance as yf
import datetime
from llm.strategy_generator import generate_strategy_code
from predictor import train_lstm_model, load_lstm_model, predict_next_lstm_return
from decision_maker import make_decision
from backtester import run_backtest

# =======================
# Load NIFTY50 data (2018–2023)
# =======================
end = datetime.datetime.today().strftime("%Y-%m-%d")
start = "2018-01-01"
nifty = yf.download("^NSEI", start=start, end=end)
df = nifty[['Close']].copy()

# =======================
# Generate strategy from LLM
# =======================
strategy_prompt = "Use a momentum strategy with 20/50 SMA crossover and generate signal column in the DataFrame."
strategy_code = generate_strategy_code(strategy_prompt)
print("🧠 Generated Strategy Code:\n", strategy_code)

# =======================
# Run strategy backtest
# =======================
result_df = run_backtest(strategy_code, df)
if result_df is not None:
    print("✅ Strategy executed. Sample output:")
    print(result_df.tail())

# =======================
# Train or Load LSTM Model
# =======================
model = train_lstm_model(df)
model, scaler = load_lstm_model()

# =======================
# Predict next-day return
# =======================
predicted = predict_next_lstm_return(model, scaler, df)
print(f"\n📈 Predicted next-day return: {predicted:.2%}")

# =======================
# Make decision
# =======================
action = make_decision(predicted)
print(f"\n🟢 ACTION: {action}")
