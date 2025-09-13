import pandas as pd
import numpy as np

# ==============================
# Call LLM (rule-based fallback)
# ==============================
def call_llm(prompt):
    """
    Fallback code generator: parses prompt to generate a Signal column
    using SMA crossover logic.
    """
    import re

    # Extract SMA periods from prompt
    nums = [int(n.replace(',', '')) for n in re.findall(r'\b\d{1,10}\b', prompt)]
    short_sma, long_sma = (nums[0], nums[1]) if len(nums) >= 2 else (20, 50)

    # Generate proper pandas code
    code = f"""
# Compute SMA columns
df['SMA{short_sma}'] = df['Close'].rolling({short_sma}).mean()
df['SMA{long_sma}'] = df['Close'].rolling({long_sma}).mean()

# Generate Signal: 1 = Buy, -1 = Sell, 0 = Hold
df['Signal'] = 0
df.loc[df['SMA{short_sma}'] > df['SMA{long_sma}'], 'Signal'] = 1
df.loc[df['SMA{short_sma}'] < df['SMA{long_sma}'], 'Signal'] = -1
"""
    return code.strip()


# ==============================
# Generate Strategy Code
# ==============================
def generate_strategy_code(prompt_text):
    """
    Takes a prompt describing an SMA crossover strategy and returns Python code
    that creates a 'Signal' column in df.
    """
    prompt = (
        "You are a Python financial analyst.\n"
        "Write pure Python Pandas code using the already-loaded DataFrame `df`.\n"
        "Don't include imports, CSV loading, or markdown code blocks.\n"
        f"\nStrategy Description:\n{prompt_text}"
    )
    return call_llm(prompt)

# ==============================
# Example Usage
# ==============================
if __name__ == "__main__":
    # Sample DataFrame
    dates = pd.date_range("2025-01-01", periods=100)
    df = pd.DataFrame({
        "Close": np.random.randn(100).cumsum() + 100
    }, index=dates)

    # Example strategies
    strategies = [
        "Use a momentum strategy with 20/50 SMA crossover.",
        "Use a strategy with 60/100 SMA crossover.",
        "Use a strategy with 100/1000 SMA crossover.",
        "Use a strategy with 1000/1000 SMA crossover."
    ]

    for s in strategies:
        code = generate_strategy_code(s)
        print(f"\n--- Strategy: {s} ---")
        print(code)

        # Execute generated code
        exec(code)
        print(df[['Close', 'Signal']].tail(5))
