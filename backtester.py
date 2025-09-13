def run_backtest(code_str, df):
    exec(code_str, globals(), locals())
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Shift signal to avoid lookahead bias
    df['Strategy_Return'] = df['Daily_Return'] * df['Signal'].shift(1)
    
    df['Cumulative_Market'] = (1 + df['Daily_Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    return df
