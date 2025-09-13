def make_decision(predicted_return, threshold_buy=0.002, threshold_sell=-0.002):
    """
    Simple rule: 
      - Buy if predicted return > threshold_buy
      - Sell if predicted return < threshold_sell
      - Hold otherwise
    """
    if predicted_return > threshold_buy:
        return "Buy"
    elif predicted_return < threshold_sell:
        return "Sell"
    else:
        return "Hold"
