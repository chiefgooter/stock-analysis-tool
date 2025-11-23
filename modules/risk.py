import numpy as np

def calculate_risk_metrics(df):
    returns = df['Close'].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    downside = returns.copy()
    downside[downside > 0] = 0
    sortino = returns.mean() / downside.std() * np.sqrt(252) if downside.std() != 0 else 0
    max_dd = ((df['Close'] / df['Close'].cummax()) - 1).min() * 100
    var_95 = returns.quantile(0.05)
    return {
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd": round(max_dd, 2),
        "var_95": var_95
    }
