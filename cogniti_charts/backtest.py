import numpy as np, pandas as pd
from .utils import CLASS_NAMES

def backtest_breakout(prices: pd.DataFrame, preds: pd.DataFrame, max_hold_days=5, slippage_bps=2, commission_bps=1):
    df = prices.copy().reset_index(drop=True)
    preds = preds.copy().reset_index(drop=True)
    df = df.iloc[len(df)-len(preds):].reset_index(drop=True)  # align to prediction dates

    signals = (preds['pred_label']=="Breakout").astype(int)
    pnl = []
    holding = 0
    entry_px = 0.0
    hold_days = 0

    for i in range(len(df)-1):
        if holding==0 and signals.iloc[i]==1:
            entry_px = df['close'].iloc[i] * (1 + (slippage_bps+commission_bps)/10000.0)
            holding = 1; hold_days = 0
        elif holding==1:
            hold_days += 1
            exit_cond = (signals.iloc[i]==0) or (hold_days >= max_hold_days)
            if exit_cond:
                exit_px = df['close'].iloc[i+1] * (1 - (slippage_bps+commission_bps)/10000.0)
                pnl.append((exit_px - entry_px)/entry_px)
                holding = 0
    pnl = pd.Series(pnl if pnl else [0.0])
    equity = (1+pnl).cumprod()
    ret = pnl.mean()
    std = pnl.std() + 1e-9
    sharpe = (ret/std) * np.sqrt(252/ max(1, len(pnl))) if std>0 else 0.0
    out = {
        "trades": int(len(pnl)),
        "avg_trade_return": float(ret),
        "win_rate": float((pnl>0).mean()),
        "cumulative_return": float(equity.iloc[-1]-1),
        "sharpe": float(sharpe),
    }
    return out, pnl, equity
