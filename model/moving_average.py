import numpy as np
from splitter.train_val_split import train_val_split

def predict_moving_average(df_filtrado, window: int = 3, train_end=2020):
    train_df, val_df, future_years = train_val_split(df_filtrado, train_end=train_end)

    train_series = train_df.sort_values("ano")["desmatado"].values

    future_preds = []
    current_window = train_series[-window:].copy()

    for year in future_years:
        next_pred = float(np.mean(current_window))
        future_preds.append(next_pred)
        current_window = np.append(current_window[1:], next_pred)

    val_preds = {}
    val_window = train_series[-window:].copy()

    for year in sorted(val_df["ano"]):
        next_val_pred = float(np.mean(val_window))
        val_preds[year] = next_val_pred
        val_window = np.append(val_window[1:], next_val_pred)

    return {
        "method": f"Janela de média({window})",
        "pred_years": future_years,
        "predictions": future_preds,
        "validation_predictions": val_preds
    }