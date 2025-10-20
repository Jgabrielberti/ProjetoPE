import numpy as np
from splitter.train_val_split import train_val_split

def predict_variance(df_filtrado, train_end=2020):
    train_df, val_df, future_years = train_val_split(df_filtrado, train_end=train_end)

    train_sorted = train_df.sort_values("ano")

    variance = train_sorted["desmatado"].var()
    std_dev = np.sqrt(variance)

    first_value = train_sorted.iloc[0]["desmatado"]
    last_value = train_sorted.iloc[-1]["desmatado"]
    trend_direction = 1 if last_value > first_value else -1

    future_preds = []
    current_value = last_value

    for year in future_years:
        current_value = current_value + (trend_direction * std_dev)
        future_preds.append(float(current_value))

    val_preds = {}
    current_val = last_value

    for year in sorted(val_df["ano"]):
        current_val = current_val + (trend_direction * std_dev)
        val_preds[year] = float(current_val)

    trend_desc = "crescente" if trend_direction > 0 else "decrescente"

    return {
        "method": "variancia",
        "pred_years": future_years,
        "predictions": future_preds,
        "validation_predictions": val_preds,
        "last_known_value": float(last_value),
        "variance": float(variance),
        "std_dev_step": float(std_dev),
        "trend_direction": trend_desc,
        "note": f"Método ingênuo usando variabilidade histórica (tendência {trend_desc}). Demonstra limitações."
    }