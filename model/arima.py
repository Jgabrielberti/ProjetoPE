from statsmodels.tsa.arima.model import ARIMA
from splitter.train_val_split import train_val_split

def predict_arima(df_filtrado, train_end=2020):
    train_df, val_df, future_years = train_val_split(df_filtrado, train_end=train_end)

    train_series = train_df.sort_values("ano")["desmatado"].values

    model = ARIMA(train_series, order=(1, 1, 1))
    model_fit = model.fit()

    val_years_count = len(val_df)
    val_predictions = model_fit.forecast(steps=val_years_count)
    val_pred_dict = dict(zip(val_df["ano"].values, val_predictions))

    future_predictions = model_fit.forecast(steps=len(future_years))

    return {
        "method": "ARIMA",
        "pred_years": future_years,
        "predictions": future_predictions.tolist(),
        "validation_predictions": val_pred_dict
    }