import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from splitter.train_val_split import train_val_split

def predict_linear_regression(df_filtrado, train_end=2020):
    train_df, val_df, future_years = train_val_split(df_filtrado, train_end=train_end)

    min_year = train_df["ano"].min()
    x_train = (train_df[["ano"]].values - min_year).reshape(-1, 1)
    y_train = train_df["desmatado"].values

    x_val = (val_df[["ano"]].values - min_year).reshape(-1, 1)
    y_val = val_df["desmatado"].values

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_val_pred = model.predict(x_val)
    val_preds = dict(zip(val_df["ano"].values, y_val_pred))

    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)

    x_future = (np.array(future_years).reshape(-1, 1) - min_year)
    y_future = model.predict(x_future)

    return {
        "method": "regressao_linear",
        "pred_years": future_years,
        "predictions": y_future.tolist(),
        "validation_predictions": val_preds,
        "val_rmse": float(val_rmse),
        "val_mae": float(val_mae),
        "coef": float(model.coef_[0]),
        "intercept": float(model.intercept_)
    }