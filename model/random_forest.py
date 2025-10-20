import numpy as np
from sklearn.ensemble import RandomForestRegressor
from splitter.train_val_split import train_val_split

def predict_random_forest(df_filtrado, train_end=2020, n_estimators=200, random_state=42):
    train_df, val_df, future_years = train_val_split(df_filtrado, train_end=train_end)

    full = df_filtrado.sort_values("ano").copy()

    full["lag1"] = full["desmatado"].shift(1)
    full["lag2"] = full["desmatado"].shift(2)
    full["lag3"] = full["desmatado"].shift(3)
    full["rolling_mean_3"] = full["desmatado"].rolling(3, min_periods=1).mean()
    full["year_normalized"] = full["ano"] - full["ano"].min()

    train = full[full["ano"] <= train_end].dropna(subset=["lag1", "lag2", "lag3"]).copy()

    if train.shape[0] < 4:
        mean_val = train["desmatado"].mean() if not train.empty else float(df_filtrado["desmatado"].mean())
        val_preds = {year: mean_val for year in val_df["ano"]}
        return {
            "method": "random_forest",
            "pred_years": future_years,
            "predictions": [mean_val] * len(future_years),
            "validation_predictions": val_preds
        }

    feature_cols = ["lag1", "lag2", "lag3", "rolling_mean_3", "year_normalized"]
    x_train = train[feature_cols].values
    y_train = train["desmatado"].values

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(x_train, y_train)

    val_preds = {}
    last_known = train[feature_cols].iloc[-1].values.copy()
    val_years = sorted(val_df["ano"])

    for year in val_years:
        features = last_known.copy()
        features[4] = year - full["ano"].min()

        x_in = features.reshape(1, -1)
        yhat = model.predict(x_in)[0]
        val_preds[year] = float(yhat)

        last_known[0] = yhat
        last_known[1] = last_known[0]
        last_known[2] = last_known[1]
        last_known[3] = np.mean([last_known[0], last_known[1], last_known[2]])

    future_preds = []
    for year in future_years:
        features = last_known.copy()
        features[4] = year - full["ano"].min()

        x_in = features.reshape(1, -1)
        yhat = model.predict(x_in)[0]
        future_preds.append(float(yhat))

        last_known[0] = yhat
        last_known[1] = last_known[0]
        last_known[2] = last_known[1]
        last_known[3] = np.mean([last_known[0], last_known[1], last_known[2]])

    return {
        "method": "random_forest",
        "pred_years": future_years,
        "predictions": future_preds,
        "validation_predictions": val_preds,
        "feature_importance": dict(zip(feature_cols, model.feature_importances_))
    }