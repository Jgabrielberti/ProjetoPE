import os
import pandas as pd
from pre_processing.pre_processing import preprocess_and_save
from splitter.train_val_split import get_municipio_df
from model import (
    predict_variance,
    predict_moving_average,
    predict_linear_regression,
    predict_random_forest,
    predict_arima,
)
from evaluation.model_evaluator import ModelEvaluator

def main():
    raw_path = r"C:\Users\Usuario\PycharmProjects\ProjetoPE\data_raw\br_inpe_prodes_municipio_bioma.csv"
    processed_path = r"C:\Users\Usuario\PycharmProjects\ProjetoPE\data_processed\br_inpe_prodes_municipio_bioma_processed.csv"

    if not os.path.exists(processed_path):
        df, _ = preprocess_and_save(raw_path, processed_path)
    else:
        df = pd.read_csv(processed_path, encoding="utf-8", low_memory=False)

    print("\nResumo do conjunto de dados:")
    print(f"Total de linhas: {len(df)}")
    print(f"Municípios únicos: {df['id_municipio'].nunique()}")
    print(f"Intervalo de anos: {df['ano'].min()} - {df['ano'].max()}\n")

    municipio_id = int(input("Digite id_municipio (ex: 2604106): ").strip())
    bioma = input("Digite o bioma ao qual a cidade pertence (Ex: Mata Atlântica): ")

    df_filtrado = get_municipio_df(municipio_id, bioma, df)
    print(f"\nEncontradas {len(df_filtrado)} linhas para o município {municipio_id} "
          f"(anos {df_filtrado['ano'].min()}..{df_filtrado['ano'].max()})")

    result_variance = predict_variance(df_filtrado)
    result_ma = predict_moving_average(df_filtrado, window=3)
    result_lr = predict_linear_regression(df_filtrado)
    result_rf = predict_random_forest(df_filtrado)
    result_arima = predict_arima(df_filtrado)

    results = [result_variance, result_ma, result_lr, result_rf, result_arima]

    print("Previsões para anos futuros (2024-2028): ")

    for r in results:
        print(f"\nMétodo: {r['method']}")
        print(f"Anos: {r['pred_years']}")
        print(f"Previsões: {[round(p, 3) for p in r['predictions']]}")
        if "val_rmse" in r:
            print(f"Validação RMSE: {r.get('val_rmse', 'N/A'):.4f}")
            print(f"Validação MAE: {r.get('val_mae', 'N/A'):.4f}")
        if "note" in r:
            print(f"{r['note']}")

    evaluator = ModelEvaluator(df_filtrado)

    models_with_validation = [
        ("Variância", result_variance.get("validation_predictions", {})),
        ("Moving Average", result_ma.get("validation_predictions", {})),
        ("Regressão Linear", result_lr.get("validation_predictions", {})),
        ("Random Forest", result_rf.get("validation_predictions", {})),
        ("ARIMA", result_arima.get("validation_predictions", {}))
    ]

    for model_name, val_preds in models_with_validation:
        evaluator.add_model_prediction(model_name, val_preds)

    comparison_df, metrics_df = evaluator.generate_comparison_table()
    evaluator.plot_comparison(save_path="model_comparison.png")
    evaluator.generate_insights(metrics_df)

    try:
        comparison_df.to_csv("resultados_comparacao.csv", index=False)
        metrics_df.to_csv("metricas_modelos.csv", index=False)
        print(f"\nResultados salvos em 'resultados_comparacao.csv' e 'metricas_modelos.csv'")
    except Exception as e:
        print(f"\nErro ao salvar arquivos: {e}")


if __name__ == "__main__":
    main()