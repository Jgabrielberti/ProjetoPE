import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, df_filtrado, train_end=2020, val_start=2021, val_end=2023):
        self.df_filtrado = df_filtrado.sort_values("ano")
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end
        self.true_values = None
        self.predictions = {}

    def get_validation_data(self):
        val_mask = (self.df_filtrado["ano"] >= self.val_start) & (self.df_filtrado["ano"] <= self.val_end)
        self.true_values = self.df_filtrado[val_mask][["ano", "desmatado"]].set_index("ano")["desmatado"]
        return self.true_values

    def add_model_prediction(self, model_name, val_predictions):
        if isinstance(val_predictions, dict):
            self.predictions[model_name] = val_predictions
        else:
            val_years = list(range(self.val_start, self.val_end + 1))
            self.predictions[model_name] = dict(zip(val_years, val_predictions))

    def calculate_metrics(self, true_vals, pred_vals):
        if len(true_vals) == 0 or len(pred_vals) == 0:
            return {}

        mask = ~np.isnan(pred_vals)
        if np.sum(mask) == 0:
            return {}

        true_filtered = true_vals[mask]
        pred_filtered = pred_vals[mask]

        if len(true_filtered) == 0:
            return {}

        return {
            "RMSE": np.sqrt(mean_squared_error(true_filtered, pred_filtered)),
            "MAE": mean_absolute_error(true_filtered, pred_filtered),
            "MAPE": np.mean(np.abs((true_filtered - pred_filtered) / true_filtered)) * 100,
            "R²": r2_score(true_filtered, pred_filtered),
            "Bias": np.mean(pred_filtered - true_filtered)
        }

    def generate_comparison_table(self):
        if self.true_values is None:
            self.get_validation_data()

        results = []
        val_years = list(self.true_values.index)
        true_array = self.true_values.values

        print(f"\nPeríodo de Validação: {self.val_start}-{self.val_end}")

        comparison_data = {"Ano": val_years, "Valor Real": true_array}

        for model_name, pred_dict in self.predictions.items():
            pred_array = [pred_dict.get(year, np.nan) for year in val_years]
            comparison_data[model_name] = pred_array

            valid_mask = ~np.isnan(pred_array)
            if np.sum(valid_mask) > 0:
                true_valid = true_array[valid_mask]
                pred_valid = np.array(pred_array)[valid_mask]
                metrics = self.calculate_metrics(true_valid, pred_valid)
            else:
                metrics = {}

            results.append({
                "Modelo": model_name,
                **metrics
            })

        df_comparison = pd.DataFrame(comparison_data)

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print("\nValores Reais vs Previstos:")
        print(df_comparison.round(3).to_string(index=False))

        df_metrics = pd.DataFrame(results)
        print("\nMétricas de Performance:")
        print(df_metrics.round(4).to_string(index=False))

        return df_comparison, df_metrics

    def plot_comparison(self, save_path=None):
        if self.true_values is None:
            self.get_validation_data()

        plt.figure(figsize=(14, 8))

        full_data = self.df_filtrado.set_index("ano")["desmatado"]

        plt.plot(full_data.index, full_data.values, 'ko-',
                 label='Dados Históricos (2000-2023)', linewidth=2, markersize=4, alpha=0.8)

        validation_data = full_data[full_data.index >= self.val_start]
        plt.plot(validation_data.index, validation_data.values, 'ro-',
                 label='Valores Reais (Validação 2021-2023)', linewidth=3, markersize=8, markerfacecolor='red')

        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        markers = ['s', 'D', '^', 'v', '<', '>', 'o']

        for i, (model_name, pred_dict) in enumerate(self.predictions.items()):
            if pred_dict:
                pred_years = sorted(pred_dict.keys())
                pred_values = [pred_dict[year] for year in pred_years]
                plt.plot(pred_years, pred_values,
                         marker=markers[i % len(markers)], linestyle='--',
                         color=colors[i % len(colors)],
                         label=f'{model_name} (Previsto)',
                         markersize=8, linewidth=2, alpha=0.8)

        plt.axvline(x=self.train_end, color='gray', linestyle=':', alpha=0.7,
                    label='Fim do Treino (2020)')
        plt.axvline(x=self.val_start, color='red', linestyle=':', alpha=0.5,
                    label='Início Validação (2021)')

        plt.xlabel('Ano', fontsize=12)
        plt.ylabel('Área Desmatada (km²)', fontsize=12)
        plt.title('Comparação de Modelos: Previsões vs Realidade (2000-2023)', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nGráfico salvo em: {save_path}")

        plt.show()

    def generate_insights(self, df_metrics):
        valid_metrics = df_metrics.dropna(subset=['RMSE'])

        if len(valid_metrics) == 0:
            print("Nenhum modelo teve métricas válidas para análise.")
            return

        best_rmse_model = valid_metrics.loc[valid_metrics['RMSE'].idxmin()]
        best_mae_model = valid_metrics.loc[valid_metrics['MAE'].idxmin()]
        best_r2_idx = valid_metrics['R²'].idxmax() if not valid_metrics['R²'].isna().all() else None
        best_r2_model = valid_metrics.loc[best_r2_idx] if best_r2_idx is not None else None

        print(f"Melhor modelo (RMSE): {best_rmse_model['Modelo']} (RMSE: {best_rmse_model['RMSE']:.3f})")
        print(f"Melhor modelo (MAE): {best_mae_model['Modelo']} (MAE: {best_mae_model['MAE']:.3f})")

        if best_r2_model is not None:
            print(f"Melhor ajuste (R²): {best_r2_model['Modelo']} (R²: {best_r2_model['R²']:.3f})")