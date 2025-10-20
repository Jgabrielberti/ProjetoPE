import pandas as pd
from typing import Tuple, List


def get_municipio_df(municipio_id: int, bioma: str, df: pd.DataFrame) -> pd.DataFrame:
    df_filtrado = df[(df["id_municipio"] == municipio_id) &
                (df["bioma"] == bioma)].sort_values("ano").reset_index(drop=True)

    if df_filtrado.empty:
        raise ValueError(f"Município {municipio_id} no bioma {bioma} não encontrado no dataframe.")

    return df_filtrado


def train_val_split(df_filtrado: pd.DataFrame,
                    train_end: int = 2020,
                    val_start: int = 2021,
                    val_end: int = 2023,
                    future_years: List[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:

    if future_years is None:
        future_years = [2024, 2025, 2026, 2027, 2028]

    train_df = df_filtrado[df_filtrado["ano"] <= train_end].copy()
    val_df = df_filtrado[(df_filtrado["ano"] >= val_start) & (df_filtrado["ano"] <= val_end)].copy()

    if train_df.empty:
        raise ValueError("Train dataframe ficou vazio — verifique train_end.")
    if val_df.empty:
        raise ValueError("Val dataframe ficou vazio — verifique val_start/val_end.")

    return train_df, val_df, future_years