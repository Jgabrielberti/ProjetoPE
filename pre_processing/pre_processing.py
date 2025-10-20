import os
import pandas as pd

def preprocess_and_save(raw_path: str, processed_path: str):
    df = pd.read_csv(raw_path, encoding="utf-8", low_memory=False)

    # Ordenar por bioma, id_municipio e ano
    df_sorted = df.sort_values(
        by=["bioma", "id_municipio", "ano"], ascending=[True, True, True]
    ).reset_index(drop=True)

    sample_mun = df_sorted["id_municipio"].drop_duplicates().sample(3, random_state=42).tolist()
    sample_data = df_sorted[df_sorted["id_municipio"].isin(sample_mun)].head(240)

    print("\nAmostra de dados ordenados:")
    print(sample_data.to_string(index=False))

    # Checar integridade: número de anos por município
    agg_years_per_mun = df_sorted.groupby("id_municipio")["ano"].nunique().describe()
    print("\nDistribuição do número de anos por município:")
    print(agg_years_per_mun.to_string())

    # Checar integridade por bioma
    agg_years_per_mun_bioma = (
        df_sorted.groupby(["bioma", "id_municipio"])["ano"]
        .nunique()
        .groupby("bioma")
        .describe()
    )
    print("\nDistribuição por bioma:")
    print(agg_years_per_mun_bioma)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    df_sorted.to_csv(processed_path, index=False, encoding="utf-8")
    print(f"\nArquivo processado salvo em: {processed_path}")

    return df_sorted, processed_path
