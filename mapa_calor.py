import pandas as pd
import folium
import geobr
from folium.plugins import HeatMap

def criar_mapa_calor(df: pd.DataFrame):
    municipios = geobr.read_municipality(year=2020)
    municipios = municipios.to_crs(epsg=4326)

    dados_geo = municipios.merge(df, left_on='code_muni', right_on='id_municipio', how='inner')

    m = folium.Map(location=[-15.5, -56.1], zoom_start=5, tiles='CartoDB positron')

    ano_filtro = 2023
    df_ano = dados_geo[dados_geo['ano'] == ano_filtro].copy()

    df_ano['latitude'] = df_ano.geometry.centroid.y
    df_ano['longitude'] = df_ano.geometry.centroid.x

    df_ano = df_ano.dropna(subset=['latitude', 'longitude', 'desmatado'])

    df_ano['intensidade'] = df_ano['desmatado'] / df_ano['desmatado'].max()

    heat_data = df_ano[['latitude', 'longitude', 'intensidade']].values.tolist()

    HeatMap(
        data=heat_data,
        radius=4,
        blur=6,
        min_opacity=0.3,
        max_opacity=0.9,
        gradient={
            0.2: 'yellow',
            0.4: 'orange',
            0.6: 'red',
            0.8: 'darkred',
            1.0: 'black'
        }
    ).add_to(m)

    m.save('mapa_calor_desmatamento.html')
    print("Mapa de calor salvo como 'mapa_calor_desmatamento.html'")
