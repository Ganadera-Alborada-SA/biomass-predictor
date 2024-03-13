import os
from datetime import timedelta

import geopandas
import pandas as pd
from tqdm import tqdm

DATABASE_URI = os.getenv("DATABABASE_URI")
SPREADSHEET_URI = os.getenv("SPREADSHEET_URI")


def _get_climate_data():
    df = pd.read_sql_table("estacion_climatica", DATABASE_URI)
    df = df[df["created_at"] >= "2024-01-01"]
    df["created_at_date"] = df["created_at"].apply(lambda x: x.date())
    daily_df = df.groupby("created_at_date").agg(
        mean_temperature=("temperatura", "mean"),
        mean_humedad=("humedad", "mean"),
        total_rain=("total_lluvia", "sum"),
    )
    return daily_df


def _get_light_data():
    df = pd.read_sql_table("medidor_de_nivel_de_luz", DATABASE_URI)
    df = df[df["created_at"] >= "2024-01-03"]
    df["created_at_date"] = df["created_at"].apply(lambda x: x.date())
    daily_df = df.groupby("created_at_date").agg(
        total_light=("iluminacion", "sum"),
    )
    return daily_df


def _get_biomass_data_samples():
    csv_export_url = SPREADSHEET_URI.replace("/edit#gid=", "/export?format=csv&gid=")
    df = pd.read_csv(csv_export_url, decimal=",")
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y")
    df = df[df["Fecha"] >= "2024-01-03"]
    df["Fecha"] = df["Fecha"].apply(lambda x: x.date())

    df_by_potrero = (
        df.groupby(["Potrero", "Fecha"])
        .agg(
            biomass=("MS_Ha", "median"),
        )
        .reset_index()
    )

    df_by_potrero = df_by_potrero.sort_values(
        ["Potrero", "Fecha"], ascending=[True, True]
    )
    df_by_potrero["days_from_last_measurement"] = (
        df_by_potrero.groupby("Potrero")["Fecha"].diff().fillna(timedelta(0))
    ).apply(lambda x: x.days)

    df_by_potrero["last_measurement_at"] = df_by_potrero["Fecha"] - df_by_potrero[
        "days_from_last_measurement"
    ].apply(lambda x: timedelta(days=x))

    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.Longitud, df.Latitud), crs="EPSG:4326"
    )

    for index, row in tqdm(gdf.iterrows()):
        by_potrero_record = df_by_potrero[
            (df_by_potrero["Potrero"] == row["Potrero"])
            & (df_by_potrero["Fecha"] == row["Fecha"])
        ].iloc[0]
        last_measurement_at = by_potrero_record["last_measurement_at"]
        days_from_last_measurement = by_potrero_record["days_from_last_measurement"]

        if not last_measurement_at == row["Fecha"]:
            row_df = row.to_frame().T
            row_gdf = geopandas.GeoDataFrame(
                row_df,
                geometry=geopandas.points_from_xy(row_df.Longitud, row_df.Latitud),
                crs="EPSG:4326",
            )
            previous_measurement_df = gdf[
                (gdf["Potrero"] == row["Potrero"])
                & (gdf["Fecha"] == last_measurement_at)
            ]
            gdf.loc[index, "previous_MS_Ha"] = (
                row_gdf.to_crs(epsg=32723)
                .sjoin_nearest(previous_measurement_df.to_crs(epsg=32723))
                .iloc[0]["MS_Ha_right"]
            )
        else:
            gdf.loc[index, "previous_MS_Ha"] = None

        gdf.loc[index, "days_from_last_measurement"] = days_from_last_measurement

    return gdf


def _get_biomass_data():
    csv_export_url = SPREADSHEET_URI.replace("/edit#gid=", "/export?format=csv&gid=")
    df = pd.read_csv(csv_export_url, decimal=",")
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y")
    df = df[df["Fecha"] >= "2024-01-03"]
    df["Fecha"] = df["Fecha"].apply(lambda x: x.date())

    df_by_potrero = (
        df.groupby(["Potrero", "Fecha"])
        .agg(
            biomass=("MS_Ha", "median"),
        )
        .reset_index()
    )

    df_by_potrero = df_by_potrero.sort_values(
        ["Potrero", "Fecha"], ascending=[True, True]
    )
    df_by_potrero["days_from_last_measurement"] = (
        df_by_potrero.groupby("Potrero")["Fecha"].diff().fillna(timedelta(0))
    ).apply(lambda x: x.days)

    df_by_potrero["last_measurement_at"] = df_by_potrero["Fecha"] - df_by_potrero[
        "days_from_last_measurement"
    ].apply(lambda x: timedelta(days=x))

    for index, row in tqdm(df_by_potrero.iterrows()):
        previous_record = df_by_potrero[
            (df_by_potrero["Potrero"] == row["Potrero"])
            & (df_by_potrero["Fecha"] == row["last_measurement_at"])
        ].iloc[0]

        df_by_potrero.loc[index, "previous_biomass"] = previous_record["biomass"]

    return df_by_potrero


def _get_biomass_growth_data():
    df = _get_biomass_data()
    return df[df["days_from_last_measurement"] == 1]


def get_growth_dataset():
    climate_df = _get_climate_data()
    light_df = _get_light_data()
    biomass_df = _get_biomass_growth_data()

    df = biomass_df.merge(
        climate_df.merge(light_df, on="created_at_date"),
        left_on="Fecha",
        right_on="created_at_date",
    )
    return df
