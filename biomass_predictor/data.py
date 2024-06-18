import os
from datetime import timedelta

import geopandas
import pandas as pd
from tqdm import tqdm
from typing import cast
from functools import partial


import utils

DATABASE_URI = os.getenv("DATABABASE_URI")
BIOMASS_SPREADSHEET_URI: str = cast(str, os.getenv("BIOMASS_SPREADSHEET_URI"))
ROTATION_SPREADSHEET_URI: str = cast(str, os.getenv("ROTATION_SPREADSHEET_URI"))


def _get_climate_data() -> pd.DataFrame:
    df = pd.read_sql_table("estacion_climatica", DATABASE_URI)
    df = utils.process_date_column(df, filter_from="2024-01-01", columns=["created_at"])
    daily_df = (
        df.groupby("created_at")
        .agg(
            mean_temperature=("temperatura", "mean"),
            mean_humidity=("humedad", "mean"),
            total_rain=("total_lluvia", "sum"),
        )
        .reset_index()
    )
    return cast(pd.DataFrame, daily_df)


def _get_light_data() -> pd.DataFrame:
    df = pd.read_sql_table("medidor_de_nivel_de_luz", DATABASE_URI)
    df = utils.process_date_column(df, filter_from="2024-01-01", columns=["created_at"])
    daily_df = (
        df.groupby("created_at")
        .agg(
            total_light=("iluminacion", "sum"),
        )
        .reset_index()
    )
    return cast(pd.DataFrame, daily_df)


def _get_biomass_data() -> pd.DataFrame:
    biomass_csv_export_url = BIOMASS_SPREADSHEET_URI.replace(
        "/edit#gid=", "/export?format=csv&gid="
    )
    biomass_df = pd.read_csv(biomass_csv_export_url, decimal=",")
    biomass_df = utils.process_date_column(
        biomass_df, filter_from="2024-01-01", columns=["Fecha"]
    )

    rotation_csv_export_url = ROTATION_SPREADSHEET_URI.replace(
        "/edit#gid=", "/export?format=csv&gid="
    )
    rotation_df = pd.read_csv(rotation_csv_export_url, decimal=",")
    rotation_df = utils.process_date_column(
        rotation_df, filter_from="2024-01-01", columns=["Entrada", "Salida"]
    )

    df_by_potrero = (
        biomass_df.groupby(["Potrero", "Fecha"])
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
        filtered_rotation_df = rotation_df[
            (rotation_df["Potrero"] == row["Potrero"])
            & (rotation_df["Entrada"] <= row["Fecha"])
            & (rotation_df["Salida"] >= row["Fecha"])
        ]
        df_by_potrero.loc[index, "cows_feeding"] = not filtered_rotation_df.empty

    df_by_potrero["biomass_change"] = (
        df_by_potrero["biomass"] - df_by_potrero["previous_biomass"]
    )
    return df_by_potrero


def _get_biomass_growth_data():
    df = _get_biomass_data()
    return df[
        (df["cows_feeding"] == False)
        & (df["days_from_last_measurement"] > 0)
        & (df["biomass_change"] > 0)
    ]


def get_growth_dataset():
    climate_df = _get_climate_data()
    light_df = _get_light_data()
    biomass_df = _get_biomass_growth_data()

    def get_cumulative_column(column_name, source_df, row):
        cumulative_value = source_df[
            (source_df["created_at"] >= row["last_measurement_at"])
            & (source_df["created_at"] < row["Fecha"])
        ][column_name].sum()
        return cumulative_value

    biomass_df["cumulative_temperature"] = biomass_df.apply(
        partial(get_cumulative_column, "mean_temperature", climate_df), axis=1
    )

    biomass_df["cumulative_humidity"] = biomass_df.apply(
        partial(get_cumulative_column, "mean_humidity", climate_df), axis=1
    )

    biomass_df["cumulative_rain"] = biomass_df.apply(
        partial(get_cumulative_column, "total_rain", climate_df), axis=1
    )

    biomass_df["cumulative_light"] = biomass_df.apply(
        partial(get_cumulative_column, "total_light", light_df), axis=1
    )
    # df = biomass_df.merge(
    #     climate_df.merge(light_df, on="created_at"),
    #     left_on="Fecha",
    #     right_on="created_at",
    # )
    #
    # biomass_df.to_csv("biomass_change.csv", index=False)

    return biomass_df
