from datetime import date

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from typing import cast


def process_date_column(df: pd.DataFrame, filter_from=str(date.today()), columns=[]):
    for column in columns:
        if not is_datetime(df[column]):
            df[column] = pd.to_datetime(df[column], format="%d/%m/%Y")
        df = cast(pd.DataFrame, df[df[column] >= "2024-01-03"])
        df.loc[:, column] = df[column].apply(lambda x: x.date())
    return df
