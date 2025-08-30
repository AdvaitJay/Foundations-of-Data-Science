import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

def process_data(df):
    df=df.drop(columns=['RecordID'])
    mode_value = df['ExtremeWeatherEvent'].mode()[0]
    df['ExtremeWeatherEvent'] = df['ExtremeWeatherEvent'].fillna(mode_value)
    return df
