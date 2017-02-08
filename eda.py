import pandas as pd
import numpy as np

def load_and_clean_data(filename='data/churn_sample.csv'):

    df = pd.read_csv(filename)
    df.index = np.arange(len(df.index))
    # change date strings to datetimes
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    # any last trips before June count as churn
    churn_date = pd.to_datetime('2014-06-01')
    df['churn'] = (df['last_trip_date'] < churn_date).astype('int')
    df_categorical = get_categorical_df(df)
    df_categorical = pd.get_dummies(df_categorical, columns=['phone', 'avg_rating_by_driver_quantiles', 'avg_rating_of_driver_quantiles', 'luxury_car_user', 'city'], drop_first=True)
    df_numeric = get_numeric_df(df)
    return df_numeric, df_categorical

def get_numeric_df(df):
    df = df.copy()
    # change luxury car to int
    df['luxury_car_user'] = df['luxury_car_user'].astype('int')
    # change phone to dummies
    df = pd.get_dummies(df, columns=['phone'], dummy_na=True)
    df.drop('phone_nan', axis=1, inplace=True)
    # change city to dummies
    df = pd.get_dummies(df, columns=['city'], drop_first=True)
    # change nans to median (typical values?)
    df.loc[df['avg_rating_by_driver'].isnull(),'avg_rating_by_driver'] = df['avg_rating_by_driver'].median()
    df.loc[df['avg_rating_of_driver'].isnull(),'avg_rating_of_driver'] = df['avg_rating_of_driver'].median()
    return df

def get_categorical_df(df):
    df = df.copy()
    df = separate_quantiles(df, 'avg_rating_by_driver')
    df = separate_quantiles(df, 'avg_rating_of_driver')
    df.drop(['avg_rating_by_driver'], axis=1, inplace=True)
    df.drop(['avg_rating_of_driver'], axis=1, inplace=True)
    return df

def separate_quantiles(df, column_name):
    quantile_column = column_name + '_quantiles'
    df[quantile_column] = 'nan'
    # df[df[column_name]==np.nan][quantile_column] = 'nan'
    labels = ['q1', 'q2', 'q3', 'q4']
    lower_qs = [0., 0.25, 0.5, 0.75]
    upper_qs = [.25, .5, .75, 1.]
    for (lower_q, upper_q, label) in zip(lower_qs, upper_qs, labels):
        quantile_df = get_quantile(df, lower_q, upper_q, column_name)
        df.loc[quantile_df.index,quantile_column] = label
    return df

def get_quantile(df, lower_q, upper_q, column_name):
    lower_val = df[column_name].dropna().quantile(lower_q)
    upper_val = df[column_name].dropna().quantile(upper_q)
    ret_df = df[(df[column_name]>=lower_val) & (df[column_name]<=upper_val)]
    return ret_df
