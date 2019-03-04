import pandas as pd
from os.path import realpath
from os import listdir
import numpy as np

__all__ = ['_get_csvs', '_get_event_csvs', '_idx_from_date', '_get_returns']
DATE_RANGE = list(pd.read_csv('./Twitter_sentiment_DJIA30/twitter_data_CVX.csv')['Unnamed: 0'])


def _get_returns(df):
    return list((df['CLOSE'] - df['OPEN']) / df['OPEN'])


def _get_csvs(data_dir='./Twitter_sentiment_DJIA30/', include_djia=False, prepro=True):
    tag = 'financial%s_data' % ('_prepro' if prepro else '')
    def keep(f):
        return include_djia or 'DJIA' not in f
    return [realpath(data_dir + f) for f in listdir(data_dir) if tag in f and keep(f)]


def _get_event_csvs(data_dir='./Twitter_sentiment_DJIA30/'):
    return [realpath(data_dir + f) for f in listdir(data_dir) if 'twitter' in f]


def _increment(date, n=1):
    return DATE_RANGE[DATE_RANGE.index(date) + n]


def _idx_from_date(df, date):
    """
    Returns index of row with given date
    :param df: pandas dataframe object
    :param date: the date specifying the row index to return in YYYY-MM-DD format
    :return: the index of the row
    """
    # Check that first idx is 0
    assert df.loc[0]['Unnamed: 0'] == df['Unnamed: 0'].values[0], "First index is not 0, returned index would be wrong"
    return list(df['Unnamed: 0']).index(date)


"""
Unnecessary now that data has been cleaned
"""
def _idx_from_date_old(df, date, first_date):
    """
    Returns index of row with given date
    :param df: pandas dataframe object
    :param date: the date specifying the row index to return in YYYY-MM-DD format
    :return: the index of the row
    """
    df_list = list(df['Unnamed: 0'])

    def decrement(string):
        string = str(int(string) - 1)
        return '0%s' % string if len(string) == 1 else string

    split = date.split('-')
    while '-'.join(split) not in df_list:
        if '-'.join(split) == first_date:
            return 0
        if split[-1] == '01':
            if split[-2] == '01':
                split[-3] = decrement(split[-3])
                split[-2] = '12'
                split[-1] = '31'
                continue
            split[-2] = decrement(split[-2])
            split[-1] = '31'
            continue
        split[-1] = decrement(split[-1])
        continue
    return df_list.index('-'.join(split))


def _clip_dates(df):
    assert df.loc[0]['Unnamed: 0'] == df['Unnamed: 0'].values[0], "First index is not 0, returned index would be wrong"
    # Get index of first date match
    date_list = list(df['Unnamed: 0'])
    first_idx = date_list.index(DATE_RANGE[0])
    last_idx = date_list.index(DATE_RANGE[-1])
    # Append with ignore_index to reorder indices
    return pd.DataFrame().append(df.loc[first_idx:last_idx], ignore_index=True)


def normalize_dates():
    for csv in _get_csvs(include_djia=True, prepro=False):
        df_old = pd.read_csv(csv)
        if df_old['Unnamed: 0'].values[0] not in DATE_RANGE or df_old['Unnamed: 0'].values[-1] not in DATE_RANGE:
            df_old = _clip_dates(df_old)
        df_new = pd.DataFrame()

        def null_df(date):
            return pd.DataFrame.from_dict({'Unnamed: 0': [date],
                                           'HIGH': [np.nan],
                                           'LOW': [np.nan],
                                           'CLOSE': [np.nan]})

        def row_from_date(df, d):
            i = _idx_from_date(df, d)
            # vvv Panda's indexing is #wack
            return df.loc[i:i]

        for d in DATE_RANGE:
            if d in df_old['Unnamed: 0'].values:
                df_new = df_new.append(row_from_date(df_old, d), ignore_index=True)
                continue
            df_new = df_new.append(null_df(d), ignore_index=True)

        new_csv = csv.split('_')
        new_csv[-3] += '_prepro'
        new_csv = '_'.join(new_csv)
        df_new.to_csv(new_csv, index=False)


if __name__ == '__main__':
    normalize_dates()
