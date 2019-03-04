import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

from market_model import MarketModel
from events import get_event_polarities
from utils import *


def get_CAR(events_by_stock, MMs):
    """
    Loads financial data and aggregates returns across stocks and days within the event window
    :param t_1: beginning of event window (exclusive) in YYYY-MM-DD format
    :param t_2: end of event window (inclusive) in YYYY-MM-DD format
    :param MMs: list of market models for each stock index used to approximate returns
    :return: the cumulative abnormal return (CAR) aggregated from the data
    """
    djia_df = pd.read_csv('./Twitter_sentiment_DJIA30/financial_prepro_data_DJIA.csv')
    csvs = _get_csvs()

    num_events = 0
    abnormal_returns_aggregate = None
    for csv in csvs:
        ticker = csv.split('.')[0].split('_')[-1]

        assert len(MMs[ticker]) == len(events_by_stock[ticker])

        stock_df = pd.read_csv(csv)
        for (t_0, t_1, t_2), mm in zip(events_by_stock[ticker], MMs[ticker]):
            # Get returns for stock
            start_idx, end_idx = [_idx_from_date(stock_df, t) + 1 for t in (t_1, t_2)]
            event_window = stock_df.loc[start_idx:end_idx-1]
            actual_returns = np.array(_get_returns(event_window))
            #plt.plot(range(len(actual_returns)), actual_returns)
            #plt.show()

            # Get returns for djia index
            start_idx, end_idx = [_idx_from_date(djia_df, t) + 1 for t in (t_1, t_2)]
            event_window = stock_df.loc[start_idx:end_idx-1]
            djia_returns = np.array(_get_returns(event_window))
            predicted_returns, VARIANCE = mm.predict(djia_returns)

            assert len(predicted_returns) == len(actual_returns), \
                "Lengths of predicted returns don't match actual returns"

            abnormal_returns = actual_returns - predicted_returns
            abnormal_returns[np.isnan(abnormal_returns)] = 0.0
            if abnormal_returns_aggregate is None:
                num_events = np.ones_like(abnormal_returns)
                num_events[np.isnan(abnormal_returns)] = 0
                abnormal_returns_aggregate = abnormal_returns
            else:
                num_events_ = np.ones_like(abnormal_returns)
                num_events_[np.isnan(abnormal_returns)] = 0
                num_events += num_events_
                abnormal_returns_aggregate += abnormal_returns

    abnormal_returns_aggregate /= num_events

    return abnormal_returns_aggregate, np.sum(abnormal_returns_aggregate)

    """
    start_idx, end_idx = [_idx_from_date(df, t) + 1 for t in (t_0, t_1)]
    estimation_window = df.loc[start_idx:end_idx]
    djia_returns = _get_returns(estimation_window)

    #TODO Do weekends contribute to aggregated returns?
    csvs = _get_csvs()

    abnormal_returns = None
    for csv in csvs:
        ticker = csv.split('.')[0].split('_')[-1]

        df = pd.read_csv(csv)
        # shift indices by 1 since t_1 exclusive and t_2 inclusive (contrary to array indexing practices)
        start_idx, end_idx = [_idx_from_date(df, t) + 1 for t in (t_1, t_2)]
        event_window = df.loc[start_idx:end_idx]
        actual_returns = np.array(_get_returns(event_window))

        assert len(djia_returns) == len(actual_returns), "Lengths of DJIA returns don't match actual returns"

        mm = MMs[ticker]
        predicted_returns = mm.predict(np.array(djia_returns))

        if abnormal_returns is None:
            abnormal_returns = actual_returns - predicted_returns
        else:
            abnormal_returns += actual_returns - predicted_returns

    # We divide summed daily abnormal returns by the number of stocks
    abnormal_returns /= len(csvs)

    return abnormal_returns, np.sum(abnormal_returns)
    """


def get_MMs(events_by_stock):
    """
    Builds and fits market models based off of DJIA index for each individual stock index over the
        specified estimation window
    :param events_by_stock: a dictionary referenced by stock ticker containing a list of tuples,
           (t_0, t_1, t_2) defining the estimation and event windows of each of that stock's twitter events
    :return: a dictionary of lists of market models for each stock index being approximated
    """
    djia_df = pd.read_csv('./Twitter_sentiment_DJIA30/financial_prepro_data_DJIA.csv')

    csvs = _get_csvs()
    MMs = {}
    for csv in csvs:
        ticker = csv.split('.')[0].split('_')[-1]
        events = events_by_stock[ticker]
        MMs[ticker] = []

        df = pd.read_csv(csv)
        for t_0, t_1, t_2 in events:
            start_idx, end_idx = [_idx_from_date(df, t) + 1 for t in (t_0, t_1)]
            estimation_window = df.loc[start_idx:end_idx-1]
            stock_returns = _get_returns(estimation_window)

            mm = MarketModel()
            start_idx, end_idx = [_idx_from_date(djia_df, t) + 1 for t in (t_0, t_1)]
            estimation_window = djia_df.loc[start_idx:end_idx-1]
            djia_returns = _get_returns(estimation_window)
            mm.fit(djia_returns, stock_returns)
            MMs[ticker] += [mm]

    return MMs


def main(L_event=10, L_estimation=120):
    # Get events for each stock
    events_by_stock = {1: {}, 0: {}, -1: {}}
    for csv in _get_event_csvs():
        ticker = csv.split('.')[0].split('_')[-1]
        df = pd.read_csv(csv)
        events = get_event_polarities(df)

        # Get estimation and event window for each event (defined by t_0, t_1, t_2)
        t_pos = [tuple([df['Unnamed: 0'][idx + modifier] for modifier in (-L_event - L_estimation, -L_event, L_event)])
                 for idx, polarity in events if L_event + L_estimation < idx <= df.shape[0] - L_event and polarity > 0]
        t_neu = [tuple([df['Unnamed: 0'][idx + modifier] for modifier in (-L_event - L_estimation, -L_event, L_event)])
                 for idx, polarity in events if L_event + L_estimation < idx <= df.shape[0] - L_event and polarity == 0]
        t_neg = [tuple([df['Unnamed: 0'][idx + modifier] for modifier in (-L_event - L_estimation, -L_event, L_event)])
                 for idx, polarity in events if L_event + L_estimation < idx <= df.shape[0] - L_event and polarity < 0]

        events_by_stock[1][ticker] = t_pos
        events_by_stock[0][ticker] = t_neu
        events_by_stock[-1][ticker] = t_neg

    # Fit market models for each event
    MMs_pos = get_MMs(events_by_stock[1])
    MMs_neu = get_MMs(events_by_stock[0])
    MMs_neg = get_MMs(events_by_stock[-1])

    # Get CAR for each event
    ARs_pos, CAR_pos = get_CAR(events_by_stock[1], MMs_pos)
    ARs_neu, CAR_neu = get_CAR(events_by_stock[0], MMs_neu)
    ARs_neg, CAR_neg = get_CAR(events_by_stock[-1], MMs_neg)

    # Aggregate for each lag
    def aggregate(ARs):
        agg = []
        for i in range(1, len(ARs) + 1):
            agg += [np.sum(ARs[:i])]
        return agg

    ARs_pos = aggregate(ARs_pos)
    ARs_neu = aggregate(ARs_neu)
    ARs_neg = aggregate(ARs_neg)

    # T-test
    print(CAR_pos, CAR_neu, CAR_neg)

    # Visualize per-day CAR's
    for ARs, label in ((ARs_pos, 'positive'), (ARs_neu, 'neutral'), (ARs_neg, 'negative')):
        plt.plot(range(len(ARs)), ARs, label=label)
    plt.legend(loc='best')
    plt.show()


def debug():
    peak = pd.read_csv('./Twitter_sentiment_DJIA30/financial_data_BA.csv')
    t0 = peak['Unnamed: 0'][10]
    t1 = peak['Unnamed: 0'][30]
    t2 = peak['Unnamed: 0'][40]
    MMs = get_MMs(t0, t1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--L_event', type=int, default=10)
    parser.add_argument('--L_estimation', type=int, default=120)
    args = parser.parse_args()
    if args.debug:
        debug()
    main(args.L_event, args.L_estimation)
