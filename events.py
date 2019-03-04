"""
All methods below are used for parsing data from "The Effects of Twitter Sentiment on Stock Price Returns" downloaded
 from https://figshare.com/articles/The_effects_of_Twitter_sentiment_on_stock_price_returns/1533283
"""
import pandas as pd


def _polarity(p):
    """
    Returns overall polarity of an event, normalizing and dividing the distribution of continuous
        polarity values output by sentiment analysis
    :param p: sentiment polarity value
    :return: one of [-1, 0, 1] indicating that events overall polarity
    """

    if p < 0.15:
        return -1
    elif p <= 0.7:
        return 0
    else:
        return 1


def events_from_data(data, L=5, n_min=10, phi_t=2, event_window=21):
    """
    :param data: a pandas dataframe containing twitter activity for a given cash-tag
    :param L: number of days before and after the day of an event that are
                        contained in the event window
    :return: a list of indexes in the data at which twitter events occur
    """

    baseline_activity = list(pd.rolling_median(data['TW'], 2 * L + 1)[2*L:])
    actual_activity = list(data['TW'])[L:-L]

    def phi(d_0):
        return (actual_activity[d_0] - baseline_activity[d_0]) / max(baseline_activity[d_0], n_min)

    activity_peaks = [i + L for i in range(len(actual_activity)) if phi(i) > phi_t]
    filtered = []
    last_peak_idx = activity_peaks[0]
    last_peak_val = data['TW'][last_peak_idx]
    for p in activity_peaks[1:]:
        if p - last_peak_idx < event_window:
            if data['TW'][p] > last_peak_val:
                last_peak_idx = p
                last_peak_val = data['TW'][p]
        else:
            filtered += [last_peak_idx]
            last_peak_idx = p
            last_peak_val = data['TW'][p]
    filtered += [last_peak_idx]

    return activity_peaks, filtered


def get_polarity(events):
    polarity = list((events['NUM_POS'] - events['NUM_NEG']) / (events['NUM_POS'] + events['NUM_NEG']))
    return polarity, [_polarity(p) for p in polarity]


def get_event_polarities(data, L=10, n_min=10, phi_t=2):
    _, idxs = events_from_data(data, L=10, n_min=10, phi_t=2)
    events = data.loc[idxs]
    polarity, threshold_polarity = get_polarity(events)

    assert len(threshold_polarity) == len(idxs)

    return zip(idxs, threshold_polarity)
