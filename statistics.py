import pandas as pd
import numpy as np


def get_loc_stats(data):
    """
    get distribution of statistical areas of data

    Parameters
    ----------
    data: pandas.DataFrame

    Returns
    -------
    distribution: pandas.DataFrame
    """
    distribution = pd.melt(data, var_name='time', value_name='location')
    distribution['ind'] = 1
    distribution = distribution.groupby(['time', 'location'])['ind'].sum().reset_index()
    distribution['share'] = distribution.ind / distribution.ind.sum()

    return distribution


def kl_divergence(p, q):
    """

    Parameters
    ----------
    p: pandas.Series
    q: pandas.Series

    Returns
    -------
    kl divergence score
    """
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def statistical_area_distribution(synthetic_data, original_data):
    """
    Get the distributions of the synthetic data and original data.
    Than calculate the kl divergence between them.

    Parameters
    ----------
    synthetic_data: pandas.DataFrame
        synthetic data
    original_data: pandas.DataFrame
        original data

    Returns
    -------
    score: float
    """
    original_data_dist = get_loc_stats(original_data)
    synthetic_data_dist = get_loc_stats(synthetic_data)

    merged = original_data_dist.merge(
        synthetic_data_dist,
        on=['time', 'location'],
        how='outer').fillna(0.000001)

    return kl_divergence(merged.share_x, merged.share_y)


# def get_transition_data(data):
#     melted_data = pd.melt(data.reset_index(), id_vars=['data', 'index'])
#     melted_data = melted_data[melted_data.value != -1]
#
#     melted_data['time'] = melted_data.variable.apply(lambda x: order_cols_weekday.index(x))
#     melted_data['day'] = melted_data.variable.apply(lambda x: int(x.split('_')[0]))
#     melted_data.sort_values('time', inplace=True)
#     melted_data['new_time'] = melted_data.groupby(['data', 'index', 'day']).cumcount() + 1
#     melted_data['new_time_plus'] = melted_data['new_time'] + 1
#
#     melted_data_merge = melted_data.merge(melted_data, left_on=['data', 'index', 'new_time_plus'],
#                                           right_on=['data', 'index', 'new_time'])
#     melted_data_merge = melted_data_merge[
#         (melted_data_merge.value_x != melted_data_merge.value_y) & (melted_data_merge.day_x == melted_data_merge.day_y)]
#     grouped = melted_data_merge.groupby(['data', 'value_x', 'value_y', 'variable_x']).time_x.count().reset_index()
#     grouped.columns = ['data', 'from', 'to', 'start_time', 'cnt']
#
#     return grouped
