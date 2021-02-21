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


def get_transition_data(data):
    """
    Create transition table

    Parameters
    ----------
    data: pandas.DataFrame

    Returns
    -------
    transfers: pandas.DataFrame
    """
    melted_data = data.copy()
    cols = melted_data.columns.to_list()

    # melt data
    melted_data.index.rename('User', inplace=True)
    melted_data = pd.melt(melted_data.reset_index(), id_vars=['User'], var_name='day_hour', value_name='location')

    # remove transitions
    melted_data = melted_data[melted_data.location != -1]

    # create index column for each user
    melted_data['time'] = melted_data.day_hour.apply(lambda x: cols.index(x))
    melted_data['day'] = melted_data.day_hour.apply(lambda x: int(x.split('_')[0]))
    melted_data.sort_values('time', inplace=True)

    # since we removed the traveling time we need the time column to fill the gaps
    melted_data['time'] = melted_data.groupby(['User', 'day']).cumcount() + 1
    melted_data['time_plus'] = melted_data['time'] + 1

    # merge the time with time plus to get the next location for every location
    transfers = melted_data.merge(melted_data, left_on=['User', 'time_plus'], right_on=['User', 'time'])

    # keep only times when user moves
    transfers = transfers[
        (transfers.location_x != transfers.location_y) &
        (transfers.day_x == transfers.day_y)
    ]

    # count the number of transfers by departure (location_x) and arival (location_y) locations
    transfers = transfers.groupby(['location_x', 'location_y', 'day_hour_x']).time_x.count().reset_index()
    transfers.columns = ['depart', 'arrive', 'start_time', 'cnt']
    transfers['share'] = transfers.cnt / transfers.cnt.sum()

    return transfers


def transitions_score(synthetic_data, original_data):
    """
    Get the transitions of the synthetic data and original data.
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

    orig_transitions = get_transition_data(original_data)
    synth_transitions = get_transition_data(synthetic_data)

    merged = (
        orig_transitions
        .merge(synth_transitions, on=['depart', 'arrive', 'start_time'], how='outer')
        .fillna(0.00000001)
    )
    return kl_divergence(merged.share_x, merged.share_y)


def user_type(locations, working_times):
    """
    Function determines if user is worker or not

    Parameters
    ----------
    locations: pandas.Series
        sequence of locations
    working_times: list
        list of columns of working days

    Returns
    -------
    is_worker: bool
        1 if worker
    """
    if locations[working_times].value_counts().iloc[1] / len(working_times) >= 0.2:
        return 1
    else:
        return 0


def working_hours(data, working_times, number_of_work_days):
    """
    Calculate working hour for each user if the user is a worker else working hours are 0

    Parameters
    ----------
    data: pandas.DataFrame
    working_times: list
        list of columns of working days
    number_of_work_days: int
        number of working days in the data

    Returns
    -------
    working_hours: pandas.DataFrame
    """
    work_hours = data.copy()
    work_hours['is_worker'] = work_hours.apply(lambda x: user_type(x, working_times), axis=1)
    work_hours['work_place'] = work_hours.apply(
        lambda x: None if x.is_worker == 0 else x[working_times].value_counts().index[1], axis=1)
    work_hours['work_hours'] = work_hours.apply(
        lambda x: 0 if x.is_worker == 0 else int(len([val for val in x[working_times] if val == x.work_place]) / number_of_work_days),
        axis=1
    )

    work_hours_distribution = work_hours.groupby('work_hours').is_worker.count().reset_index()
    work_hours_distribution.rename(columns={'is_worker': 'cnt'}, inplace=True)
    work_hours_distribution['share'] = work_hours_distribution.cnt / work_hours_distribution.cnt.sum()

    return work_hours_distribution


def working_hours_distribution(synthetic_data, original_data, working_times, number_of_work_days):
    """
    Get the working hours distribution of the synthetic data and original data.
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
    orig_working_hours = working_hours(original_data, working_times, number_of_work_days)
    synth_working_hours = working_hours(synthetic_data, working_times, number_of_work_days)

    merged = (
        orig_working_hours
        .merge(synth_working_hours, on=['work_hours'], how='outer')
        .fillna(0.00000001)
    )
    return kl_divergence(merged.share_x, merged.share_y)
