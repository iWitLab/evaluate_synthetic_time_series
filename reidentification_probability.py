import numpy as np
import pandas as pd


def reidentification_prob(synthetic_data, original_data, n, number_of_test_users, random_state=0):
    """
    calculate the reidentification probability:
    1. sample users as initial information
    2. sample times in which we know the locations for each user
    3. join to get for each user the time and location that were sampled
    4. find the synthetic users with similar location in the times we sampled for each original user
    5. keep only users that had the most matches
    6. get the synthetic users entire sequences
    7. for each real user get an estimated diary using the synthetic users
    8. compare the estimated diaries with the real diaries

    Parameters
    ----------
    synthetic_data: pandas.DataFrame
        synthetic data
    original_data: pandas.DataFrame
        original data
    n: int
        number of locations that are known
    number_of_test_users:
        number of original users to test
    random_state: int

    Returns
    -------
    score: float
    """
    cols = original_data.columns
    sequence_len = original_data.shape[1]

    # sample users as initial information
    test_users = original_data.sample(number_of_test_users, random_state=random_state)
    sample_sequence_time = np.random.randint(sequence_len, size=(number_of_test_users, n))

    # sample times in which we know the locations for each user
    sample_sequence_time = pd.melt(
        pd.DataFrame(sample_sequence_time, index=test_users.index).reset_index(),
        id_vars=['User'],
        value_name='time'
    ).drop(['variable'], axis=1)
    sample_sequence_time['time'] = sample_sequence_time['time'].apply(lambda x: cols[x])
    test_users = pd.melt(test_users.reset_index(), id_vars=['User'], var_name='time', value_name='location')

    # join to get for each user the time and location that were sampled
    test_users_known_locations = test_users.merge(sample_sequence_time, on=['User', 'time'])

    # find the synthetic users with similar location in the times we sampled for each original user
    synthetic_data.index.rename('synth_user', inplace=True)
    synthetic_melted_data = pd.melt(
        synthetic_data.reset_index(),
        id_vars='synth_user',
        var_name='time',
        value_name='synth_location'
    )

    test_users_known_locations_with_synthetic_users = test_users_known_locations.merge(
        synthetic_melted_data, on=['time'], how='left')
    test_users_known_locations_with_synthetic_users['is_same'] = \
        test_users_known_locations_with_synthetic_users.apply(lambda x: x.location == x.synth_location, axis=1)
    test_users_known_locations_with_synthetic_users = test_users_known_locations_with_synthetic_users\
        .groupby(['User', 'synth_user']).is_same.mean().reset_index()
    test_users_known_locations_with_synthetic_users['max_same'] = test_users_known_locations_with_synthetic_users\
        .groupby('User').is_same.transform(lambda x: max(x))

    # keep only users that had the most matches
    test_users_known_locations_with_synthetic_users = test_users_known_locations_with_synthetic_users[
        (test_users_known_locations_with_synthetic_users.apply(lambda x: x.is_same == x.max_same, axis=1)) &
        (test_users_known_locations_with_synthetic_users.max_same > 0.8)
    ]

    # get the synthetic users entire sequences
    test_users_known_locations_with_synthetic_users = (
        test_users_known_locations_with_synthetic_users[['User', 'synth_user']]
        .merge(synthetic_data.reset_index(), on='synth_user').drop(['synth_user'], axis=1)
    )
    test_users_known_locations_with_synthetic_users = pd.melt(
        test_users_known_locations_with_synthetic_users,
        id_vars=['User'],
        var_name='time',
        value_name='synth_location'
    )

    # for each real user get an estimated diary using the synthetic users
    test_users_known_locations_with_synthetic_users = (
        test_users_known_locations_with_synthetic_users
        .groupby(['User', 'time']).synth_location
        .apply(lambda x: x.value_counts().index[0])
        .reset_index()
    )

    # compare the estimated diaries with the real diaries
    compare = test_users.merge(test_users_known_locations_with_synthetic_users, on=['User', 'time'], how='left')
    compare['is_same'] = compare.apply(lambda x: 1 if x.location == x.synth_location else 0, axis=1)
    compare = compare.groupby('User').is_same.mean()

    return compare.mean()