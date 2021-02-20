
def identical_diaries(synthetic_data, original_diary):
    """
    Count how many identical diaries are in the synthetic data divided by the total_number of synthetic diaries

    Parameters
    ----------
    synthetic_data: pandas.DataFrame
        synthetic data
    original_diary: pandas.DataFrame
        original data

    Returns
    -------
    score: float
    """
    original_data_string = original_diary.astype(str).apply(lambda x: '_'.join(x), axis=1).values
    synthetic_data_string = synthetic_data.astype(str).apply(lambda x: '_'.join(x), axis=1).values
    score = len(set(synthetic_data_string) & set(original_data_string))/len(synthetic_data_string)

    return score
