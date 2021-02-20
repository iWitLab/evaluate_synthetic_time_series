import pandas as pd
import numpy as np


# def get_ngrams(df, n):
#     """
#     Get all n-grams of data in df
#
#     Parameters
#     ----------
#     df: pandas.DataFrame
#         data
#     n: int
#         the size of the n-gram
#
#     Returns
#     -------
#     ngrams: pandas.DataFrame
#     """
#     ngrams = []
#
#     for row in df.values:
#         for i in range(len(row)-n+1):
#             if n == 1:
#                 ngrams.append([i, row[i], 1])
#             else:
#                 ngrams.append([i, '|'.join(row[i:i+n]), 1])
#
#     return pd.DataFrame(ngrams, columns=['time', 'ngram', 'ind'])


def get_ngrams(df, n):
    """
    Get all n-grams of data in df

    Parameters
    ----------
    df: pandas.DataFrame
        data
    n: int
        the size of the n-gram

    Returns
    -------
    ngrams: pandas.DataFrame
    """
    ngrams = []

    for i in range(df.shape[1] - n + 1):
        vals = df.iloc[:, i:i+n].astype('str').apply(lambda x: str(i) + '_' + '_'.join(x), axis=1).unique().tolist()
        list_vals = list(zip(vals, np.ones(len(vals))))
        ngrams = ngrams + list_vals

    return pd.DataFrame(ngrams, columns=['sample_name', 'ind']).groupby(['sample_name']).ind.min().reset_index()


def bleu_score(synth_ngrams, orig_ngrams):
    """
    Given the ngrams occurrence tables calculate the bleu score

    Parameters
    ----------
    synth_ngrams: pandas.DataFrame
        synthetic data ngrams
    orig_ngrams: pandas.DataFrame
        original data ngrams

    Returns
    -------
    score: pandas.DataFrame
    """
    synth_for_join = synth_ngrams.merge(orig_ngrams, on='sample_name', how='left').fillna(0)
    return synth_for_join['ind_y'].sum()/synth_for_join['ind_x'].sum()


def bleu(synthetic_data, original_data, n, is_reverse):
    synth_ngrams = get_ngrams(synthetic_data, n)
    orig_ngrams = get_ngrams(original_data, n)

    if is_reverse:
        return bleu_score(orig_ngrams, synth_ngrams)
    else:
        return bleu_score(synth_ngrams, orig_ngrams)
