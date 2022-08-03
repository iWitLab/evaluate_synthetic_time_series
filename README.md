# evaluating_synthetic_time_series
The repository contains code that evaluates measures for synthetic time series data, such as synthesis of the locations a set of people visit over a couple of weeks. The scripts compare the synthetic data to the original data and analyze how well the synthesis preserves the privacy of the original data subjects, the statistical similarity, the per-instance similarity, and the diversity of the synthetic data.

Privay measures:
- reidentification_probability.py
- proportion_of_identical_diaries.py

Statistical similarity measures:
- statistical area distribution (in statistics.py)
- transition matrix (in statistics.py)
- worl hours distribution (in statistics.py)

Per-instance similarity:
- likelihood (in likelihood_and_ws_likelihood.py)
- BLEU (in bleu_and_reverse_bleu.py)
- Discriminator (in discriminator_and_ws_discriminator.py)

Diversity measures:
- WS likelihood (in likelihood_and_ws_likelihood.py)
- Reverse BLEU (in bleu_and_reverse_bleu.py)
- WS Discriminator (in discriminator_and_ws_discriminator.py)

Examples of how to use these measures are in Examples_code.py.
