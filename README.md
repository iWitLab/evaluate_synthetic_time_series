# evaluating_synthetic_time_series
Evaluation measures for synthetic time series data:

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
