import pandas as pd
import numpy as np

from bleu_and_reverse_bleu import bleu
from likelihood_and_ws_likelihood import MarkovChain, liklihood, ws_likelihood
from discriminator_and_ws_discriminator import discriminator_model, discrimiator_score, ws_discrimiator_score

orig = pd.DataFrame(
    [
        [1,1,1,1,2,2,2,2,1,1,1,1],
        [1,1,1,2,2,2,2,1,1,1,1,1],
        [2,2,2,2,3,3,3,3,2,2,2,2],
        [2,2,2,2,4,4,4,4,4,2,2,2],
        [1,1,1,1,3,3,3,3,3,3,1,1],
        [2,2,3,3,3,3,3,2,2,2,2,2],
        [2,2,2,2,2,2,4,4,4,4,4,2],
    ],
    columns=[f"1_Hour{i}" for i in range(12)]
).astype(str)

synth = pd.DataFrame(
    [
        [1,1,1,1,2,2,2,2,1,1,1,1],
        [1,1,1,2,2,2,2,1,1,1,1,1],
        [2,2,2,2,3,3,3,3,2,2,2,2],
        [2,2,2,2,4,4,4,4,4,2,2,2],
        [2,2,2,2,1,1,1,1,1,2,2,2]
    ],
    columns=[f"1_Hour{i}" for i in range(12)]
).astype(str)

# ##############################
# ########### BLEU #############
#
# print("BLEU score", bleu(synth, orig, 3, False))
# print("Reverse BLEU score", bleu(synth, orig, 3, True))
#
# ##############################
# ######## Likelihood ##########
#
# markov_chain_model = MarkovChain(3, orig.columns)
# # markov_chain_model.train(orig)
#
# print("Likelihood score", liklihood(synth, markov_chain_model))
# print("WS likelihood score", ws_likelihood(synth, orig, markov_chain_model))
#
##############################
####### Discriminator ########

d = discriminator_model()
d.train_model(orig, random_data_size=5, batch_size=4, epochs=1)

print("Discriminator score", discrimiator_score(d, synth))
print("WS Discriminator score", ws_discrimiator_score(d, synth, orig))
