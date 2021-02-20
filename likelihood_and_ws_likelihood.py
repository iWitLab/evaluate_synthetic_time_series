from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd
from tqdm import tqdm


class MarkovChain(object):
    def __init__(self, n, cols):
        """
        Initialize the MarkovChain instance.

        Parameters
        ----------
        n : int
            Markov chain order
        cols : list
            input data columns

        """
        self.n = n
        self.cols = cols


    def get_first_hour(self, train_data):
        """
        Create the table of all first hours options

        Parameters
        ----------
        train_data : pandas.DataFrame
            input data for training

        Returns
        -------
        first_hour_prob: pandas.DataFrame
            Table with First hours options and probabilities
        """

        first_hour_prob = self.cols[0] + '_' + ((train_data[self.cols[:self.n]] + '|').sum(axis=1)).apply(lambda x: x[:-1])
        first_hour_prob = first_hour_prob.to_frame()
        first_hour_prob.columns = ['state']
        first_hour_prob['ind'] = 1

        first_hour_prob = first_hour_prob.groupby(['state']).ind.sum().reset_index()
        first_hour_prob['ind'] = first_hour_prob.ind / first_hour_prob.ind.sum()

        return first_hour_prob


    def first_state(self, first_hour_prob):
        """
        Randomly chooses n first location to start sampling from

        Parameters
        ----------
        first_hour_prob: pandas.DataFrame
            Table with First hours options and probabilities

        Returns
        -------
        first_states: numpy.array
            n first states
        """

        return np.random.choice(
            first_hour_prob.state,
            p=first_hour_prob.ind.values
        )


    def get_transition_matrix(self, train_data, i):
        """
        Calculate the transitions probabilities

        Parameters
        ----------
        train_data: pandas.DataFrame
            Training data
        i: int
            The time of the state transition table

        Returns
        -------
        transitions: pandas.DataFrame
            transition table for time i
        """

        transitions = self.cols[i + self.n] + '_' + train_data[[self.cols[i + self.n]]]
        transitions.columns = ['value']
        transitions['state'] = self.cols[i] + '_' + ((train_data[self.cols[i:i + self.n]] + '|').sum(axis=1)).apply(
            lambda x: x[:-1])
        transitions['ind'] = 1

        transitions = transitions.groupby(['state', 'value'])[['ind']].sum().reset_index()
        transitions['ind'] = transitions.groupby(['state']).ind.transform(lambda x: x / x.sum())

        return transitions.set_index('state')


    def next_state(self, current_state, transitions):
        """
        Randomly select the next state

        Parameters
        ----------
        current_state: string
            The current state the chain is in
        transitions: pandas.DataFrame
            the transition table

        Returns
        -------
        next_state: string
        """

        temp_transitions = transitions.loc[current_state]

        try:
            return np.random.choice(
                temp_transitions.value,
                p=temp_transitions.ind.values
            )
        except:
            return temp_transitions.value


    def generate_sequences(self, train_data, samples_number, sequence_length):
        """
        Generate sequences using the input data given
        Saves the transitions tables on the way

        Parameters
        ----------
        train_data: pandas.DataFrame
            training data
        samples_number: int
            number of sequences to generate
        sequence_length: int
            sequence length to generate

        Returns
        -------
        generated_sequences: pandas.DataFrame
            generated sequences
        """

        generated_sequences = [[] for i in range(samples_number)]
        current_state = ['' for i in range(samples_number)]
        next_state = ['' for i in range(samples_number)]

        first_hour = self.get_first_hour(train_data)
        first_hour.to_csv(f'files/mc_first_hour_order_{self.n}.csv')

        for i in tqdm(range(samples_number), desc='first hour'):
            current_state[i] = self.first_state(first_hour)
            generated_sequences[i] = current_state[i].split('_')[-1].split('|')

        for i in tqdm(range(sequence_length - self.n), desc='other states'):
            transitions = self.get_transition_matrix(train_data, i)
            transitions.to_csv(f'files/mc_transitions_order_{self.n}_time_{i}.csv')

            for j in range(samples_number):
                next_state[j] = self.next_state(current_state[j], transitions)
                generated_sequences[j].append(next_state[j].split('_')[-1])
                current_state[j] = self.cols[i + 1] + '_' + '|'.join(
                    current_state[j].split('_')[2].split('|')[1:] + [next_state[j].split('_')[-1]])

        return pd.DataFrame(generated_sequences, columns=self.cols)

    def train(self, train_data):
        """
        Calculate all the transition tables

        Parameters
        ----------
        train_data: pandas.DataFrame
            training data
        """

        first_hour = self.get_first_hour(train_data)
        first_hour.to_csv(f'files/mc_first_hour_order_{self.n}.csv')

        for i in tqdm(range(train_data.shape[1] - self.n), desc='other states'):
            transitions = self.get_transition_matrix(train_data, i)
            transitions.to_csv(f'files/mc_transitions_order_{self.n}_time_{i}.csv')

    def calc_likelihood(self, like_data):
        """
        Calculates the likelihood of sequences

        Parameters
        ----------
        like_data: pandas.DataFrame
            sequences

        Returns
        -------
        like_res: numpy.array
            The likelihood of the sequences
        """

        samples_number = like_data.shape[0]
        sequence_length = like_data.shape[1]
        like_res = np.ones(samples_number)

        try:
            first_hour = pd.read_csv(f'files/mc_first_hour_order_{self.n}.csv', index_col=0)
        except:
            raise("markov chain is not trained")

        current_state = like_data[self.cols[:self.n]].apply(lambda x: self.cols[0] + '_' + '|'.join(x),
                                                            axis=1).to_frame()
        current_state.columns = ['state']
        current_state = current_state.merge(first_hour, on='state', how='left')
        current_state.ind = current_state.ind.fillna(0.001)

        like_res = like_res * current_state.ind.values

        for i in range(sequence_length - self.n):
            transitions = pd.read_csv(f'files/mc_transitions_order_{self.n}_time_{i}.csv')
            current_state = like_data[self.cols[i:i + self.n]].apply(lambda x: self.cols[i] + '_' + '|'.join(x),
                                                                     axis=1).to_frame()
            current_state.columns = ['state']
            current_state['value'] = (self.cols[i + self.n] + '_' + like_data[self.cols[i + self.n]]).values
            current_state = current_state.merge(transitions, on=['state', 'value'], how='left')
            current_state.ind = current_state.ind.fillna(0.001)

            like_res = like_res * current_state.ind.values

        return like_res


def liklihood(synthetic_data, markov_chain_model):
    likelihood_scores = markov_chain_model.calc_likelihood(synthetic_data.astype(str))

    return np.mean(likelihood_scores)


def ws_likelihood(synthetic_data, original_data, markov_chain_model):
    orig_likelihood_score = markov_chain_model.calc_likelihood(original_data.astype(str))
    synth_likelihood_score = markov_chain_model.calc_likelihood(synthetic_data.astype(str))

    return wasserstein_distance(synth_likelihood_score, orig_likelihood_score)

