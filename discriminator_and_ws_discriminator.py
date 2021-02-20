import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.stats import wasserstein_distance

from keras.layers import Input, Concatenate, Reshape, Embedding
from keras.models import Model
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split

class discriminator_model():

    def create_training_data(self, original_training_data, random_data_size):

        self.data_cols = original_training_data.columns
        y_train = np.ones(original_training_data.shape[0])
        self.locations = sorted(np.unique(original_training_data.values))

        location_time_stat = (
            pd.melt(original_training_data)
            .reset_index()
            .groupby(['variable', 'value'])
            .count().reset_index()
        )
        location_time_stat['index'] = location_time_stat.groupby('variable')['index'].transform(lambda x: x/sum(x))
        location_time_stat.set_index('variable', inplace=True)

        X_train_random = np.ones((random_data_size, original_training_data.shape[1])).astype(str)

        for i in tqdm(range(random_data_size)):
            for j in range(original_training_data.shape[1]):
                X_train_random[i, j] = int(
                    np.random.choice(
                        location_time_stat.loc[self.data_cols[j]]['value'],
                        p=location_time_stat.loc[self.data_cols[j]]['index']
                ))

        X_train_random = pd.DataFrame(X_train_random, columns=self.data_cols)
        y_train_random = np.zeros(random_data_size)

        self.X_train = pd.concat([original_training_data, X_train_random])
        self.Y_train = pd.Series(np.concatenate([y_train, y_train_random]))

        for col in tqdm(self.data_cols):
            self.X_train[col] = self.X_train[col].apply(lambda x: self.locations.index(x))

    def model(self):

        inputs = []
        embeddings = []
        for i in range(self.X_train.shape[1]):
            feature_input = Input(shape=(1,), name=self.data_cols[i])
            feature_embedding = Embedding(input_dim=len(self.locations), output_dim=8,
                                          name="{}-embeddings".format(self.data_cols[i]))(feature_input)
            feature_embedding = Reshape(target_shape=(8,))(feature_embedding)

            inputs.append(feature_input)
            embeddings.append(feature_embedding)

        layers = Concatenate(axis=-1)(embeddings)

        layers = Dense(256)(layers)
        layers = LeakyReLU()(layers)

        layers = Dense(128)(layers)
        layers = LeakyReLU()(layers)

        layers = Dense(64)(layers)
        layers = LeakyReLU()(layers)

        output = Dense(1, activation="sigmoid")(layers)

        return Model(inputs=inputs, output=output)

    def train_model(self, original_training_data, random_data_size, batch_size, epochs, random_state=0):

        self.create_training_data(original_training_data, random_data_size)
        self.discriminator = self.model()
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        x_train, x_test, y_train, y_test = train_test_split(
            self.X_train,
            self.Y_train,
            test_size=0.2,
            random_state=random_state
        )

        x_train_disc = {}
        for col in x_train.columns:
            x_train_disc[col] = x_train[col].values.astype(object)

        x_test_disc = {}
        for col in x_test.columns:
            x_test_disc[col] = x_test[col].values.astype(object)

        self.history = self.discriminator.fit(x_train_disc, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_split=0.1
        )
        print('val_accuracy', self.history.history['val_accuracy'])

        y_test_predicted = self.discriminator.predict(x_test_disc)
        y_test_predicted = pd.Series(np.reshape(y_test_predicted, y_test_predicted.shape[0]) > 0.5).astype(int)
        test_accuracy = accuracy_score(y_test, y_test_predicted)
        print('test_accuracy', test_accuracy)


def discrimiator_score(discriminator_model, synthetic_data):
    synthetic_data_disc = {}
    for col in synthetic_data.columns:
        synthetic_data_disc[col] = synthetic_data[col].apply(
            lambda x: discriminator_model.locations.index(x)).values.astype(object)

    disc_score = discriminator_model.discriminator.predict(synthetic_data_disc)

    return np.mean(disc_score)


def ws_discrimiator_score(discriminator_model, synthetic_data, original_data):
    original_data_disc = {}
    for col in original_data.columns:
        original_data_disc[col] = original_data[col].apply(
            lambda x: discriminator_model.locations.index(x)).values.astype(object)

    synthetic_data_disc = {}
    for col in synthetic_data.columns:
        synthetic_data_disc[col] = synthetic_data[col].apply(
            lambda x: discriminator_model.locations.index(x)).values.astype(object)

    orig_score = discriminator_model.discriminator.predict(synthetic_data_disc)
    synth_score = discriminator_model.discriminator.predict(synthetic_data_disc)

    orig_score = np.reshape(orig_score, len(orig_score))
    synth_score = np.reshape(synth_score, len(synth_score))

    return wasserstein_distance(synth_score, orig_score)