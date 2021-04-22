import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

dataset = pd.read_csv('admissions_data.csv')

# print(dataset.head())

feature = dataset.iloc[:,:-1]

labels = dataset.iloc[:,-1]

# print(feature)
# print(labels)

features_train, features_test, labels_train, labels_test = train_test_split(feature, labels, test_size=0.25)

stand_scaler = StandardScaler()

# print(features_test)

# import sys
# sys.exit()
# appears to be necessary to scale features_test that we first reshape
# StandardScaler.transform() expects multidimensional array
# features_test = np.array(features_test.values.tolist()).reshape(-1,1)

standardized_features_train = stand_scaler.fit_transform(features_train)
standardized_features_test = stand_scaler.transform(features_test)

standardized_features_train = pd.DataFrame(standardized_features_train)
standardized_features_test = pd.DataFrame(standardized_features_test)

# print(standardized_features_test.head())

# Designing the model function itself


def design_model(data, learning_rate):  # our function to design the model
    model = Sequential(name="my_first_model")
    new_input = tf.keras.Input(shape=(data.shape[1],))
    model.add(new_input)
    model.add(layers.Dense(128, activation='relu', name='first_layer'))  # addinf one hidden layer
    model.add(layers.Dense(64, activation='relu', name='second_layer'))
    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(
        learning_rate=learning_rate)  # setting the learning rate of Adam to the one specified in the function parameter
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model


learning_rate = 0.001
my_model = design_model(standardized_features_train, learning_rate)

my_model.fit(standardized_features_train, labels_train, batch_size=4, epochs=10, verbose=2)

my_model.evaluate(standardized_features_test, labels_test, batch_size=4, verbose=2)

predicted_values = my_model.predict(standardized_features_test)
print(r2_score(labels_test, predicted_values))