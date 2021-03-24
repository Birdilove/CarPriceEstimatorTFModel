# -*- coding: utf-8 -*-
"""Suggesting Used Car Prices — a case of multi-linear regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cwuhfEfjKEneSZbAYv5y7bgWBCwGWkWR

##### Copyright 2020 Olayinka Peter.
"""

# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""## Suggesting Car Prices — a case of regreshun

###Problem
You've got a used car you'd like to sell. You know the acquired price, but how do you measure how much a fair sale price would be based on how much it has been used?

---

###Dataset description

I make use of a particular used cars dataset and thier prices (https://www.kaggle.com/avikasliwal/used-cars-price-prediction) which has the following columns: 

Name: The brand and model of the car.

Location: The location in which the car is being sold or is available for purchase.

Year: The year or edition of the model.

Kilometers_Driven: The total kilometres driven in the car by the previous owner(s) in KM.

Fuel_Type: The type of fuel used by the car. (Petrol, Diesel, Electric, CNG, LPG)

Transmission: The type of transmission used by the car. (Automatic / Manual)

Owner_Type: 

Mileage: The standard mileage offered by the car company in kmpl or km/kg

Engine: The displacement volume.

---
"""

# Use seaborn for pairplot
# !pip install seaborn

# Use some functions from tensorflow_docs
# !pip install git+https://github.com/tensorflow/docs

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

print(tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

"""### Get the data

"""

"""Import it using pandas"""

column_names = ['Price','year','company','name','kms_driven']
raw_dataset = pd.read_csv("FINAL_DATASET.csv", names=column_names,
                          na_values="?", comment='\t', skiprows=1,
                          sep=",", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.head()

dataset.describe()

"""### Clean the data

The dataset contains a few unknown values.
"""

dataset.isna().sum()

"""I drop those rows."""

dataset = dataset.dropna()

dataset = dataset.reset_index(drop=True)
dataset.head()

dataset['kms_driven'] = pd.Series([float(str(val)) for val in dataset['kms_driven']], index=dataset.index)
dataset['Price'] = pd.Series([float(str(val)) for val in dataset['Price']], index=dataset.index)
dataset['year'] = pd.Series([float(val) for val in dataset['year']], index=dataset.index)

"""## One-hot Encoding for categorical data"""

print(dataset['company'].unique())
dataset['company'] = pd.Categorical(dataset['company'])
dfFuel_Type = pd.get_dummies(dataset['company'], prefix='company')

dfFuel_Type.head()

print(dataset['name'].unique())
dataset['name'] = pd.Categorical(dataset['name'])
dfTransmission = pd.get_dummies(dataset['name'], prefix='name')

dfTransmission.head()

dataset = pd.concat([dataset, dfFuel_Type, dfTransmission], axis=1)
dataset.head()

dataset = dataset.drop(columns=['company', 'name'])
dataset.head()

"""###Splitting

I split the dataset into a training set and a test set.

And use the test set in the final evaluation of the model.
"""

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

"""### Inspecting the data

Using sns, I try to have a quick look at the joint distribution of a few pairs of columns from the training set.
"""

#sns.pairplot(train_dataset[["Price", "year", "Engine", "Power"]], diag_kind="kde")

"""### Split features from labels

I now separate the target value, or "label", from the features.
"""

train_labels = train_dataset.pop('Price')
test_labels = test_dataset.pop('Price')

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
train_stats

"""**Also** at the overall statistics:"""

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
train_stats

train_dataset.head()

"""**bold text**### Normalize the data


"""


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

normed_train_data.head()

"""This normalized data is what we will use to train the model.

### Build the model

Here, I use a `Sequential` model with eight multiple connected hidden layers, and an output layer that returns a single, continuous value.
"""


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=[len(train_dataset.keys())]),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()

model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

print("here")
print(normed_test_data[5:6])
"""Train the model

I now train the model for 1000 epochs, and record the training and validation accuracy in the `history` object.
"""

EPOCHS = 20

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()])

"""
**Visualize** the model's training progress using the stats stored in the `history` object."""

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Basic': history}, metric="mae")
plt.ylim([0, 5000])
plt.ylabel('MAE [Price]')

plotter.plot({'Basic': history}, metric="mse")
plt.ylim([0, 10000000])
plt.ylabel('MSE [Price^2]')

example_batch = normed_train_data[:5]
example_batch

example_batch = normed_test_data[5:6]
example_batch

example_result = model.predict(example_batch)
print(example_result)

"""### Make predictions

I finally predict the Price values using data in the testing set:
"""

test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Price]')
plt.ylabel('Predictions [Price]')
lims = [0, 100000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

import time

t = time.time()

export_path_sm = "./{}".format(int(t))
print(export_path_sm)

tf.saved_model.save(model, export_path_sm)

# !ls {export_path_sm}

import tensorflow_hub as hub

reload_saved_model = tf.keras.models.load_model(
    export_path_sm,
    custom_objects={'KerasLayer': hub.KerasLayer})

reload_saved_model.summary()

DEFAULT_FUNCTION_KEY = "serving_default"
loaded = tf.saved_model.load(export_path_sm)
inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]

loaded

inference_func

# example_result = inference_func(example_batch)

example_tf_batch = tf.convert_to_tensor(example_batch, dtype='float32')

example_result = inference_func(example_tf_batch)
# for batch in example_batch.take(1):
#   print(inference_func(batch))

example_result

result_ = example_result['dense_8']
print(result_)
result_

result_.numpy()

reload_saved_model.predict(example_batch)

"""Let's take a look at the error distribution.

"""

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [Price]")
_ = plt.ylabel("Count")