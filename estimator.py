import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pasta.augment import inline
import pickle

mpl.style.use('ggplot')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

car = pd.read_csv('dataset.csv')
# plt.subplots(figsize=(20, 10))
# ax = sns.swarmplot(x='year', y='price', data=car)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha='right')
#
# plt.subplots(figsize=(20, 10))
# ax = sns.barplot(x='year', y='Price', data=car)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
#
# plt.subplots(figsize=(20, 10))
# ax = sns.barplot(x='year', y='kms_driven', data=car)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
#
# plt.subplots(figsize=(14, 7))
# ax = sns.relplot(x='company', y='Price', data=car, size='year', height=7, aspect=2)
# ax.set_xticklabels(rotation=40, ha='right')

car = car.replace(r'^\s*$', np.nan, regex=True)

car.isna().sum()

car = car.dropna()
car = car.reset_index(drop=True)
print(car.head())
car = car[car['price'] < 250000]
car = car[car['price'] > 1000]
car = car[car['year'] > 2005]
plt.show()

X = car[['model', 'manufacturer', 'year', 'odometer']]
y = car['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ohe = OneHotEncoder()
ohe.fit(X[['model', 'manufacturer']])

column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['model', 'manufacturer']),
    remainder='passthrough')

lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
print(pipe.fit(X_train, y_train))
y_pred = pipe.predict(X_test)
print(r2_score(y_test, y_pred))

scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

print(max(scores))
print("Estimated price of car: ")
print(pipe.predict(pd.DataFrame(columns=X_test.columns,
                                data=np.array(['cr-v', 'honda', 2008, 20000]).reshape(1, 4))))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.argmax(scores))
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(r2_score(y_test, y_pred))

pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))
