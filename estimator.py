import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pasta.augment import inline

mpl.style.use('ggplot')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

car = pd.read_csv('Vehicles_Dataset_Clean.csv')
plt.subplots(figsize=(20, 10))
ax = sns.barplot(x='company', y='Price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha='right')
plt.show()

# plt.subplots(figsize=(20, 10))
# ax = sns.swarmplot(x='year', y='Price', data=car)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
# plt.show()

sns.relplot(x='kms_driven', y='Price', data=car, height=7, aspect=1.5)

plt.subplots(figsize=(14, 7))
sns.boxplot(y='Price', data=car)

ax = sns.relplot(x='company', y='Price', data=car, size='year', height=7, aspect=2)
ax.set_xticklabels(rotation=40, ha='right')

X = car[['name', 'company', 'year', 'kms_driven']]
y = car['Price']
X
y.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ohe = OneHotEncoder()
ohe.fit(X[['name', 'company']])

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['name', 'company']),
                                       remainder='passthrough')

lr = LinearRegression()

pipe = make_pipeline(column_trans, lr)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

r2_score(y_test, y_pred)

scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

print(np.argmax(scores))

scores[np.argmax(scores)]

print(pipe.predict(pd.DataFrame(columns=X_test.columns,
                                data=np.array(['q7', 'bmw', 2020, 40000]).reshape(1, 4))))
