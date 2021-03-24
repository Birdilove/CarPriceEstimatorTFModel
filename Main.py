from flask import Flask, render_template, request, redirect
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('dataset.csv')

company = 'audi'
car_model = 'a4'
year = 2015
driven = 20000

prediction = model.predict(pd.DataFrame(columns=['model', 'manufacturer', 'year', 'odometer'],
                                        data=np.array([car_model, company, year, driven]).reshape(1, 4)))
print(prediction)

