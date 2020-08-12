import pandas as pd
import numpy as np
#import matplotlib.pyplot as mpl
from sklearn import linear_model  as lm

data = pd.read_csv('FCC.csv')
#print(data.head())
data = data[['ENGINESIZE','CO2EMISSIONS']]
#print(data.head())
train = data[:(int((len(data)*0.8)))]
testdata = data[(int((len(data)*0.8))):]

rgr = lm.LinearRegression()
trainTIKTOK = np.array(train[['ENGINESIZE']])
trainINSTAGRAM = np.array(train[['CO2EMISSIONS']])

rgr.fit(trainTIKTOK,trainINSTAGRAM)
#.fit applies model to data

print(rgr.coef_)
print(rgr.intercept_)