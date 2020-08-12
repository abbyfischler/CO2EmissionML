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

def future(data1,intercept,slope):
  return data1 * slope + intercept
enginesizevar1 = 3.3 #liters
ee = future(enginesizevar1,rgr.intercept_[0],rgr.coef_[0][0])
print(ee)

#check for acccuracy
from sklearn.metrics import r2_score as rs
testTIKTOK = np.array(testdata[['ENGINESIZE']])
testINSTAGRAM = np.array(testdata[['CO2EMISSIONS']])

ra = rgr.predict(testTIKTOK)

absolute = np.mean(np.absolute(ra - testINSTAGRAM))

r2score= rs(ra, testINSTAGRAM)
print(absolute)
print(r2score)


