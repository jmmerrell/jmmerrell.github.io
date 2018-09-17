from __future__ import print_function
import pandas as pd
import random
import math
from scipy.linalg import toeplitz
import statsmodels.api as sm
from statsmodels.formula.api import ols
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)
df= pd.read_csv('C:\\Users\\merre\\Desktop\\jmmerrell.github.io\\movie_random_forest\\movie_data.txt', sep="|", header=0)
df.columns = ["date","movie","studio","genre","basedon","actionanim","factfict","budget","thrcount","threngag","fisrtweeknd","domgross","infdomgross","intgross","totgross","direct","compose","act1","act2","act3","act4","act5","act6","act7","act8","act9","act10","act11","act12","act13","act14","act15","act16","act17","act18","act19","act20","rating","franch"]
df['date'] = pd.to_datetime(df['date'])
cols = df.columns[list(range(7,15))]
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
###order data by date
df = df.sort_values(by=['date'], ascending=True)
###Drop all movies with no inflation adjusted gross and without actors
df= df.loc[(df['infdomgross'] > 0) & (df['act1'].isnull() == False)]
###Create inflation adjusted budget
df['infbudget'] = df['infdomgross']/df['domgross']*df['budget']
####Create a new dataframe with only new movies
df2 = df.loc[df['date']>='2000-01-01']

print(df2)
# df['Date'] = pd.to_datetime(df['Date'])
# df['month'] = df['Date'].dt.month
# df['winter'] = np.where(df['month'].isin([12,1,2,3]) , 1, 0)
# df['summer']= np.where(df['month'].isin([7,8,9,10]), 1, 0)
# plt.plot(df.Date, df.PowerBill)
# plt.show()
# temps = np.array([32,31,36,44.5,52,61,69.5,77.5,76,66.5,53.5,41.5])
# temps = abs(temps-65)
# temps = [temps]*4
# temps = np.concatenate(temps)
# temps = temps.tolist()
# temps2 = temps[:3]
# temps = temps+temps2
# df['temp'] = temps
# df['solar_winter'] = np.where((df['Solar']=='Y')&(df['winter']==1) , 1, 0)
# df['solar_summer'] = np.where((df['Solar']=='Y')&(df['summer']==1) , 1, 0)
# df['summer_temp'] = df['temp']*df['summer']
# df['winter_temp'] = df['temp']*df['winter']

# nsims=100
# out = [0.0]*(nsims*53)
# out = np.reshape(out,(nsims,53))
#
# for i in range(0,nsims):
#     rowz = np.random.choice(df.shape[0], 5, replace=False)
#     train = df.ix[set(range(1, df.shape[0])).difference(rowz)]
#     test = df.ix[rowz]
#     ols_resid = sm.OLS.from_formula('PowerBill ~ C(Solar) + C(solar_winter) + C(solar_summer) + summer_temp + winter_temp', data=df).fit().resid
#     resid_fit = sm.OLS(endog=list(ols_resid[1:]), exog=sm.add_constant(ols_resid[:-1])).fit()
#     rho = resid_fit.params[1]
#     toeplitz(range(5))
#     order = toeplitz(range(train.shape[0]))
#     sigma = rho**order
#     gls_model = sm.GLS.from_formula('PowerBill ~ C(Solar) + C(solar_winter) + C(solar_summer) + summer_temp + winter_temp', data=train, sigma=sigma)
#     gls_results = gls_model.fit()
#     preds=gls_results.predict(test)
#     out[i][0]=np.mean(test['PowerBill']-preds)
#     out[i][1]=math.sqrt(np.mean((test['PowerBill']-preds)**2))
#     out[i][(rowz+1)]=preds
#
# def column(matrix, i):
#     return [row[i] for row in matrix]
# print(np.mean(column(out,0)))
# print(np.mean(column(out,1)))
