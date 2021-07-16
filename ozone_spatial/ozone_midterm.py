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

loc = pd.read_csv('C:\\Users\\merre\\Desktop\\jmmerrell.github.io\\ozone_spatial\\PredLocs.csv', header=0)
cmaq = pd.read_csv('C:\\Users\\merre\\Desktop\\jmmerrell.github.io\\ozone_spatial\\CMAQ.csv', header=0)
oz = pd.read_csv('C:\\Users\\merre\\Desktop\\jmmerrell.github.io\\ozone_spatial\\Ozone.csv', header=0)

def min_dist(i):
    dist1= np.hypot(cmaq.Longitude-loc.Longitude[i],cmaq.Latitude-loc.Latitude[i])
    disC= pd.DataFrame({'dist1':dist1, 'CMAQ_O3':cmaq.CMAQ_O3})
    disC = disC.sort_values('dist1')
    disC= np.concatenate((disC['dist1'][:1000],disC['CMAQ_O3'][:1000]),axis=0)
    return(disC)

data = [0.0]*(2685*2000)
data = np.reshape(data,(2685,2000))
data = pd.DataFrame(data=data[0:,0:],index=data[0:,0],columns=data[0,0:])
data = np.concatenate((loc,data),axis=1)
data = pd.DataFrame(data)

for i in range(0,2685):
    r=min_dist(i)
    data.iloc[[i], list(range(2,2002))]=r

print(data.iloc[list(range(0,10)), list(range(0,7))])
