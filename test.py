from numpy import genfromtxt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pydot
'''
strawberry=np.load('./numpydatasets/strawberry.npy')
img=np.reshape(strawberry[1], (28, 28))
plt.imshow(img)
plt.show()

y=[[1,1],[2,2],[3,3]]
print(y[:-1])
print(y[-3:])

test=np.load('./datasets/traindataX.npy')
print(len(test))
'''
y=[[1,1],[2,2],[3,3]]
print(y[:-1])
print(y[1:])