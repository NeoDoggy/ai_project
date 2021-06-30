from numpy import genfromtxt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pydot

model=keras.models.load_model('fruit')

#making test dataset
path="./numpydatasets"
apple=np.load(f'{path}/apple.npy')
banana=np.load(f'{path}/banana.npy')
blackberry=np.load(f'{path}/blackberry.npy')
blueberry=np.load(f'{path}/blueberry.npy')
grapes=np.load(f'{path}/grapes.npy')
pear=np.load(f'{path}/pear.npy')
strawberry=np.load(f'{path}/strawberry.npy')
'''
Tx=np.concatenate((apple[-10:],banana[-10:]))
Tx=np.concatenate((Tx,blackberry[-10:]))
Tx=np.concatenate((Tx,blueberry[-10:]))
Tx=np.concatenate((Tx,grapes[-10:]))
Tx=np.concatenate((Tx,pear[-10:]))
Tx=np.concatenate((Tx,strawberry[-10:]))
'''
Tx=strawberry[-100:]
Tx=Tx/255
prediction=model.predict_classes(Tx)
print(prediction)