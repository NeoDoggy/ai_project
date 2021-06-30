from numpy import genfromtxt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pydot
#import files and label
path="./numpydatasets"
apple=np.load(f'{path}/apple.npy')
ay=np.full(len(apple),1)
banana=np.load(f'{path}/banana.npy')
by=np.full(len(banana),2)
blackberry=np.load(f'{path}/blackberry.npy')
blay=np.full(len(blackberry),3)
blueberry=np.load(f'{path}/blueberry.npy')
bluy=np.full(len(blueberry),4)
grapes=np.load(f'{path}/grapes.npy')
gy=np.full(len(grapes),5)
pear=np.load(f'{path}/pear.npy')
py=np.full(len(pear),6)
strawberry=np.load(f'{path}/strawberry.npy')
sty=np.full(len(strawberry),7)
#training dataset
Dx=np.concatenate((apple[:-4000],banana[:-4000]))
Dy=np.concatenate((ay[:-4000],by[:-4000]))
Dx=np.concatenate((Dx,blackberry[:-4000]))
Dy=np.concatenate((Dy,blay[:-4000]))
Dx=np.concatenate((Dx,blueberry[:-4000]))
Dy=np.concatenate((Dy,bluy[:-4000]))
Dx=np.concatenate((Dx,grapes[:-4000]))
Dy=np.concatenate((Dy,gy[:-4000]))
Dx=np.concatenate((Dx,pear[:-4000]))
Dy=np.concatenate((Dy,py[:-4000]))
Dx=np.concatenate((Dx,strawberry[:-4000]))
Dy=np.concatenate((Dy,sty[:-4000]))
#testing dataset
Tx=np.concatenate((apple[-4000:],banana[-4000:]))
Ty=np.concatenate((ay[-4000:],by[-4000:]))
Tx=np.concatenate((Tx,blackberry[-4000:]))
Ty=np.concatenate((Ty,blay[-4000:]))
Tx=np.concatenate((Tx,blueberry[-4000:]))
Ty=np.concatenate((Ty,bluy[-4000:]))
Tx=np.concatenate((Tx,grapes[-4000:]))
Ty=np.concatenate((Ty,gy[-4000:]))
Tx=np.concatenate((Tx,pear[-4000:]))
Ty=np.concatenate((Ty,py[-4000:]))
Tx=np.concatenate((Tx,strawberry[-4000:]))
Ty=np.concatenate((Ty,sty[-4000:]))

np.save('traindataX',Dx)
np.save('traindataY',Dy)
np.save('testdataX',Tx)
np.save('testdataY',Ty)

#print(len(apple)+len(banana)+len(blackberry)+len(blueberry)+len(grapes)+len(pear)+len(strawberry))
