from numpy import genfromtxt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pydot

Dx=np.load('./datasets/traindataX.npy')
Dy=np.load('./datasets/traindataY.npy')
Tx=np.load('./datasets/testdataX.npy')
Ty=np.load('./datasets/testdataY.npy')

Dx=Dx/255
Tx=Tx/255

Dy=np_utils.to_categorical(Dy,8)
Ty=np_utils.to_categorical(Ty,8)

model=Sequential()
model.add(Dense(input_dim=28*28,units=256,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=8,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=Dx,
                        y=Dy,
                        validation_split=0.2,
                        epochs=50,
                        batch_size=600,
                        verbose=2)
#plt.plot(train_history.history['loss'])
plt.plot(train_history.history['accuracy'])
plt.show()
model.evaluate(Tx,Ty,batch_size=50)
prediction=model.predict_classes(Tx)
print(prediction[:10])