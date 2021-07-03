from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pydot
import os
import cv2
from PIL import Image
import random


dir='./dataimg'

for f in os.scandir(dir):
    if (f.path.endswith(".png")) and f.is_file():
        #print(f.path)
        fn=f.name
        #print(f'30{fn}')
        fi=Image.open(f.path)
        r = fi.rotate(60)
        r.save(f'./img/60{fn}')
        r = fi.rotate(90)
        r.save(f'./img/90{fn}')
        r = fi.rotate(120)
        r.save(f'./img/120{fn}')
        r = fi.rotate(180)
        r.save(f'./img/180{fn}')
        r = fi.rotate(240)
        r.save(f'./img/240{fn}')
        r = fi.rotate(300)
        r.save(f'./img/300{fn}')
        for i in range(100):
            rnd=random.randint(0, 360)
            r = fi.rotate(rnd)
            r.save(f'./img/{rnd}{fn}')

dir='./img'

meow=np.full(784,0)
meow=np.reshape(meow,(1,len(meow)))
for f in os.scandir(dir):
    if (f.path.endswith(".png")) and f.is_file():
        im=cv2.imread(f.path,cv2.IMREAD_GRAYSCALE)
        im=cv2.resize(im,(28,28),interpolation=cv2.INTER_NEAREST)
        im=~im
        im=im.flatten()
        nf=np.reshape(im,(1,len(im)))
        meow=np.concatenate((meow,nf))

meow=meow[1:]
np.save('data',meow)

