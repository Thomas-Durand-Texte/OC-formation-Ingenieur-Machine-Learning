#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% 
import cv2 as cv ;
import matplotlib.pyplot as plt ;
import os ;
os.system('clear') ;
###



# %% 
I = plt.imread('OC.jpeg').copy() ;
shape = I.shape ;
mask = (I[:,:,0] == I[:,:,1]) & (I[:,:,0] == I[:,:,2]) ;
values = I[:,:,0][mask] ;
values = 255-values ;
for i in range(3):
    Ii = I[:,:,i] ;
    Ii[mask] = values ;
# for i #

cv.imwrite( 'OC_black.jpeg', I ) ;

###


# %% END OF FILE
###
