#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% 
import cv2 as cv ;
import matplotlib.pyplot as plt ;
import os ;
os.system('clear') ;
###



# %% 
filename = 'Mass_Spring-{:}.png' ;
filename_out = 'Mass_Spring_dark-{:}.png' ;

for i in range(20):
    I = plt.imread( filename.format( i ) ).copy() ;
    I[:,:,:3] *= 255. ;
    shape = I.shape ;
    # mask = (I[:,:,0] == I[:,:,1]) & (I[:,:,0] == I[:,:,2]) ;
    # values = I[:,:,0][mask] ;
    # values = I.max()-values ;
    # for ii in range(3):
    #     Ii = I[:,:,ii] ;
    #     Ii[mask] = values ;
    # # for i #
    I[:,:,:3] = I.max() - I[:,:,:3] ;
    

    # plt.imshow( I )
    # plt.axis('off')

    # display( I[:10,:10] )

    cv.imwrite( filename_out.format(i), I[:,:,:3] ) ;
    continue ;
# for i #


###


# %% END OF FILE
###
