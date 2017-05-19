
# coding: utf-8

import numpy as np
from scipy import ndimage
from scipy.misc import imsave

o3 = ndimage.imread('Kids_DrawA_Otter3.jpg')


o3_not0 = o3_sum(2) <= 254*3

if_crop_d1 = []
for i, v in enumerate(o3_not0.sum(1)):
    if 50 < i < 330:
        if_crop_d1.append(False)
    elif v > 399:
        if_crop_d1.append(True)
    else:
        if_crop_d1.append(False)



o3_cropped = o3[np.invert(np.array(if_crop_d1)), :, :]

m = np.min(o3_cropped, 2)
alpha = 1 - m/255

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.floor_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

new_rgb = div0((o3_cropped - m[:,:,np.newaxis]), alpha[:,:,np.newaxis])

o3_trans = np.dstack((new_rgb, 255*alpha))

imsave('o3.png', o3_trans)