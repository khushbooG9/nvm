import numpy as np
from tensor import *

H_20 = np.array([
    [+1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
    [+1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1], 
    [+1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1], 
    [+1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1], 
    [+1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1], 
    [+1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1], 
    [+1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1], 
    [+1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1], 
    [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1], 
    [+1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1], 
    [+1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1], 
    [+1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1], 
    [+1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1, -1], 
    [+1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -1], 
    [+1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1], 
    [+1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1], 
    [+1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, +1], 
    [+1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1], 
    [+1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, -1], 
    [+1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1]])
H_20 = totensor(H_20)

H_12=np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1, 1],
               [1,-1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1],
               [1, 1,-1,-1,-1, 1, 1, 1,-1, 1,-1,-1],
               [1,-1,-1,-1, 1, 1, 1,-1, 1,-1,-1, 1],
               [1,-1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1],
               [1,-1, 1, 1, 1,-1, 1,-1,-1, 1,-1,-1],
               [1, 1, 1, 1,-1, 1,-1,-1, 1,-1,-1,-1],
               [1, 1, 1,-1, 1,-1,-1, 1,-1,-1,-1, 1],
               [1, 1,-1, 1,-1,-1, 1,-1,-1,-1, 1, 1],
               [1,-1, 1,-1,-1, 1,-1,-1,-1, 1, 1, 1],
               [1, 1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1]])
H_12 = totensor(H_12)

H_4=np.array([[1, 1, 1, 1],
              [1,-1, 1,-1],
              [1, 1,-1,-1],
              [1,-1,-1, 1]
              ])
H_4 = totensor(H_4)

