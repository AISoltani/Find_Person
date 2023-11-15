import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading image
p = cv2.imread('findHim.jpg')

# Split Channels
b = p[:, :, 0]
g = p[:, :, 1]
r = p[:, :, 2]

minb = min(b.ravel())
ming = min(g.ravel())
minr = min(r.ravel())
maxb = max(b.ravel())
maxg = max(g.ravel())
maxr = max(r.ravel())

for i in range(p.shape[0]):
    for j in range(p.shape[1]):
        b[i,j] = ((b[i,j]-minb)*255 / (maxb-minb))
        g[i,j] = ((g[i,j]-ming)*255 / (maxg-ming))
        r[i,j] = ((r[i,j]-minr)*255 / (maxr-minr))
        
q = p.copy()
q[:, :, 0] = b
q[:, :, 1] = g
q[:, :, 2] = r

#cv2.imshow('r',r)
#cv2.imshow('g',g)
#cv2.imshow('b',b)

print(q.shape)

dst = cv2.fastNlMeansDenoising(q,None,10,7,21)
cv2.imshow('dst',dst)
cv2.imshow('q',q)
plt.show()

