import cv2
import sys
import numpy as np
from face_descriptor import FaceDescriptor

fd = FaceDescriptor()

def readImage(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def getCosineDist(v1, v2):
    return np.dot(v1,v2)

f1 = fd.getFaceFeatures(readImage(sys.argv[1]))
f2 = fd.getFaceFeatures(readImage(sys.argv[2]))

print('Euclide diatance:', np.sum(np.square(f1-f2)))
print('Cosine simarlity:', getCosineDist(f1, f2))
