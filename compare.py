import cv2
import sys
import numpy as np
from face_descriptor import FaceDescriptor

fd = FaceDescriptor()

def readImage(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def getCosineDist(v1, v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))

f11,f12 = fd.getFaceFeatures(readImage(sys.argv[1]))
f21,f22 = fd.getFaceFeatures(readImage(sys.argv[2]))

print('Euclide diatance:', np.sum(np.square(f11-f21)), np.sum(np.square(f12-f22)))
print('Cosine simarlity:', getCosineDist(f11, f21), getCosineDist(f12, f22))
