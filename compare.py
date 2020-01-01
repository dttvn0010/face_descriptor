import cv2
import sys
import numpy as np
from face_descriptor import FaceDescriptor

fd = FaceDescriptor()

def readImage(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

f11,f12 = fd.getFaceFeatures(readImage(sys.argv[1]))
f21,f22 = fd.getFaceFeatures(readImage(sys.argv[2]))

print('Euclide diatance:', np.sum(np.square(f11-f21)), np.sum(np.square(f12-f22)))
print('Cosine simarlity:', np.dot(f11, f21.T), np.dot(f12, f22.T))
