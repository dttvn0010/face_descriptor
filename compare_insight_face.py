import cv2
import sys
import numpy as np
from insight_face.face_model import FaceModel

model = FaceModel()

f1 = model.getFaceFeatures(cv2.imread(sys.argv[1]))
f1 = model.getFaceFeatures(cv2.imread(sys.argv[2]))

print('Euclide diatance:', np.sum(np.square(f1-f2)))
print('Cosine simarlity:', np.dot(v1,v2))