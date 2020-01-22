from insight_face import face_model
import cv2
import sys
import numpy as np

model = face_model.FaceModel()
img = cv2.imread(sys.argv[1])
f1 = model.getFaceFeatures(img)

img = cv2.imread(sys.argv[2])
f2 = model.getFaceFeatures(img)

dist = np.sum(np.square(f1-f2))
print('Euclide diatance:', dist)
sim = np.dot(f1, f2.T)
print('Cosine simarlity:', sim)
