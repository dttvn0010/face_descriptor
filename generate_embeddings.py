import cv2
import sys
import os
import numpy as np
from random import shuffle
import shutil

from insight_face.face_model import FaceModel
#from facenet.face_model import FaceModel

DATA_DIR = 'face_db'
EMB_DIRS = 'embeddings'
model = FaceModel()

if os.path.exists(EMB_DIRS):
    shutil.rmtree(EMB_DIRS)
    
os.mkdir(EMB_DIRS)
    
for p in os.listdir(DATA_DIR):
    print(p)
    os.mkdir(os.path.join(EMB_DIRS, p))
    for f in os.listdir(os.path.join(DATA_DIR, p)):
        img = cv2.imread(os.path.join(DATA_DIR, p, f))
        features = model.getFaceFeatures(img)
        
        if features is not None:
            features.tofile(os.path.join(EMB_DIRS, p, f.lower().replace('.jpg', '.np')))
