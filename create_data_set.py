import cv2
import sys
import os
import numpy as np
from random import shuffle
import shutil

#from insight_face.face_model import FaceModel
from facenet.face_model import FaceModel

DATA_DIR = 'face_db'
EMB_DIRS = 'embeddings'
model = FaceModel()

shutil.remove(EMB_DIRS)
os.mkdir(EMB_DIRS)
    
for p in os.listdir(DATA_DIR):
    print(p)
    os.mkdir(os.path.join(EMB_DIRS, p))
    for f in os.listdir(os.path.join(DATA_DIR, p)):
        img = cv2.imread(os.path.join(DATA_DIR, p, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features = model.getFaceFeatures(img)
        
        if features is not None:
            features.tofile(os.path.join(EMB_DIRS, p, f.lower().replace('.jpg', '.np')))
            

Xtrain = []
ytrain = []
Xtest = []
ytest = []
labels = list(os.listdir(EMB_DIRS))
label_map = {label:i for i, label in enumerate(labels)}
paths_test = []

for p in os.listdir(EMB_DIRS):
    files = os.listdir(os.path.join(EMB_DIRS, p))
    shuffle(files)
    ntrain = min(20, len(files)//2)
    
    for f in files[:ntrain]:
        emb_path = os.path.join(EMB_DIRS, p, f)
        Xtrain.append(np.fromfile(emb_path, dtype='float32'))
        ytrain.append(label_map[p])
    
    for f in files[ntrain:]:
        emb_path = os.path.join(EMB_DIRS, p, f)
        Xtest.append(np.fromfile(emb_path, dtype='float32'))
        ytest.append(label_map[p])  
        paths_test.append(emb_path)
    
Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)
Xtest = np.array(Xtest)
ytest = np.array(ytest)

Xtrain.tofile('Xtrain.np')
ytrain.tofile('ytrain.np')
Xtest.tofile('Xtest.np')
ytest.tofile('ytest.np')

with open('labels.txt', 'w') as f:
    f.write(','.join(labels))
    
with open('paths_test.txt', 'w') as f:
    for path in paths_test:
        f.write(path + '\n')            
       