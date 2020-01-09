import cv2
import sys
import os
import numpy as np
from random import shuffle

from face_descriptor import FaceDescriptor

fd = FaceDescriptor()

if not os.path.exist('embeddings'):
    os.mkdir('embeddings')
    
for p in os.listdir('face_db'):
    print(p)
    os.mkdir('embeddings/' + p)
    for f in os.listdir(os.path.join('face_db', p)):
        img = cv2.imread(os.path.join('face_db', p, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features = fd.getFaceFeatures(img)
        
        if features is not None:
            features.tofile(os.path.join('embeddings', p, f+'.np'))
            

Xtrain = []
ytrain = []
Xtest = []
ytest = []
labels = list(os.listdir('embeddings'))
label_map = {label:i for i,label in enumerate(labels)}
paths_test = []

for p in os.listdir('embeddings'):
    files = os.listdir(os.path.join('embeddings',p))
    shuffle(files)
    ntrain = min(20, len(files)//2)
    #
    for f in files[:ntrain]:
        emb_path = os.path.join('embeddings', p, f)
        Xtrain.append(np.fromfile(emb_path, dtype='float32'))
        ytrain.append(label_map[p])
    #
    for f in files[ntrain:]:
        emb_path = os.path.join('embeddings', p, f)
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
       