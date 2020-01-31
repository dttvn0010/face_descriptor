import os
import numpy as np
from random import shuffle
import shutil

EMB_DIRS = 'embeddings'
Xtrain = []
ytrain = []
Xtest = []
ytest = []
labels0 = list(os.listdir(EMB_DIRS))
shuffle(labels0)

labels1 = labels0[:len(labels0)//2]
labels2 = labels0[len(labels0)//2:]
label_map = {label:i for i, label in enumerate(labels1)}
paths_test = []

for p in labels1:
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

Xver = []
for p in labels2:
    files = os.listdir(os.path.join(EMB_DIRS, p))
    for f in files:
        emb_path = os.path.join(EMB_DIRS, p, f)
        Xver.append(np.fromfile(emb_path, dtype='float32'))

np.array(Xtrain).tofile('Xtrain.np')
np.array(ytrain).tofile('ytrain.np')
np.array(Xtest).tofile('Xtest.np')
np.array(ytest).tofile('ytest.np')
np.array(Xver).tofile('Xver.np')

with open('labels.txt', 'w') as f:
    f.write(','.join(labels1))
    
with open('paths_test.txt', 'w') as f:
    for path in paths_test:
        f.write(path + '\n')    
