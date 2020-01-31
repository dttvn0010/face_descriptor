from sklearn.svm import SVC
import numpy as np

Xtrain = np.fromfile('Xtrain.np', dtype='float32').reshape((-1,512))
ytrain = np.fromfile('ytrain.np', dtype='int64')

Xtest = np.fromfile('Xtest.np', dtype='float32').reshape((-1,512))
ytest = np.fromfile('ytest.np', dtype='int64')

Xver = np.fromfile('Xver.np', dtype='float32').reshape((-1,512))

def getCosineDist(v1, v2):
    return np.dot(v1,v2)

def calcAcc(thresh):
    false_neg = 0
    false_pos = 0
    Ntrain = len(Xtrain)
    Ntest = len(Xtest)
    Nver = len(Xver)
    for (x,y) in zip(Xtest, ytest):
        scores = [(getCosineDist(x, Xtrain[i]), i) for i in range(Ntrain)]
        scores = sorted(scores, reverse=True)
        score,i = scores[0]
        if score < thresh or ytrain[i] != y:
            false_neg += 1
    
    for x in Xver:
        scores = [getCosineDist(x, Xtrain[i]) for i in range(Ntrain)]
        score = np.max(scores)
        if score >= thresh:
            false_pos += 1
        
    return (1-false_neg/Ntest), (1-false_pos/Nver)
    
print(calcAcc(0.65))  # insight_face~0.65, facenet~0.84-0.085
