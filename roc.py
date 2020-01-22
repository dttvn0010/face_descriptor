from sklearn.svm import SVC
import numpy as np

Xtrain = np.fromfile('Xtrain.np', dtype='float32').reshape((-1,512))
ytrain = np.fromfile('ytrain.np', dtype='int32')

Xtest = np.fromfile('Xtest.np', dtype='float32').reshape((-1,512))
ytest = np.fromfile('ytest.np', dtype='int32')

Xtrain = [x/np.linalg.norm(x) for x in Xtrain]
Xtest = [x/np.linalg.norm(x) for x in Xtest]

def getCosineDist(v1, v2):
    return np.dot(v1,v2)

def calcAcc(thresh):
    false_neg = 0
    false_pos = 0
    Ntrain = len(Xtrain)
    Ntest = len(Xtest)
    for (x,y) in zip(Xtest, ytest):
        scores = [(getCosineDist(x, Xtrain[i]), i) for i in range(Ntrain)]
        scores = sorted(scores, reverse=True)
        score,i = scores[0]
        if score < thresh:
            false_neg += 1
        elif ytrain[i] != y:
            false_pos += 1
    return (1-false_neg/Ntest), (1-false_pos/Ntest)
    
print(calcAcc(0.775))    
            
        
        