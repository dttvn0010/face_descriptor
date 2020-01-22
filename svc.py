from sklearn.svm import SVC
import numpy as np

Xtrain = np.fromfile('Xtrain.np', dtype='float32').reshape((-1,512))
ytrain = np.fromfile('ytrain.np', dtype='int64')

Xtest = np.fromfile('Xtest.np', dtype='float32').reshape((-1,512))
ytest = np.fromfile('ytest.np', dtype='int64')

model = SVC(kernel='linear')
model.fit(Xtrain, ytrain)
score = model.score(Xtest, ytest)
print('test accuracy = ', score)
