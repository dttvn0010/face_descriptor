import numpy as np
import mxnet as mx
import cv2
import sklearn
from sklearn.decomposition import PCA
from .mtcnn_detector import MtcnnDetector
from . import face_preprocess

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self):
    ctx = mx.cpu()    
    image_size = (112,112)
    self.model = get_model(ctx, image_size, 'model_weights/insight_face/face-model-r100-ii/model,0', 'fc1')
    self.image_size = image_size
    self.detector =  MtcnnDetector(minsize=20, model_folder='model_weights/insight_face/mtcnn-model', ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.6,0.7,0.8])


  def getAlignedImage(self, img):
    ret = self.detector.detect_face(img, det_type = 0)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    data = sorted(zip(bbox, points), key=lambda x: (x[0][2]-x[0][0])*(x[0][3]-x[0][1]), reverse=True)
    bbox, points = data[0]
    bbox = bbox[:4]
    points = points.reshape((2,5)).T
    nimg = face_preprocess.preprocess(img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    return aligned

  def getFaceFeatures(self, image):
    aligned = self.getAlignedImage(image)
    if aligned is None:
      return None
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding

