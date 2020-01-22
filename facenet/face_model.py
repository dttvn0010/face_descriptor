import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from .preprocess import preprocess
from .align import detect_face

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = (x-mean)/std_adj
    return y 

def getArea(box):   
    return (box[2]-box[0]) * (box[3]-box[1])
    
class FaceModel():
    def __init__(self):
        pnet, rnet, onet = detect_face.create_mtcnn(tf.Session(), 'facenet/align')
        self.pnet = pnet
        self.rnet = rnet
        self.onet = onet
        
        graph = tf.Graph()
        with graph.as_default():
            with gfile.FastGFile('model_weights/face_net/20180402-114759.pb', 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                self.input_image = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        self.sess = tf.Session(graph=graph)
        
    def getAlignedImage(self, img):
        bbox, _ = detect_face.detect_face(img, minsize=50, pnet=self.pnet, rnet=self.rnet, onet=self.onet, 
                                    threshold=[ 0.6, 0.7, 0.8 ], factor=0.709)
    
            
        if len(bbox) >= 1:
            bbox = sorted(bbox, key=getArea, reverse=True)
            bbox =  bbox[0][:4]

            aligned = preprocess(img, bbox, None, image_size='160,160')
            aligned = prewhiten(aligned)        
        
            return aligned
        
        return None
        
    def getFaceFeatures(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        aligned = self.getAlignedImage(image)
        embedding = None
        
        if aligned is not None:
            feed_dict = { self.input_image: np.array([aligned]), self.phase_train_placeholder:False }
            embedding = self.sess.run(self.embeddings, feed_dict=feed_dict)[0]
            return embedding