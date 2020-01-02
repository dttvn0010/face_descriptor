import cv2
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from preprocess import preprocess
import align.detect_face

minsize = 50

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = (x-mean)/std_adj
    return y 

class FaceDescriptor():
    def __init__(self):
        pnet, rnet, onet = align.detect_face.create_mtcnn(tf.Session(), 'align')
        self.pnet = pnet
        self.rnet = rnet
        self.onet = onet
        
        graph = tf.Graph()
        with graph.as_default():
            with gfile.FastGFile('model_weights/face_net/20180402-114759.pb', 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                self.input_image1 = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings1 = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        self.sess1 = tf.Session(graph=graph)

        graph2 = tf.Graph()
        with graph2.as_default():    
            with gfile.FastGFile('model_weights/insight_face/face_model.pb', 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')                
                self.input_image2 = graph2.get_tensor_by_name("input_image:0")
                self.embeddings2 = graph2.get_tensor_by_name("embd_extractor/BatchNorm_1/Reshape_1:0")
                self.train_phase = graph2.get_tensor_by_name("train_phase:0")
                self.train_phase_last = graph2.get_tensor_by_name("train_phase_last:0")
                
        self.sess2 = tf.Session(graph=graph2)
        
    def getAlignedImages(self, img):
        bbox, points = align.detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, [ 0.6, 0.7, 0.8 ], 0.709)
        bbox = bbox[0,0:4]
        points = points.reshape((2,5)).T

        aligned1 = preprocess(img, bbox, None, image_size='160,160')
        aligned1 = prewhiten(aligned1)        
        
        aligned2 = preprocess(img, bbox, points, image_size='112,112')
        aligned2 = aligned2/127.5-1.0
             
        return aligned1, aligned2
        
    def getFaceFeatures(self, image):
        aligned1, aligned2 = self.getAlignedImages(image)
       
        feed_dict = { self.input_image1: np.array([aligned1]), self.phase_train_placeholder:False }
        embedding1 = self.sess1.run(self.embeddings1, feed_dict=feed_dict)[0]

        feed_dict = { self.input_image2: np.array([aligned2]), self.train_phase: False, self.train_phase_last: False}
        embedding2 = self.sess2.run(self.embeddings2, feed_dict=feed_dict)[0]
        embedding2 = embedding2/np.linalg.norm(embedding2)
                    
        return embedding1, embedding2