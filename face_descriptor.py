import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
from preprocess import preprocess

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = (x-mean)/std_adj
    return y 

class FaceDescriptor():
    def __init__(self):

        detect_graph = tf.Graph()
        with detect_graph.as_default():    
            with tf.gfile.GFile('model_weights/face_detect/face0.75.pb', 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                self.input_image_detect = detect_graph.get_tensor_by_name('image_tensor:0')
                self.boxes_node = detect_graph.get_tensor_by_name('detection_boxes:0')
                self.scores_node = detect_graph.get_tensor_by_name('detection_scores:0')
                self.classes_node = detect_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections_node = detect_graph.get_tensor_by_name('num_detections:0')

        self.detect_sess = tf.Session(graph=detect_graph)

        landmark_graph = tf.Graph()
        with landmark_graph.as_default():
            with tf.gfile.GFile('model_weights/face_detect/landmark.pb', 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                self.input_image_landmark = landmark_graph.get_tensor_by_name('Cast:0')
                self.logits_node = landmark_graph.get_tensor_by_name('layer6/logits/BiasAdd:0')

        self.landmark_sess = tf.Session(graph=landmark_graph)

        graph1 = tf.Graph()
        with graph1.as_default():
            with gfile.FastGFile('model_weights/face_net/20180402-114759.pb', 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                self.input_image1 = graph1.get_tensor_by_name("input:0")
                self.embeddings1 = graph1.get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = graph1.get_tensor_by_name("phase_train:0")

        self.sess1 = tf.Session(graph=graph1)

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
    
    def convert_to_square(self, bbox):
        square_bbox = [0, 0, 0, 0]
        h = bbox[3] - bbox[1] + 1
        w = bbox[2] - bbox[0] + 1
        max_side = np.maximum(h,w)
        square_bbox[0] = bbox[0] + w*0.5 - max_side*0.5
        square_bbox[1] = bbox[1] + h*0.5 - max_side*0.5
        square_bbox[2] = square_bbox[0] + max_side - 1
        square_bbox[3] = square_bbox[1] + max_side - 1
        return square_bbox

    def getAlignedImages(self, img):
        (boxes, scores, _, _) = self.detect_sess.run([self.boxes_node, self.scores_node, self.classes_node, self.num_detections_node],
                                                feed_dict={self.input_image_detect: np.array([img])})
        if len(boxes[0]) == 0:
            return None

        h,w,_ = img.shape
        boxes = sorted(zip(boxes[0], scores[0]), key=lambda x: x[1], reverse=True)
        bbox, score = boxes[0]
        if score < 0.33:
            return None

        y1, x1, y2, x2 = [bbox[0]*h, bbox[1]*w, bbox[2]*h, bbox[3]*w]
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        x1, y1, x2, y2 = self.convert_to_square((x1, y1, x2, y2))

        x1 = int(max(x1, 0))
        y1 = int(max(y1, 0))
        x2 = int(min(x2, w - 1))
        y2 = int(min(y2, h - 1))

        crop =  img[y1:y2,x1:x2]
        crop = cv2.resize(crop, (112,112))
        logits = self.landmark_sess.run([self.logits_node], feed_dict={self.input_image_landmark: np.array([crop])})[0]
        logits = logits[0]

        points = []
        for i in range(5):
            x = x1 + logits[2*i] * (x2-x1) / 112
            y = y1 + logits[2*i+1]*(y2-y1) / 112
            points.append([x, y])


        aligned1 = preprocess(img, bbox, None, image_size='160,160')
        aligned1 = prewhiten(aligned1)        
        
        aligned2 = preprocess(img, bbox, np.array(points), image_size='112,112')
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
