import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
from preprocess import preprocess

class DetectModel:
    def __init__(self, model_path):
        graph = tf.Graph()
        with graph.as_default():
            with tf.gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                self.input_image = graph.get_tensor_by_name('image_tensor:0')
                self.boxes_node = graph.get_tensor_by_name('detection_boxes:0')
                self.scores_node = graph.get_tensor_by_name('detection_scores:0')
                self.classes_node = graph.get_tensor_by_name('detection_classes:0')
                self.num_detections_node = graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=graph)

    def detect(self, img):
        (boxes, scores, _, _) = self.sess.run([self.boxes_node, self.scores_node, self.classes_node, self.num_detections_node],
                        feed_dict={self.input_image: np.array([img])})
        if len(boxes[0]) == 0:
            return None

        h,w,_ = img.shape
        boxes = sorted(zip(boxes[0], scores[0]), key=lambda x: x[1], reverse=True)
        bbox, score = boxes[0]
        if score < 0.33:
            return None

        y1, x1, y2, x2 = [bbox[0]*h, bbox[1]*w, bbox[2]*h, bbox[3]*w]
        return [x1, y1, x2, y2]

class LandmarkModel:
    def __init__(self, model_path):
        graph = tf.Graph()
        with graph.as_default():
            with tf.gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                self.input_image = graph.get_tensor_by_name('Cast:0')
                self.logits_node = graph.get_tensor_by_name('layer6/logits/BiasAdd:0')

        self.sess = tf.Session(graph=graph)

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

    def detect(self, img, bbox):
        if bbox == None:
            return None

        h, w, _ = img.shape
        x1, y1, x2, y2 = self.convert_to_square(bbox)

        x1 = int(max(x1, 0))
        y1 = int(max(y1, 0))
        x2 = int(min(x2, w - 1))
        y2 = int(min(y2, h - 1))

        crop =  img[y1:y2,x1:x2]
        crop = cv2.resize(crop, (112,112))
        logits = self.sess.run([self.logits_node], feed_dict={self.input_image: np.array([crop])})[0]
        logits = logits[0]

        points = []
        for i in range(5):
            x = x1 + logits[2*i] * (x2-x1) / 112
            y = y1 + logits[2*i+1]*(y2-y1) / 112
            points.append([x, y])

        return np.array(points)

class FaceNetModel:
    def __init__(self, model_path):
        graph = tf.Graph()
        with graph.as_default():
            with tf.gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                self.input_image = graph.get_tensor_by_name("input:0")
                self.embeddings = graph.get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

        self.sess = tf.Session(graph=graph)

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = (x-mean)/std_adj
        return y

    def getEmbedding(self, img):
        x = self.prewhiten(img)
        feed_dict = { self.input_image: np.array([x]), self.phase_train_placeholder:False }
        return self.sess.run(self.embeddings, feed_dict=feed_dict)[0]    

class InsightFaceModel:
    def __init__(self, model_path):
        graph = tf.Graph()
        with graph.as_default():
            with tf.gfile.GFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                self.input_image = graph.get_tensor_by_name("input_image:0")
                self.embeddings = graph.get_tensor_by_name("embd_extractor/BatchNorm_1/Reshape_1:0")
                self.train_phase = graph.get_tensor_by_name("train_phase:0")
                self.train_phase_last = graph.get_tensor_by_name("train_phase_last:0")


        self.sess = tf.Session(graph=graph)

    def getEmbedding(self, img):
        x = img/127.5-1.0
        feed_dict = { self.input_image: np.array([x]), self.train_phase: False, self.train_phase_last: False}
        embedding = self.sess.run(self.embeddings, feed_dict=feed_dict)[0]
        return embedding/np.linalg.norm(embedding)


class FaceDescriptor():
    def __init__(self):

        self.detect_model = DetectModel('model_weights/face_detect/face0.75.pb')
        self.landmark_model = LandmarkModel('model_weights/face_detect/landmark.pb')
        self.model1 = FaceNetModel('model_weights/face_net/20180402-114759.pb')
        self.model2 = InsightFaceModel('model_weights/insight_face/face_model.pb')
    
    def getFaceFeatures(self, img):
        bbox = self.detect_model.detect(img)
        
        if bbox == None:
            return None

        points = self.landmark_model.detect(img, bbox)

        img1 = preprocess(img, bbox, None, image_size='160,160')
        img2 = preprocess(img, bbox, points, image_size='112,112')
                    
        return self.model1.getEmbedding(img1), self.model2.getEmbedding(img2)
