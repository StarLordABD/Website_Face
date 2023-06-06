import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json
from tqdm import tqdm
import time
import face_recognition

import face_recognition
import os, sys
import cv2
import numpy as np
import math

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('face recognition'):
            face_image = face_recognition.load_image_file(f"face recognition/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)

    def run_recognition(self):
        video_capture =cv2.VideoCapture(0)

root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
json_file = open('antispoofing_models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_weights('antispoofing_models/antispoofing_model.h5')
print("Model loaded from disk")

for img in tqdm(os.listdir(os.path.join(root_dir,'test'))):
    t1 = time.time()
    img_arr = cv2.imread(os.path.join(root_dir,'test',img))
    resized_face = cv2.resize(img_arr,(160,160))
    resized_face = resized_face.astype("float") / 255.0
    # resized_face = img_to_array(resized_face)
    resized_face = np.expand_dims(resized_face, axis=0)
    # pass the face ROI through the trained liveness detector
    # model to determine if the face is "real" or "fake"
    preds = model.predict(resized_face)[0]
    if preds> 0.7:
        label = 'fake'
        t2 = time.time()
        print( 'Time taken was {} seconds'.format( t2 - t1))
    else:
        label = 'real'
        t2 = time.time()
        print( 'Time taken was {} seconds'.format( t2 - t1))
        
if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()

