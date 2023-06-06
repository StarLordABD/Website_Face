import sys
sys.setrecursionlimit(50000)

import face_recognition
import os, sys
import cv2
import numpy as np
import math
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
import face_recognition
from tensorflow.keras.models import model_from_json
import paho.mqtt.client as mqtt

mqtt_server = "91.121.93.94"
mqtt_port = 1883
mqtt_topic = "device/esp/relay"
mqtt_client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
    else:
        print("Connection failed")

mqtt_client.on_connect = on_connect
mqtt_client.connect(mqtt_server, mqtt_port)

relay_on = False

def ressource_path(relative_path):
    try:
        base_path=sys._MEIPASS

    except Exception:
        base_path=os.path.abspath(",")

    return os.path.join(base_path,relative_path)


root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
json_file = open('antispoofing_models/finalyearproject_antispoofing_model_mobilenet.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_weights('antispoofing_models/finalyearproject_antispoofing_model_98-0.967368.h5')
print("Model loaded from disk")
# video.open("http://192.168.1.101:8080/video")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)



# Helper
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


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

            name = os.path.splitext(image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
        print(self.known_face_names)
        

    def run_recognition(self):
        video_capture =cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')
        
        while True:
            try:
                ret,frame = video_capture.read()
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,1.3,5)

                if len(faces) == 0 and relay_on:
                    mqtt_client.publish(mqtt_topic, "0")  # turn relay off
                    relay_on = False

                for (x,y,w,h) in faces:  
                    face = frame[y-5:y+h+5,x-5:x+w+5]
                    resized_face = cv2.resize(face,(160,160))
                    resized_face = resized_face.astype("float") / 255.0
            # resized_face = img_to_array(resized_face)
                    resized_face = np.expand_dims(resized_face, axis=0)
            # pass the face ROI through the trained liveness detector
            # model to determine if the face is "real" or "fake"
                    preds = model.predict(resized_face)[0]
                    print(preds)
                    if preds> 0.5:
                        label = 'fake'
                        cv2.putText(frame, label, (x,y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        cv2.rectangle(frame, (x, y), (x+w,y+h),(0, 0, 255), 2)
                        
                        mqtt_client.publish(mqtt_topic, "0")  # turn relay off
                        relay_on = False

                        if self.process_current_frame:
                            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                            rgb_small_frame = small_frame[:, :, ::-1]
                            self.face_locations = face_recognition.face_locations(rgb_small_frame)
                            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                            self.face_names = []
                            for face_encoding in self.face_encodings:
                        # See if the face is a match for the known face(s)
                                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                                name = "Unknown"
                                confidence = '???'

                                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    name = self.known_face_names[best_match_index]
                                    confidence = face_confidence(face_distances[best_match_index])

                                self.face_names.append(f'{name} ({confidence})')

                        self.process_current_frame = not self.process_current_frame

                        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4

                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                        cv2.imshow('Face Recognition', frame)

                        
                              
                    else:
                        label = 'real'
                        cv2.putText(frame, label, (x,y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        cv2.rectangle(frame, (x, y), (x+w,y+h),  (0, 255, 0), 2)

                        
                   
                        if self.process_current_frame:
                            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                            rgb_small_frame = small_frame[:, :, ::-1]
                            self.face_locations = face_recognition.face_locations(rgb_small_frame)
                            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                            self.face_names = []
                            for face_encoding in self.face_encodings:
                        # See if the face is a match for the known face(s)
                                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                                name = "Unknown"
                                confidence = '???'

                                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                                best_match_index = np.argmin(face_distances)
                                if name == "Unknown":
                                    relay_on=False
                    
                                if matches[best_match_index]:
                                    name = self.known_face_names[best_match_index]
                                    confidence = face_confidence(face_distances[best_match_index])

                                    mqtt_client.publish(mqtt_topic, "1")  # turn relay on
                                    relay_on = True

                                self.face_names.append(f'{name} ({confidence})')

                        self.process_current_frame = not self.process_current_frame


                        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4

                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                        cv2.imshow('Face Recognition', frame)

                if cv2.waitKey(1) == ord('q'):
                    break
                
                key = cv2.waitKey(1)
            except Exception as e:
                pass

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

    print(sys.getrecursionlimit())


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
