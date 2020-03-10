import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep

# get_encoded_faces function encodes all the faces in the specified folder in a dictionary encoded as (name,data)
def get_encoded_faces():

    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".JPEG"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded

#unknown_image_encoded function encodes the data of the test image given to identify
def unknown_image_encoded(img):
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding
'''
classify_face function checks the test image with the dataset to mark them if the program recognizes the image 
and draws a box around them and writes a label below the box
'''
def classify_face(im):

    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)

 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (0, 255, 0), 2)

            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    '''
	Displaying the image with the box and label on the face recognized from the data set
	'''
    while True:

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names 

print(classify_face("test.jpg"))


