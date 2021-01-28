import os
from PIL import Image
import numpy as np
import cv2
import pickle as pkl

#EXTRACTING CURRENT FILE PATH AND PATH WHERE IMAGES ARE STORED
def train():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(file_dir,"images")

    face_cascade = cv2.CascadeClassifier('libraries\cv2\data\haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    current_id = 1
    label_id = {}
    y_labels = []
    x_train = []


    #labeling the images
    for root , dirs , files in os.walk(image_dir):
        for  file in files:
            path = os.path.join(root,file)
            label = os.path.basename(root)

            if  label not in label_id:
                  label_id[label] = current_id
                  current_id = current_id + 1
            id_for_label = label_id[label]
            

            #extracting whole image info and converting it in numpy array
            pil_image = Image.open(path).convert("L")
            size = (500,550)
            final_image = pil_image.resize(size , Image.ANTIALIAS)
            image_array = np.array(final_image , "uint8")


            #extracting region of intrest from the images and appending for training purpose
            faces = face_cascade.detectMultiScale(image_array, scaleFactor = 1.6 ,minNeighbors=3)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h , x:x+w]
                x_train.append(roi)
                y_labels.append(id_for_label)

    with open("labels.pkl" , "wb") as file:
        pkl.dump(label_id,file)

    #training the face data and saving it using LBPHFACERECOGNIZER
    recognizer.train(x_train , np.array(y_labels))
    recognizer.save("trainer.yml")
