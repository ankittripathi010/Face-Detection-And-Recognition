import cv2,numpy
import pickle as pkl


def detection():
    #capturing the video from your webcam
    video = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('libraries\cv2\data\haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")


    #opening the label pickle file and dumping the labels from it to labels dictionary
    labels = {}
    with open("labels.pkl",'rb') as file:
        original_labels = pkl.load(file)
        labels = {value:key for key,value in original_labels.items()}

    
    while True:
        check,frame =  video.read()

        #converting the frame to greyscale and detecting face
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.6,minNeighbors=7)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h , x:x+w]
            roi_color = frame[y:y+h , x:x+w]


            #recognizing the face and wrting the name on image.
            id_for_label ,loss = recognizer.predict(roi_gray)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_for_label]
            color = (0,225,255,0)
            stroke = 2
            cv2.putText(frame , name, (x,y), font , 1 ,color,stroke,cv2.LINE_AA)
        
            rec_color = (255,255,255)
            rec_width = 2
            rec = cv2.rectangle(frame,(x,y),((x+w),(y+h)),rec_color,rec_width)
        cv2.imshow("capture",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;  
    video.release()
    cv2.destroyAllWindows()
