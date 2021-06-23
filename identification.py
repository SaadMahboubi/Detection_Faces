import cv2
import pickle
import common as c

face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml") # lire le fichier d'entrainement par le recognizer
idImg = 0
colorInfo = (255, 255, 255)
colorKo = (0, 0, 255)
colorOk = (0, 255, 0)

with open("labels.pickle", "rb") as f: # lecture du pickle contenant les labels (tableau)
    ogLabels = pickle.load(f)
    labels = {v: k for k, v in ogLabels.items()}

cap = cv2.VideoCapture("Plan_9_from_Outer_Space_1959_640_x_480.webm")
while True:
    ret, frame = cap.read()
    tickMark = cv2.getTickCount()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # image récupéré du fichier du film
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(c.min_size, c.min_size))
    for(x, y, w, h) in faces: # parcours les visage detecté avec le fichier cascade
        roiGray = cv2.resize(gray[y:y+h, x:x+w], (c.min_size, c.min_size)) # on recupere le visage (avec y+h et x+w) en le redimentionnant
        id_, conf = recognizer.predict(roiGray) # on lance la fonction de prédiction, 2 var : un id - indice de confiance 0 sur, 200 pas confiant
        if conf <= 95:
            color = colorOk
            name = labels[id_]
        else:
            color = colorKo
            name = "Inconnu"
        label = name + " " + '{:5.2f}'.format(conf)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, colorInfo, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    fps = cv2.getTickFrequency()/(cv2.getTickCount()-tickMark)
    cv2.putText(frame, 'FPS: {:5.2f}'.format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, colorInfo, 2)
    cv2.imshow('L42Project', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('a'):
        for cpt in range(100):
            ret, frame = cap.read()

cv2.destroyAllWindows()
cap.release()

