import cv2
import os
import common as c
import numpy as np
import pickle

imageDir = "./IMAGES/"
currentId = 0;
label_ids = {}
x_train = [] # photos
y_labels = [] # ensemble des idendité

for root, dirs, files in os.walk(imageDir): # parcours l'arboresance en l'explosant en 3 tableau : root = répertoir parcourir /
                                                                                                #  dirs = liste des repertoire dans le repertoire
                                                                                                #  files = les fichiers présent dans le root
    if len(files): #si on a des images dans le repertoire
        label = root.split("/")[-1] # on recupere le nom du repertoire dernier arg du tableau
        for file in files:
            if file.endswith("png"): # recupere les images
                path = os.path.join(root, file) # concatenation de root et file
                if not label in label_ids:
                    label_ids[label] = currentId
                    currentId += 1
                id_ = label_ids[label]
                image = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (c.min_size, c.min_size)) #resize la taille des images afin d'avoir plus de precision
                fm = cv2.Laplacian(image, cv2.CV_64F).var() #on gere le flou
                if fm < 250:
                    print("Photo exclue:", path, fm) # si trop flou elle est exlue
                else:
                    x_train.append(image) # stockage des images a données a l'entraineur tout a l'heur
                    y_labels.append(id_) # stockage du label

with open("labels.pickle", "wb") as f: #stockage des labels dans le fichier pickle utilisable par l'entraineur
    pickle.dump(label_ids, f)

x_train = np.array(x_train)
y_labels = np.array(y_labels)
recognizer = cv2.face.LBPHFaceRecognizer_create() #appel de la fonction d'apprentissage
recognizer.train(x_train, y_labels) # entraineur prend deux tableau (de même taille) à cote de chaque photo on a une id
recognizer.save("trainner.yml") # enregistrer l'entrainement dans le fichier yml
