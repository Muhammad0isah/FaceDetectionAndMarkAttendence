import numpy as np
import cv2
import os
import face_recognition
from datetime import datetime

path = 'Images'
images = []
className = []
myList = os.listdir(path)
# print(myList)
for currentList in myList:
    currentImage = cv2.imread(f'{path}/{currentList}')
    images.append(currentImage)
    className.append(os.path.splitext(currentList)[0])


# print(className)

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendence(name):
    with open('Attendence\list_attendence.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        # print(myDataList)
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dstring = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dstring}')


# markAttendence('Muhammad')
# markAttendence('Bale')

encodeListKnown = findEncoding(images)
# print(len(encodeListKnown))
# print("Encode complete")
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDistance)
        machIndex = np.argmin(faceDistance)

        if matches[machIndex]:
            name = className[machIndex]
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendence(name)

    cv2.imshow('webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
