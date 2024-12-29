import cv2
import face_recognition as fr
import os
import numpy as np
from datetime import datetime

path = 'faculty_images'    # directory where all the images are stored

#--------------- Traversing images present in the specified path -------------#
images = []
Names = []
myList = os.listdir(path)   # list of all files (images) in the path specified
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    Names.append(os.path.splitext(cl)[0])
#

#----- Function to encode all train image and storing them in a variable -----#
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = fr.face_encodings(img)[0]
        encodeList.append(encoded_face)
    #
    return encodeList
#

encoded_face_train = findEncodings(images)

#------------ Function to mark attendance in CSV file -------------#
p=0
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        now = datetime.now()
        date=now.strftime('%d-%B-%Y')
        datelist=[]
        nameList=[]
        time = now.strftime('%I:%M:%S:%p')
        entrytime="08:00:00:PM"
        exittime="10:00:00:PM"
        global p
        p=0
        for line in myDataList:
            entry = line.split(',')
            datelist.append(entry[2].replace("\n","").replace(" ",""))
            nameList.append(entry[0]) 
        
        #
        if name not in nameList :
            if time > entrytime: 
                ch=int(time.split(':')[0])
                cm=int(time.split(':')[1])
                eh=int(entrytime.split(':')[0])
                em=int(entrytime.split(':')[1])
                h=ch-eh
                m=cm-em
                if h!=0:  
                    if m==0:
                        f.writelines(f'{name}, {time}, {date},"{h} Hour late"\n')
                        p=1
                        print(f"{name} Attendance Done")
                    elif m>0:
                        f.writelines(f'{name}, {time}, {date},"{h} Hour and {m} min late"\n')
                        p=1
                        print(f"{name} Attendance Done")
                    else:
                        f.writelines(f'{name}, {time}, {date},"{m} min late"\n')
                        p=1
                        print(f"{name} Attendance Done")
                            
                else:
                    f.writelines(f'{name}, {time}, {date},"Puch In"\n')
                    p=1
                    print(f"{name} Attendance Done")
        
        if name in nameList :
            flag=0
            count=0
            r=0
            for line in myDataList:
                entry = line.split(',')
                if name==entry[0]:
                    d=entry[2].replace("\n","").replace(" ","")
                    t=entry[1]
                    tm=int(t.split(':')[1])
                    ct=int(time.split(':')[1])
                    r=ct-tm
                    if date==d:
                        if count==0:
                            count=count+1
                            continue
                        if count==1:
                            count=count+1
                            flag=1
                            p=2
                            
            if flag==0:
                if count==1:
                    p=1
                    if r>0:
                        if time < exittime:
                            f.writelines(f'{name}, {time}, {date},"Exit Before Time"\n')
                            p=2
                            print(f"{name} Punch Out")
                        else:
                            f.writelines(f'{name}, {time}, {date},"Punch Out"\n')
                            p=2
                            print(f"{name} Punch Out")
                else:
                    if time > entrytime: 
                        ch=int(time.split(':')[0])
                        cm=int(time.split(':')[1])
                        eh=int(entrytime.split(':')[0])
                        em=int(entrytime.split(':')[1])
                        h=ch-eh
                        m=cm-em
                        if h!=0:  
                            if m==0:
                                f.writelines(f'{name}, {time}, {date},"{h} Hour late"\n')
                                p=1
                                print(f"{name} Attendance Done")
                            elif m>0:
                                f.writelines(f'{name}, {time}, {date},"{h} Hour and {m} min late"\n')
                                p=1
                                print(f"{name} Attendance Done")
                        else:
                            f.writelines(f'{name}, {time}, {date},"{m} min late"\n')
                            p=1
                            print(f"{name} Attendance Done")
                            
                    else:
                        f.writelines(f'{name}, {time}, {date},"Punch In"\n')
                        p=1
                        print(f"{name} Attendance Done")
                        
                        
                     
#--------------- Taking pictures from webcam -----------------#
print("\nInitiating camera...\n")
cap = cv2.VideoCapture(0)
# adds="http://100.86.207.23:8080/video"
# cap.open(adds)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)    # resizing the image to 1/4 (scale down)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)    # scaling down will increase frames per second
    faces_in_frame = fr.face_locations(imgS)
    encoded_faces = fr.face_encodings(imgS, faces_in_frame)
    frame_face_landmarks = fr.face_landmarks(imgS, faces_in_frame)

    for encode_face, faceloc,landmark in zip(encoded_faces, faces_in_frame,frame_face_landmarks):
        matches = fr.compare_faces(encoded_face_train, encode_face)
        faceDist = fr.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        # print(matchIndex)

        if matches[matchIndex]:
            if p==1:
                name = Names[matchIndex]
                y1,x2,y2,x1 = faceloc
                y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4    # since we scaled down by 4 times
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{name} Already Punch In', (x1+6, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                markAttendance(name)
            
            elif p==2:
                name = Names[matchIndex]
                y1,x2,y2,x1 = faceloc
                y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4    # since we scaled down by 4 times
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{name} Already Punch out', (x1+6, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                markAttendance(name)
            
            else:
                name = Names[matchIndex]
                y1,x2,y2,x1 = faceloc
                y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4    # since we scaled down by 4 times
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                markAttendance(name)
        #
        else:
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4    # since we scaled down by 4 times
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, 'Unknown!', (x1+6, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #
    #

    cv2.imshow('webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #
#
