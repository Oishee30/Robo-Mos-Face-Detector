import socket
import cv2
import numpy as np
import os
from PIL import Image
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf
TCP_IP = '0.0.0.0'
TCP_PORT = 8000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)
conn, addr = s.accept()
faceDetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
recognizer=cv2.face.LBPHFaceRecognizer_create();
path='dataSet'

choice=conn.recv(1024).decode('utf-8')
def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print ID
        IDs.append(ID)
        cv2.imshow("trainning",faceNp)
        cv2.waitKey(10)
    return IDs, faces
    cv2.destroyAllWindows()
if(choice=='1'):

     #cam=cv2.VideoCapture(0);

     #id=raw_input('enter user id')
    
     sampleNum=0;
     id=conn.recv(1024).decode('utf-8')
     while(True):
        length = recvall(conn,16)
        stringData = recvall(conn, int(length))
        data = np.fromstring(stringData, dtype='uint8')
   
   
        img=cv2.imdecode(data,1)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray,1.3,5);
        for(x,y,w,h) in faces:
           sampleNum=sampleNum+1;
           #print sampleNum
           cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
           cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.waitKey(100);
        cv2.imshow("Face",img);
       
        cv2.waitKey(1);         
        if(sampleNum>20):
         #print "bla"
          break
       


     Ids,faces=getImagesWithID(path)
     recognizer.train(faces,np.array(Ids))
     recognizer.save('recognizer/trainningData.yml')
     cv2.destroyAllWindows()
if(choice=='2'):
     #cam=cv2.VideoCapture(0);
     rec=cv2.face.LBPHFaceRecognizer_create();
     rec.read("trainningData.yml")
     id=0
     font = cv2.FONT_HERSHEY_DUPLEX
     while(True):
        length = recvall(conn,16)
        stringData = recvall(conn, int(length))
        data = np.fromstring(stringData, dtype='uint8')
        img=cv2.imdecode(data,1)
        #img=cv2.imread('User.4.4.')
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray,1.3,5);
        for(x,y,w,h) in faces:
           cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
           id,conf=rec.predict(gray[y:y+h,x:x+w])
           if(id==1):
              id="maisha"
           elif(id==2):
              id="oishee"
           elif(id==3):
              id="Murad Sir"
           elif(id==4):
              id="Zawad"
           else:
              id="unknown"
        conn.send(id.encode('utf-8'))

     #cam.release()
     cv2.destroyAllWindows()





    
