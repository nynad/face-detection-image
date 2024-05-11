import cv2 

img=cv2.imread("peopleface.jpeg")

imggrey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

haarcascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face=haarcascade.detectMultiScale(imggrey,scaleFactor=1.1,minNeighbors=9)
# 
for (x,y,w,h) in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("face tracking",img) 
cv2.waitKey(0)
