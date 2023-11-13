
import cv2

camara = cv2.VideoCapture(0,cv2.CAP_DSHOW)

dog = cv2.CascadeClassifier('dogface.xml')

while True:
	
	ret,frame = camara.read()
	gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	det = dog.detectMultiScale(gris,
	scaleFactor = 3,
	minNeighbors = 30,
	minSize=(55,55),
	maxSize=(500,500))

	for (x,y,w,h) in det:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		cv2.putText(frame,'Perro',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)

	cv2.imshow('frame',frame)
	
	if cv2.waitKey(1) == 27:
		break
camara.release()
cv2.destroyAllWindows()



