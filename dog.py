
import cv2

camara = cv2.VideoCapture(0,cv2.CAP_DSHOW)

dog = cv2.CascadeClassifier('mydogdetector.xml')

while True:
	
	ret,frame = camara.read()
	gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	det = dog.detectMultiScale(gris,
	scaleFactor = 1.05,
	minNeighbors = 60,
	flags=5,
	minSize=(50,50))#,
	#maxSize=(500,500))

	for (x,y,w,h) in det:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		cv2.putText(frame,'Perro',(x,y-10),2,0.7,(0,255,0),2,cv2.LINE_AA)

	cv2.imshow('frame',frame)
	
	if cv2.waitKey(1) == 27:
		break
camara.release()



