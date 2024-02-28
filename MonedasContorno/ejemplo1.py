import cv2

#MANERA EN LA CUAL SE MODIFICARA LA IMAGEN
img = cv2.imread('Naruto.jpg')
img_2 = cv2.resize(img,(0,0),fx=0.5, fy=0.5)
img3= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img4= cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
_,umbral=cv2.threshold(img3,100,255,cv2.THRESH_BINARY)
contorno,jerarquía = cv2.findContours(umbral,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contorno,-1,(251,63,52),3)

#SE MOSTRARAN LAS IMAGENES
cv2.imshow('Naruto', img_2)
cv2.imshow('Naruto_GRIS', img3)
cv2.imshow('normal', img)
cv2.imshow('YUv', img4)
cv2.imshow('Imagen Umbral', umbral)
#SE GUARDA LA IMAGEN
cv2.imwrite("Naruto_reducido.jpg", img_2)
cv2.imwrite("Naruto_gris.jpg", img3)
cv2.waitKey(0)