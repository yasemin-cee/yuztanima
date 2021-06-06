import cv2
import matplotlib.pyplot as plt

#içe aktarma
steve = cv2.imread("images/steve.jpg", 0)
plt.figure(), plt.imshow(steve, cmap = "gray"), plt.axis("off")
cv2.waitKey(0)

#yüz olup olmadığını sınıflandırma
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

face_rect = face_cascade.detectMultiScale(steve)

for (x, y, w, h) in face_rect :
    cv2.rectangle(steve, (x, y), (x+w, y+h),(255,255,255,), 10)
    plt.figure(), plt.imshow(steve, cmap="gray"), plt.axis("off")

#Milli takım fotoğrafı
turkiye = cv2.imread("images/turkiye.jpg", 0)
plt.figure(), plt.imshow(turkiye, cmap = "gray"), plt.axis("off")

face_rect = face_cascade.detectMultiScale(turkiye, minNeighbors=7)

for (x, y, w, h) in face_rect :
    cv2.rectangle(turkiye, (x, y), (x+w, y+h),(255,255,255,), 10)
    plt.figure(), plt.imshow(turkiye, cmap="gray"), plt.axis("off")

#Bilgisayarın kamerasını açarak yüz tanıma
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors=7)

        for (x, y, w, h) in face_rect:
            cv2.rectangle(frame, (x, y), (x+w, y+h),(255,255,255,), 10)
        cv2.imshow("Yuz tanima", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
