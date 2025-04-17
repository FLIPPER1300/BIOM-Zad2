import cv2

# Načítanie Haar Cascade modelu pre detekciu tvárí
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Zapnutie webkamery (0 = default kamera)
cap = cv2.VideoCapture(0)

print("Stlač Q pre ukončenie...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konvertuj na odtiene sivej – zlepšuje výkon detekcie
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekcia tvárí
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Nakreslenie obdlžníkov okolo detegovaných tvárí
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Zobrazenie výsledku
    cv2.imshow('Detekcia tvárí - Live', frame)

    # Ukonči, ak stlačíš klávesu 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Uvoľnenie kamery a zatvorenie okna
cap.release()
cv2.destroyAllWindows()
