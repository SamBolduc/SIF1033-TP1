import cv2
import numpy as np

# Chemin vers votre vidéo ou utilisez 0 pour la webcam
video_path = './basket.mp4'
# video_path = 0

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()  # Lire une image de la vidéo
    if not ret:
        print("Fin de la vidéo ou erreur de lecture")
        break

    # Définir la couleur minimale et maximale pour le masque
    lower = np.array([5, 70, 70])
    upper = np.array([15, 255, 255])

    # Réduire le bruit avec un flou Gaussian
    mask = cv2.GaussianBlur(frame, (11, 11), 0)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    # Appliquer le range de couleur à masquer
    mask = cv2.inRange(mask, lower, upper)

    # Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Trouver les contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edged = cv2.Canny(mask, 10, 40)
    min_circularity = 0.7
    min_area = 850
    # # Filtrer les contours
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        # Garder seulement une circularité minimale et un air minimal de cercle
        if circularity > min_circularity and area > min_area:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)

    # Afficher l'image originale avec les contours
    cv2.imshow('Image originale avec contours', frame)

    # Afficher l'image seuillée (debugging ou visualisation des contours uniquement)
    cv2.imshow('Image seuillée', mask)

    # Sortie avec la touche 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Libérer la capture et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
