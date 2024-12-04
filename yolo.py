from ultralytics import YOLO
import cv2
import numpy as np

# Charger le modèle YOLOv8
model = YOLO('yolov8s.pt')  # Utilisez 'yolov8n.pt' pour un modèle plus léger si nécessaire

def detect_from_video(video_path):
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin de la vidéo.")
            break

        # Redimensionner la frame pour accélérer le traitement (optionnel)
        frame = cv2.resize(frame, (640, 384))

        # Détection avec YOLOv8
        results = model(frame, conf=0.25)  # Ajuster le seuil de confiance si nécessaire

        # Annoter manuellement les objets détectés
        for obj in results[0].boxes:
            # Extraire les coordonnées, classe et confiance
            x1, y1, x2, y2 = map(int, obj.xyxy[0])  # Convertir les coordonnées en entiers
            cls = int(obj.cls[0])  # Classe détectée
            conf = float(obj.conf[0])  # Confiance
            
            # Récupérer le nom de la classe (si disponible)
            label = f"{model.names[cls]} {conf:.2f}" if model.names else f"Classe {cls} {conf:.2f}"
            
            # Dessiner le rectangle autour de l'objet
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vert pour la boîte
            # Ajouter le label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Afficher la frame annotée
        cv2.imshow('Détection en temps réel', frame)
        
        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Chemin vers la vidéo
video_path = 'video.mp4'  # Remplacez par le chemin correct vers votre vidéo
detect_from_video(video_path)
