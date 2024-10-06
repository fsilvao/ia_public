import cv2
import numpy as np

# Cargar las clases de COCO
with open("/Users/fredysilva/Documents/Python/IA/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Cargar la red YOLO
net = cv2.dnn.readNet("/Users/fredysilva/Documents/Python/IA/yolov3.weights", "/Users/fredysilva/Documents/Python/IA/yolov3.cfg")

# Obtener las capas de salida del modelo YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Asignar colores aleatorios para cada clase detectada
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Iniciar captura de video (archivo o cámara)
cap = cv2.VideoCapture(1)  # Reemplaza "video.mp4" por 0 para la cámara en vivo

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Obtener dimensiones del frame
    height, width, channels = frame.shape

    # Preprocesar la imagen para YOLO (reducir tamaño y normalizar)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    # Realizar la detección de objetos
    outs = net.forward(output_layers)

    # Variables para almacenar información de los objetos detectados
    class_ids = []
    confidences = []
    boxes = []

    # Procesar cada salida
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filtrar las detecciones por confianza mínima
            if confidence > 0.5:
                # Escalar las coordenadas de la caja delimitadora al tamaño de la imagen original
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordenadas de la caja delimitadora
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Eliminar las cajas redundantes utilizando Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dibujar las cajas y los nombres de los objetos detectados
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Mostrar el frame con la detección de objetos
    cv2.imshow("Detección de Objetos con YOLOv3", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
