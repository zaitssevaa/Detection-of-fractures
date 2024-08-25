# yolo_utils.py
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# Загрузка модели YOLO из файла best.pt
model = YOLO('best2.pt')

def detect_objects(image: Image.Image, confidence_threshold: float):
    # Преобразование изображения PIL в формат numpy array
    img_array = np.array(image)

    # Выполнение детекции с использованием модели YOLO
    results = model(img_array)

    # Извлечение предсказаний
    predictions = results[0].boxes

    # Создание аннотированного изображения
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    detections = []

    for box in predictions:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        confidence = box.conf[0].item()

        if confidence < confidence_threshold:
            continue

        class_name = model.names[class_id]

        # Рисуем прямоугольник и текст на изображении
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        draw.text((x1, y1 - 10), f"{class_name} ({confidence:.2f})", fill='red')

        # Добавляем результат в список детекций
        detections.append({
            'class': class_name,
            'confidence': confidence,
            'bounding_box': [x1, y1, x2, y2]
        })

    return annotated_image, detections
