import cv2
import numpy as np
import os
import random
from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

model = YOLO(str(MODEL_PATH))
GSD = 0.05  # 5 cm per pixel

def detect_and_annotate(image_path):
    image = cv2.imread(image_path)
    results = model(image, save=False, conf=0.3)

    class_names = list(model.names.values())
    colors = {class_name: [random.randint(0, 255) for _ in range(3)] for class_name in class_names}
    colors[class_names[0]] = [0, 165, 255]  # Orange for solar_panel

    counts = {}
    real_areas = {}
    confidences = {}
    image_area = image.shape[0] * image.shape[1]

    for r in results:
        if r.masks is None:
            continue
        for box, mask in zip(r.boxes, r.masks.data):
            class_id = int(box.cls[0].item())
            class_name = class_names[class_id]

            counts.setdefault(class_name, [0, 0.0])
            real_areas.setdefault(class_name, 0.0)
            confidences.setdefault(class_name, 0.0)

            counts[class_name][0] += 1

            mask = mask.cpu().numpy()
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_pixels = np.sum(mask > 0.5)
            mask_percentage = (mask_pixels / image_area) * 100
            counts[class_name][1] += mask_percentage

            real_areas[class_name] += mask_pixels * (GSD ** 2)
            confidences[class_name] = (
                confidences[class_name] * (counts[class_name][0] - 1) + box.conf[0].item()
            ) / counts[class_name][0]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = colors[class_name]
            label = f"{class_name} {box.conf[0]:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            mask = (mask > 0.5).astype(np.uint8)
            color_mask = np.zeros_like(image, dtype=np.uint8)
            color_mask[mask == 1] = color
            image = cv2.addWeighted(image, 1.0, color_mask, 0.5, 0)

    output_filename = 'result.jpg'
    output_path = str(BASE_DIR / "outputs" / output_filename)
    cv2.imwrite(output_path, image)

    # Prepare detection summary
    detection_summary = {}
    for k in counts:
        detection_summary[k] = {
            "count": counts[k][0],
            "percentage": round(counts[k][1], 2),
            "area_sqm": round(real_areas[k], 2),
            "confidence": round(confidences[k], 2)
        }

    return output_filename, detection_summary
