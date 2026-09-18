import os
import cv2
import numpy as np
import pandas as pd
import random

def generate_data():
    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)
    os.makedirs("dataset/images/val", exist_ok=True)
    os.makedirs("dataset/labels/val", exist_ok=True)

    records = []

    for i in range(120):
        split = "train" if i < 100 else "val"
        img_name = f"product_{i:04d}"
        img_path = f"dataset/images/{split}/{img_name}.jpg"
        label_path = f"dataset/labels/{split}/{img_name}.txt"

        img = np.full((300, 300, 3), 180, dtype=np.uint8)
        noise = np.random.normal(0, 8, (300, 300, 3)).astype(np.uint8)
        img = cv2.add(img, noise)

        temp = round(random.uniform(60.0, 95.0), 1)
        vibration = round(random.uniform(0.5, 4.5), 2)
        humidity = round(random.uniform(30.0, 80.0), 1)
        shift = random.choice(["Morning", "Evening", "Night"])

        has_defect = random.choice([True, True, False])
        defect_type = "None"
        cls_id = -1

        if has_defect:
            if temp > 78.0:
                defect_type = "Thermal Scratch"
                cls_id = 0
                x1, y1, x2, y2 = 50, 50, 150, 70
                cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), -1)
            elif vibration > 2.8:
                defect_type = "Vibration Chatter"
                cls_id = 1
                x1, y1, x2, y2 = 100, 100, 150, 150
                cv2.circle(img, (125, 125), 25, (40, 40, 40), -1)
            else:
                defect_type = "Surface Crack"
                cls_id = 2
                x1, y1, x2, y2 = 80, 80, 180, 180
                cv2.line(img, (x1, y1), (x2, y2), (10, 10, 10), 3)

            bw = (x2 - x1) / 300.0
            bh = (y2 - y1) / 300.0
            bx = (x1 / 300.0) + (bw / 2.0)
            by = (y1 / 300.0) + (bh / 2.0)

            with open(label_path, "w") as f:
                f.write(f"{cls_id} {bx:.4f} {by:.4f} {bw:.4f} {bh:.4f}\n")

        cv2.imwrite(img_path, img)
        records.append({
            "image_id": f"{img_name}.jpg",
            "shift": shift,
            "machine_temp": temp,
            "vibration": vibration,
            "humidity": humidity,
            "has_defect": int(has_defect),
            "defect_type": defect_type
        })

    pd.DataFrame(records).to_csv("telemetry_data.csv", index=False)
    
    yaml_content = "path: ./dataset\ntrain: images/train\nval: images/val\nnames:\n  0: Thermal Scratch\n  1: Vibration Chatter\n  2: Surface Crack\n"
    with open("data.yaml", "w") as f:
        f.write(yaml_content)

if __name__ == "__main__":
    generate_data()