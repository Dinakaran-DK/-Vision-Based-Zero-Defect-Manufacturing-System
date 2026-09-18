import cv2
import pandas as pd
import joblib
import os
from ultralytics import YOLO

def run_live_inspection():
    # 1. Pre-trained YOLOv8 model (Automatic download on first run)
    yolo_model_path = "runs/detect/defect_yolo/weights/best.pt"
    
    if os.path.exists(yolo_model_path):
        print("Using custom trained model...")
        yolo = YOLO(yolo_model_path)
    else:
        print("Custom trained weight not found! Falling back to standard YOLOv8n...")
        yolo = YOLO("yolov8n.pt")

    # 2. Causal Engine load check
    causal_model = None
    shift_encoder = None
    if os.path.exists("causal_model.pkl") and os.path.exists("shift_encoder.pkl"):
        causal_model = joblib.load("causal_model.pkl")
        shift_encoder = joblib.load("shift_encoder.pkl")
    else:
        print("WARNING: Run 'python train_causal.py' first to enable Root-Cause AI.")

    cap = cv2.VideoCapture(0) # Camera setup

    telemetry = {
        "shift": "Night",
        "machine_temp": 82.5,
        "vibration": 3.1,
        "humidity": 45.0
    }

    print("Press 'q' to exit live camera.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, verbose=False)[0]
        boxes = results.boxes

        if len(boxes) > 0:
            box = boxes[0]
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            defect_type = yolo.names[cls_id]

            # Draw Bounding Box
            xyxy = list(map(int, box.xyxy[0].tolist()))
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
            cv2.putText(frame, f"Object/Defect: {defect_type} ({conf*100:.1f}%)", 
                        (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Causal Diagnosis
            if causal_model and shift_encoder:
                shift_enc = shift_encoder.transform([telemetry["shift"]])[0]
                X_in = pd.DataFrame([{
                    "machine_temp": telemetry["machine_temp"],
                    "vibration": telemetry["vibration"],
                    "humidity": telemetry["humidity"],
                    "shift_enc": shift_enc
                }])
                cause = causal_model.predict(X_in)[0]
                cv2.putText(frame, f"Root Cause: {cause}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Overlay Diagnostics
            cv2.putText(frame, f"Temp: {telemetry['machine_temp']}C | Vib: {telemetry['vibration']}mm/s", 
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Smart Factory Live Inspection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_inspection()