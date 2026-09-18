import os
import sys

def train_model():
    print("--- Starting YOLO Training Setup ---")
    
    # Check if data.yaml exists
    if not os.path.exists("data.yaml"):
        print("ERROR: data.yaml file missing! Please run 'python dataset_generator.py' first.")
        return

    try:
        from ultralytics import YOLO
        print("Ultralytics library loaded successfully.")
        
        # Load pre-trained nano model
        model = YOLO("yolov8n.pt") 
        print("Base model loaded. Starting training...")
        
        # Train model
        model.train(
            data="data.yaml",
            epochs=5,         # Fast training for prototype
            imgsz=300,
            batch=4,
            name="defect_yolo",
            workers=0,        # Prevents Windows multiprocessing crash
            device="cpu"      # Forces CPU execution to avoid CUDA/GPU issues
        )
        print("--- Training Completed Successfully! ---")
        print("Model saved to: runs/detect/defect_yolo/weights/best.pt")

    except Exception as e:
        print(f"CRITICAL ERROR DURING TRAINING: {e}")

if __name__ == "__main__":
    train_model()