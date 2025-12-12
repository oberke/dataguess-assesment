from ultralytics import YOLO
import os

def train_model():
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    # We use coco8.yaml as a small placeholder dataset. 
    # In a real scenario, you would replace this with 'dataset.yaml' pointing to custom data.
    # Enables Mosaic, MixUp via hyperparameters
    results = model.train(
        data="coco8.yaml",  
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        save=True,
        device=0, # Assumes GPU 0
        project="training_logs",
        name="yolov8n_optim",
        
        # Augmentation hyperparameters (approximate high values for 'Strong Augmentation')
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0004,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,  # Mosaic 
        mixup=0.1,   # MixUp
        copy_paste=0.0,
        
        # Training mechanics
        cos_lr=True, # Cosine LR schedule
        amp=True,    # Mixed Precision
    )
    
    # Export the model to ONNX as part of the pipeline entry
    success = model.export(format="onnx", dynamic=True, opset=12)
    print("Training finished. Export success:", success)

if __name__ == "__main__":
    # Ensure the log directory exists
    os.makedirs("training_logs", exist_ok=True)
    train_model()
