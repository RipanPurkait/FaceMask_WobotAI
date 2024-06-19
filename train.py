import argparse
from ultralytics import YOLO

def train(data_path, epochs, img_size):
    # Load a model
    model = YOLO("yolov8n-cls.pt")
    # Train the model
    results = model.train(data=data_path, epochs=epochs, imgsz=img_size)

def main():
    parser = argparse.ArgumentParser(description="Train the YOLO model")
    parser.add_argument("--data", type=str, required=True, help="Path to the training data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")

    args = parser.parse_args()
    train(args.data, args.epochs, args.imgsz)

if __name__ == "__main__":
    main()
