import argparse
from ultralytics import YOLO

def train(data_path, epochs, img_size):
    # Load a model
    model = YOLO("yolov8n-cls.pt")
    # Train the model
    results = model.train(data=data_path, epochs=epochs, imgsz=img_size)

def infer(model_path, video_path, save_results):
    # Load the model
    model = YOLO(model_path)
    # Predict with the model
    results = model(video_path, save=save_results)

def main():
    parser = argparse.ArgumentParser(description="Train or Inference with YOLO model")
    subparsers = parser.add_subparsers(dest="command")

    # Subparser for training
    train_parser = subparsers.add_parser("train", help="Train the YOLO model")
    train_parser.add_argument("--data", type=str, required=True, help="Path to the training data")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")

    # Subparser for inference
    infer_parser = subparsers.add_parser("infer", help="Inference with the YOLO model")
    infer_parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    infer_parser.add_argument("--video", type=str, required=True, help="Path to the video for inference")
    infer_parser.add_argument("--save", action='store_true', help="Save the inference results")

    args = parser.parse_args()

    if args.command == "train":
        train(args.data, args.epochs, args.imgsz)
    elif args.command == "infer":
        infer(args.model, args.video, args.save)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
