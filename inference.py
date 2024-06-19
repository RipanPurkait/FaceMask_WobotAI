import argparse
from ultralytics import YOLO

def infer(model_path, video_path, save_results):
    # Load the model
    model = YOLO(model_path)
    # Predict with the model
    results = model(video_path, save=save_results)

def main():
    parser = argparse.ArgumentParser(description="Inference with the YOLO model")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--video", type=str, required=True, help="Path to the video for inference")
    parser.add_argument("--save", action='store_true', help="Save the inference results")

    args = parser.parse_args()
    infer(args.model, args.video, args.save)

if __name__ == "__main__":
    main()
