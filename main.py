"""
Crowd Analysis using YOLOv8

Main entry point for running the people counter.
"""
import os
import argparse
from src.people_counter import PeopleCounter

def main():
    parser = argparse.ArgumentParser(description="Crowd Analysis using YOLOv8")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to the YOLO model file")
    parser.add_argument("--video", type=str, default="cr.mp4", help="Path to the video file")
    parser.add_argument("--classes", type=str, default="coco1.txt", help="Path to the class names file")
    parser.add_argument("--threshold", type=int, default=40, help="Crowd threshold count")
    parser.add_argument("--no-display", action="store_true", help="Run without displaying video")
    parser.add_argument("--skip-frames", type=int, default=3, help="Number of frames to skip")
    
    args = parser.parse_args()
    
    # Try to find files in the expected directory structure
    # If not found, fall back to the original paths
    model_path = args.model
    if not os.path.exists(model_path):
        model_path = os.path.join("models", args.model)
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), args.model)
        
    video_path = args.video
    if not os.path.exists(video_path):
        video_path = os.path.join("data", "videos", args.video)
    if not os.path.exists(video_path):
        video_path = os.path.join(os.path.dirname(__file__), args.video)
        
    class_file = args.classes
    if not os.path.exists(class_file):
        class_file = os.path.join("data", args.classes)
    if not os.path.exists(class_file):
        class_file = os.path.join(os.path.dirname(__file__), args.classes)
    
    print(f"Using model: {model_path}")
    print(f"Using video: {video_path}")
    print(f"Using class file: {class_file}")
    
    counter = PeopleCounter(
        model_path=model_path,
        video_path=video_path,
        class_file=class_file,
        threshold=args.threshold
    )
    
    frame_counts = counter.run(
        display=not args.no_display,
        skip_frames=args.skip_frames
    )
    
    print(f"Maximum count: {max(frame_counts) if frame_counts else 0}")
    print(f"Average count: {sum(frame_counts) / len(frame_counts) if frame_counts else 0:.2f}")

if __name__ == "__main__":
    main() 