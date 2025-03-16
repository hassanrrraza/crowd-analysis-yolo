import cv2
import time
import os
import argparse

def extract_frames(video_path, output_dir, max_frames=200, interval=0.01, resize=(1080, 500)):
    """
    Extract frames from a video file.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frames
        max_frames (int): Maximum number of frames to extract
        interval (float): Time interval between frame captures in seconds
        resize (tuple): Width and height to resize frames to
    
    Returns:
        int: Number of frames successfully extracted
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0
    
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video reached after {frame_count} frames")
            break
            
        # Resize frame if resize is specified
        if resize:
            frame = cv2.resize(frame, resize)
            
        # Save frame
        output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_path, frame)
        
        # Optional: Display frame
        cv2.imshow("Extracting Frames", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to stop
            break
            
        frame_count += 1
        time.sleep(interval)  # Wait between frame captures
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Successfully extracted {frame_count} frames to {output_dir}")
    return frame_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save extracted frames")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum number of frames to extract")
    parser.add_argument("--interval", type=float, default=0.01, help="Time interval between frame captures in seconds")
    parser.add_argument("--width", type=int, default=1080, help="Width to resize frames to")
    parser.add_argument("--height", type=int, default=500, help="Height to resize frames to")
    
    args = parser.parse_args()
    
    extract_frames(
        args.video, 
        args.output, 
        args.max_frames, 
        args.interval, 
        (args.width, args.height)
    ) 