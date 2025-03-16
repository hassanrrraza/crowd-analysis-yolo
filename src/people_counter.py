import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import os
import time
from datetime import datetime

class PeopleCounter:
    def __init__(self, model_path, video_path, class_file, threshold=40):
        """
        Initialize the PeopleCounter with model and video paths.
        
        Args:
            model_path (str): Path to the YOLO model file
            video_path (str): Path to the video file for analysis
            class_file (str): Path to the class names file
            threshold (int): Crowd threshold count
        """
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.threshold = threshold
        self.start_time = datetime.now()
        self.frame_count = 0
        self.max_count = 0
        self.fps_list = []
        self.fps = 0
        self.last_time = time.time()
        
        # Read class names
        with open(class_file, "r") as my_file:
            data = my_file.read()
            self.class_list = data.split("\n")
        
        # Define the polygon area for counting
        self.area1 = [(463, 147), (426, 460), (89, 428), (14, 222), (249, 105)]
        
        # Color schemes
        self.colors = {
            'normal': (0, 255, 0),       # Green
            'warning': (0, 165, 255),    # Orange
            'critical': (0, 0, 255),     # Red
            'text_bg': (44, 44, 44),     # Dark gray
            'accent': (255, 204, 0),     # Accent color (gold)
            'border': (180, 180, 180),   # Border color
            'polygon': (0, 140, 255),    # Polygon color
            'grid': (50, 50, 50),        # Grid color
        }
        
        # Detection history for the last 100 frames
        self.count_history = []
        self.max_history_length = 100

    def get_crowd_status(self, current_count):
        """Determine crowd status based on the current count and threshold."""
        percentage = (current_count / self.threshold) * 100
        
        if percentage < 60:
            return "NORMAL", self.colors['normal']
        elif percentage < 90:
            return "WARNING", self.colors['warning']
        else:
            return "CRITICAL", self.colors['critical']

    def calculate_fps(self):
        """Calculate frames per second."""
        current_time = time.time()
        time_diff = current_time - self.last_time
        
        if time_diff > 0:
            fps = 1 / time_diff
            self.fps_list.append(fps)
            
            # Keep only the last 10 FPS measurements for smoothing
            if len(self.fps_list) > 10:
                self.fps_list.pop(0)
                
            self.fps = sum(self.fps_list) / len(self.fps_list)
            
        self.last_time = current_time
        return int(self.fps)

    def draw_counting_bar(self, frame, current_count):
        """Draw a progress bar showing the current count."""
        # Update styling to make it more modern
        bar_width = 200
        bar_height = 25
        padding = 20
        bar_x = padding
        bar_y = padding
        
        # Calculate how full the bar should be
        percentage = current_count / self.threshold
        filled_width = min(int(percentage * bar_width), bar_width)
        
        # Determine color based on percentage
        status, color = self.get_crowd_status(current_count)
        
        # Draw background
        cv2.rectangle(frame, (bar_x - 5, bar_y - 5), (bar_x + bar_width + 5, bar_y + bar_height + 5), 
                    self.colors['border'], -1)
        
        # Draw empty bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                    (50, 50, 50), -1)
        
        # Draw filled portion
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                    color, -1)
        
        # Draw border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                    self.colors['border'], 1)
        
        # Add text with background
        label = f'COUNT: {current_count}'
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = bar_x + 10
        text_y = bar_y + bar_height//2 + text_size[1]//2
        
        cvzone.putTextRect(frame, label, (text_x, text_y), 
                          colorR=self.colors['text_bg'], 
                          colorT=(255, 255, 255), 
                          font=cv2.FONT_HERSHEY_SIMPLEX, 
                          scale=0.7, thickness=2,
                          offset=0)

    def draw_threshold_bar(self, frame, current_count):
        """Draw a progress bar showing the threshold percentage."""
        bar_width = 250
        bar_height = 25
        padding = 20
        bar_x = frame.shape[1] - bar_width - padding
        bar_y = padding
        
        percentage = min((current_count / self.threshold), 1.0)
        percentage_display = min(int(percentage * 100), 100)
        filled_width = min(int(percentage * bar_width), bar_width)
        
        status, color = self.get_crowd_status(current_count)
        
        # Draw threshold display
        # Draw background
        cv2.rectangle(frame, (bar_x - 5, bar_y - 5), (bar_x + bar_width + 5, bar_y + bar_height + 5), 
                    self.colors['border'], -1)
        
        # Draw empty bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                    (50, 50, 50), -1)
        
        # Draw filled portion
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                    color, -1)
        
        # Draw border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                    self.colors['border'], 1)
        
        # Add text with background
        label = f'THRESHOLD: {percentage_display}%'
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = bar_x + bar_width - text_size[0] - 10
        text_y = bar_y + bar_height//2 + text_size[1]//2
        
        cvzone.putTextRect(frame, label, (text_x, text_y), 
                          colorR=self.colors['text_bg'], 
                          colorT=(255, 255, 255), 
                          font=cv2.FONT_HERSHEY_SIMPLEX,
                          scale=0.7, thickness=2,
                          offset=0)

    def draw_statistics_panel(self, frame, current_count):
        """Draw a panel with key statistics."""
        panel_width = 250
        panel_height = 180
        panel_x = 20
        panel_y = frame.shape[0] - panel_height - 40  # Moved up to avoid footer overlap
        
        # Update max count
        self.max_count = max(self.max_count, current_count)
        
        # Update count history
        self.count_history.append(current_count)
        if len(self.count_history) > self.max_history_length:
            self.count_history.pop(0)
        
        avg_count = sum(self.count_history) / len(self.count_history) if self.count_history else 0
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                    (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                    self.colors['accent'], 1)
        
        # Add title
        title_x = panel_x + 10
        title_y = panel_y + 30
        cv2.putText(frame, "STATISTICS", (title_x, title_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['accent'], 2)
        
        # Add horizontal line
        cv2.line(frame, (panel_x + 10, title_y + 10), (panel_x + panel_width - 10, title_y + 10), 
               self.colors['accent'], 1)
        
        # Add statistics
        stats_y_start = title_y + 40
        stats_x = panel_x + 15
        line_height = 25  # Reduced line height to fit better
        
        stats = [
            f"Current Count: {current_count}",
            f"Maximum Count: {self.max_count}",
            f"Average Count: {int(avg_count)}",
            f"FPS: {self.calculate_fps()}",
            f"Runtime: {str(datetime.now() - self.start_time).split('.')[0]}"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (stats_x, stats_y_start + i * line_height), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def draw_mini_graph(self, frame, current_count):
        """Draw a mini graph showing count history."""
        if not self.count_history:
            return
            
        graph_width = 200
        graph_height = 100
        graph_x = frame.shape[1] - graph_width - 20
        graph_y = frame.shape[0] - graph_height - 40  # Moved up to avoid footer overlap
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), 
                    (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), 
                    self.colors['accent'], 1)
        
        # Add title
        title_x = graph_x + 10
        title_y = graph_y + 20
        cv2.putText(frame, "COUNT HISTORY", (title_x, title_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['accent'], 1)
        
        # Draw grid
        max_val = max(max(self.count_history), self.threshold)
        grid_step_y = graph_height / 4
        
        for i in range(1, 4):
            y_pos = int(graph_y + graph_height - i * grid_step_y)
            cv2.line(frame, (graph_x, y_pos), (graph_x + graph_width, y_pos), 
                   self.colors['grid'], 1)
            val = int((i / 4) * max_val)
            cv2.putText(frame, str(val), (graph_x - 25, y_pos + 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw graph
        points = []
        history_to_plot = self.count_history[-min(len(self.count_history), graph_width):]
        
        for i, count in enumerate(history_to_plot):
            x = graph_x + i * (graph_width / len(history_to_plot))
            y = graph_y + graph_height - (count / max_val) * graph_height
            points.append((int(x), int(y)))
        
        # Draw line graph
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], self.colors['accent'], 2)

    def draw_detection_boxes(self, frame, detections):
        """Draw improved detection boxes with ID numbers."""
        for (x1, y1, w, h) in detections:
            # Improved visuals for bounding boxes - thinner lines as requested
            color = self.colors['normal']
            thickness = 1  # Reduced thickness from 2 to 1
            
            # Draw the main rectangle with thinner styling
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), color, thickness)
            
            # Add diagonal corner lines for better visibility (also thinner)
            corner_length = min(20, w//4, h//4)  # Slightly shorter corners
            
            # Top-left corner
            cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
            
            # Top-right corner
            cv2.line(frame, (x1 + w, y1), (x1 + w - corner_length, y1), color, thickness)
            cv2.line(frame, (x1 + w, y1), (x1 + w, y1 + corner_length), color, thickness)
            
            # Bottom-left corner
            cv2.line(frame, (x1, y1 + h), (x1 + corner_length, y1 + h), color, thickness)
            cv2.line(frame, (x1, y1 + h), (x1, y1 + h - corner_length), color, thickness)
            
            # Bottom-right corner
            cv2.line(frame, (x1 + w, y1 + h), (x1 + w - corner_length, y1 + h), color, thickness)
            cv2.line(frame, (x1 + w, y1 + h), (x1 + w, y1 + h - corner_length), color, thickness)

    def draw_header(self, frame):
        """Draw a header with title and basic info."""
        # Draw header background
        header_height = 50
        cv2.rectangle(frame, (0, 0), (frame.shape[1], header_height), (30, 30, 30), -1)
        cv2.line(frame, (0, header_height), (frame.shape[1], header_height), self.colors['accent'], 2)
        
        # Add title - center it in the header
        title = "BRT CROWD ANALYSIS SYSTEM"
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        title_x = (frame.shape[1] - text_size[0]) // 2  # Center the title
        cv2.putText(frame, title, (title_x, 35), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['accent'], 2)
        
        # Time is shown in the footer now

    def run(self, display=True, skip_frames=3):
        """
        Run the people counter on the video.
        
        Args:
            display (bool): Whether to display the video feed
            skip_frames (int): Number of frames to skip for processing efficiency
        
        Returns:
            list: Counts per processed frame
        """
        if display:
            cv2.namedWindow('Crowd Analysis', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Crowd Analysis', 1280, 720)
        
        cap = cv2.VideoCapture(self.video_path)
        
        count = 0
        frame_counts = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            count += 1
            if count % skip_frames != 0:
                continue
                
            frame = cv2.resize(frame, (1280, 720))
            results = self.model.predict(frame)
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")

            detection_boxes = []

            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                c = self.class_list[d]
                cx = int(x1 + x2) // 2
                cy = int(y1 + y2) // 2
                w, h = x2 - x1, y2 - y1
                
                # Check if center point is inside the polygon area
                result = cv2.pointPolygonTest(np.array(self.area1, np.int32), ((cx, cy)), False)
                if result >= 0:
                    detection_boxes.append((x1, y1, w, h))

            current_count = len(detection_boxes)
            frame_counts.append(current_count)
            self.frame_count += 1
            
            if display:
                # Create clean frame first
                display_frame = frame.copy()
                
                # Draw stylized region of interest polygon
                cv2.polylines(display_frame, [np.array(self.area1, np.int32)], True, self.colors['polygon'], 2)
                
                # Add a transparent overlay to highlight the counting zone
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [np.array(self.area1, np.int32)], (100, 100, 100))
                cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)
                
                # Draw improved detection boxes
                self.draw_detection_boxes(display_frame, detection_boxes)
                
                # Draw header last (to avoid overlay issues)
                self.draw_header(display_frame)
                
                # Draw bars and panels
                self.draw_counting_bar(display_frame, current_count)
                self.draw_threshold_bar(display_frame, current_count)
                self.draw_statistics_panel(display_frame, current_count)
                self.draw_mini_graph(display_frame, current_count)
                
                # Add a footer with copyright and timestamp information
                footer_y = display_frame.shape[0] - 15  # Slightly higher footer
                
                # Left aligned copyright
                copyright_text = "Note: BRT footage for educational use only. All rights reserved."
                cv2.putText(display_frame, copyright_text, 
                          (20, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Right aligned timestamp
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                time_size = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                time_x = display_frame.shape[1] - time_size[0] - 20
                cv2.putText(display_frame, time_str, 
                          (time_x, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow("Crowd Analysis", display_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                    break

        cap.release()
        if display:
            cv2.destroyAllWindows()
            
        return frame_counts

# Example usage when script is run directly
if __name__ == "__main__":
    # Get paths relative to script location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "best.pt")
    video_path = os.path.join(base_dir, "data", "videos", "cr.mp4")
    class_file = os.path.join(base_dir, "data", "coco1.txt")
    
    # Check if files exist at expected locations
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "best.pt")  # Fallback to root directory
    
    if not os.path.exists(video_path):
        video_path = os.path.join(base_dir, "cr.mp4")  # Fallback to root directory
    
    if not os.path.exists(class_file):
        class_file = os.path.join(base_dir, "coco1.txt")  # Fallback to root directory
    
    counter = PeopleCounter(model_path, video_path, class_file, threshold=40)
    counter.run(display=True) 