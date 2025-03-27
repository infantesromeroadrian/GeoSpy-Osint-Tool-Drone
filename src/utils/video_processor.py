import cv2
import os
import time
import numpy as np
from typing import List, Tuple, Optional, Generator
import threading

class VideoProcessor:
    """
    Utility for processing video streams and extracting frames for analysis.
    """
    
    def __init__(self, output_dir: str = "./data/frames"):
        """
        Initialize the video processor with output directory.
        
        Args:
            output_dir: Directory to save extracted frames
        """
        self.output_dir = output_dir
        self.is_processing = False
        self._ensure_dir_exists(output_dir)
        self.current_video_capture = None
        self.processing_thread = None
    
    def _ensure_dir_exists(self, directory: str):
        """Create directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def extract_frames(self, video_path: str, interval_sec: float = 1.0, max_frames: int = 10) -> List[str]:
        """
        Extract frames from a video file at specified intervals.
        
        Args:
            video_path: Path to the video file
            interval_sec: Time interval between frames in seconds
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of paths to extracted frames
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_sec)
        
        extracted_frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames at intervals
        for i in range(0, min(total_frames, max_frames * frame_interval), frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Generate output filename
            filename = f"{os.path.basename(video_path)}_frame_{frame_count:04d}.jpg"
            output_path = os.path.join(self.output_dir, filename)
            
            # Save the frame
            cv2.imwrite(output_path, frame)
            extracted_frames.append(output_path)
            
            frame_count += 1
            if frame_count >= max_frames:
                break
                
        cap.release()
        return extracted_frames
    
    def start_stream_processing(self, stream_url: str, process_interval_sec: float = 2.0):
        """
        Start processing a live video stream in a separate thread.
        
        Args:
            stream_url: URL or camera index for the video stream
            process_interval_sec: Time interval between frame processing
        """
        if self.is_processing:
            self.stop_stream_processing()
            
        self.is_processing = True
        self.processing_thread = threading.Thread(
            target=self._process_stream,
            args=(stream_url, process_interval_sec),
            daemon=True
        )
        self.processing_thread.start()
    
    def _process_stream(self, stream_url: str, process_interval_sec: float):
        """
        Process a live video stream, extracting frames at intervals.
        
        Args:
            stream_url: URL or camera index for the video stream
            process_interval_sec: Time interval between frame processing
        """
        # Convert string to int if it's a camera index
        if stream_url.isdigit():
            stream_url = int(stream_url)
            
        # Open the video stream
        self.current_video_capture = cv2.VideoCapture(stream_url)
        
        if not self.current_video_capture.isOpened():
            print(f"Error: Could not open video stream at {stream_url}")
            self.is_processing = False
            return
            
        frame_count = 0
        last_process_time = time.time()
        
        try:
            while self.is_processing:
                # Read a frame
                ret, frame = self.current_video_capture.read()
                
                if not ret:
                    print("Error: Failed to read frame from stream")
                    break
                    
                current_time = time.time()
                elapsed = current_time - last_process_time
                
                # Process frame at specified interval
                if elapsed >= process_interval_sec:
                    # Generate output filename
                    timestamp = int(current_time)
                    filename = f"stream_frame_{timestamp}.jpg"
                    output_path = os.path.join(self.output_dir, filename)
                    
                    # Save the frame
                    cv2.imwrite(output_path, frame)
                    print(f"Saved frame: {output_path}")
                    
                    last_process_time = current_time
                    frame_count += 1
                    
                # Don't max out CPU - small delay between frame reads
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error in stream processing: {str(e)}")
        finally:
            if self.current_video_capture:
                self.current_video_capture.release()
            self.is_processing = False
    
    def stop_stream_processing(self):
        """Stop the video stream processing."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        if self.current_video_capture:
            self.current_video_capture.release()
            self.current_video_capture = None
    
    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, str]]:
        """
        Get the latest frame from the output directory.
        
        Returns:
            Tuple of (frame as numpy array, frame path) or None if no frames exist
        """
        files = [f for f in os.listdir(self.output_dir) if f.endswith('.jpg')]
        if not files:
            return None
            
        # Get the most recent file
        files.sort(reverse=True)
        latest_frame_path = os.path.join(self.output_dir, files[0])
        
        # Read the frame
        frame = cv2.imread(latest_frame_path)
        if frame is None:
            return None
            
        return (frame, latest_frame_path)
    
    def get_frame_generator(self, video_path: str, interval_sec: float = 0.5) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Create a generator that yields frames from a video file at specified intervals.
        
        Args:
            video_path: Path to the video file
            interval_sec: Time interval between frames in seconds
            
        Yields:
            Tuple of (frame as numpy array, timestamp in seconds)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_sec)
        frame_count = 0
        
        try:
            while True:
                # Set position to next frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                # Calculate timestamp in seconds
                timestamp = (frame_count * frame_interval) / fps
                
                yield (frame, timestamp)
                frame_count += 1
                
        finally:
            cap.release() 