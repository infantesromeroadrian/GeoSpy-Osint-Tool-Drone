import os
import cv2
import numpy as np
from PIL import Image
import torch
from typing import Dict, List, Any, Tuple, Optional
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Class to detect objects in images using YOLOv8.
    """
    
    def __init__(self, model_name: str = "yolov8x.pt"):
        """
        Initialize the object detector with YOLOv8.
        
        Args:
            model_name: Name or path of the YOLOv8 model to use.
                        Default is yolov8x.pt (extra large model for best accuracy).
                        Other options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
        """
        try:
            # Check if CUDA is available
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load the YOLO model
            self.model = YOLO(model_name)
            logger.info(f"YOLOv8 model {model_name} loaded successfully")
            
            # Get class names from the model
            self.class_names = self.model.names
            logger.info(f"Model has {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"Error initializing YOLOv8 model: {str(e)}")
            raise
    
    def detect_objects(self, image_path: str, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Detect objects in an image using YOLOv8.
        
        Args:
            image_path: Path to the image file
            conf_threshold: Confidence threshold for detection (0.0 to 1.0)
            
        Returns:
            Dictionary with detection results including:
            - detected objects (class, confidence, bounding box)
            - annotated image
            - summary of findings
        """
        try:
            # Run YOLOv8 inference
            results = self.model(image_path, conf=conf_threshold)
            
            # Process results
            detections = []
            object_counts = {}
            has_people = False
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get the detection class
                    cls_id = int(box.cls.item())
                    class_name = self.class_names[cls_id]
                    
                    # Get confidence score
                    confidence = float(box.conf.item())
                    
                    # Get bounding box coordinates (convert to int for display)
                    x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                    
                    # Update object counts
                    if class_name not in object_counts:
                        object_counts[class_name] = 1
                    else:
                        object_counts[class_name] += 1
                    
                    # Check if person detected
                    if class_name == "person":
                        has_people = True
                    
                    # Add detection to the list
                    detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2]
                    })
            
            # Generate annotated image
            annotated_img = results[0].plot()
            annotated_img_path = image_path.replace(".", "_annotated.")
            # Ensure extension is preserved
            if "." not in os.path.basename(annotated_img_path):
                base, ext = os.path.splitext(image_path)
                annotated_img_path = f"{base}_annotated{ext}"
            print(f"Saving annotated image to: {annotated_img_path}")
            cv2.imwrite(annotated_img_path, annotated_img)
            # Verify the file was saved
            if not os.path.exists(annotated_img_path):
                print(f"Warning: Failed to save annotated image at {annotated_img_path}")
                # Try with an alternative path in uploads directory
                uploads_dir = os.path.join("./data/uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                annotated_img_path = os.path.join(uploads_dir, f"annotated_{os.path.basename(image_path)}")
                print(f"Trying alternative path: {annotated_img_path}")
                cv2.imwrite(annotated_img_path, annotated_img)
            
            # Prepare summary
            summary = {
                "total_objects_detected": len(detections),
                "object_counts": object_counts,
                "has_people": has_people,
                "most_common_object": max(object_counts.items(), key=lambda x: x[1])[0] if object_counts else None
            }
            
            # Prepare detailed military-grade assessment
            assessment = self._generate_assessment(object_counts, has_people, image_path)
            
            return {
                "detections": detections,
                "annotated_image_path": annotated_img_path,
                "summary": summary,
                "assessment": assessment
            }
            
        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            return {
                "error": str(e),
                "detections": [],
                "annotated_image_path": None,
                "summary": {"error": "Detection failed"}
            }
    
    def _generate_assessment(self, object_counts: Dict[str, int], has_people: bool, image_path: str) -> Dict[str, Any]:
        """
        Generate a detailed military-grade assessment of the detected objects.
        """
        try:
            # Get image dimensions for area estimation
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            image_area = height * width
            
            # Assess based on detected objects
            threat_level = "LOW"
            tactical_value = "LOW"
            intel_value = "LOW"
            
            # Military/security relevant objects
            security_relevant = {
                "car", "truck", "bus", "motorcycle", "airplane", "helicopter", 
                "boat", "person", "backpack", "umbrella", "handbag", "tie", "suitcase",
                "cell phone", "laptop", "keyboard", "mouse", "remote", "camera",
                "tvmonitor", "microwave", "oven", "toaster", "refrigerator", "book"
            }
            
            # Possible threat objects
            threat_objects = {
                "knife", "scissors", "baseball bat", "sports ball", "skateboard"
            }
            
            # Military vehicles/equipment
            military_objects = {
                "truck", "airplane", "boat"
            }
            
            # Count relevant objects
            security_count = sum(object_counts.get(obj, 0) for obj in security_relevant)
            threat_count = sum(object_counts.get(obj, 0) for obj in threat_objects)
            military_count = sum(object_counts.get(obj, 0) for obj in military_objects)
            
            # Assess threat level
            if threat_count > 2 or object_counts.get("person", 0) > 5:
                threat_level = "MEDIUM"
            if threat_count > 5 or object_counts.get("person", 0) > 10:
                threat_level = "HIGH"
                
            # Assess tactical value
            if security_count > 3 or military_count > 0:
                tactical_value = "MEDIUM"
            if security_count > 7 or military_count > 2:
                tactical_value = "HIGH"
                
            # Assess intelligence value
            if has_people and security_count > 2:
                intel_value = "MEDIUM"
            if has_people and (security_count > 5 or military_count > 0):
                intel_value = "HIGH"
            
            # Calculate population density if people are detected
            population_density = "NONE"
            if has_people:
                person_count = object_counts.get("person", 0)
                # Rough estimation of density
                if person_count > 0:
                    density_ratio = person_count / (image_area / 1000000)  # normalize by MP
                    if density_ratio < 0.5:
                        population_density = "LOW"
                    elif density_ratio < 2:
                        population_density = "MEDIUM"
                    else:
                        population_density = "HIGH"
            
            # Calculate vehicle presence
            vehicle_presence = "NONE"
            vehicle_types = ["car", "truck", "bus", "motorcycle", "bicycle"]
            vehicle_count = sum(object_counts.get(v, 0) for v in vehicle_types)
            if vehicle_count > 0:
                if vehicle_count < 3:
                    vehicle_presence = "LOW"
                elif vehicle_count < 8:
                    vehicle_presence = "MEDIUM"
                else:
                    vehicle_presence = "HIGH"
            
            return {
                "threat_level": threat_level,
                "tactical_value": tactical_value,
                "intelligence_value": intel_value,
                "population_density": population_density,
                "vehicle_presence": vehicle_presence,
                "infrastructure_assessment": self._assess_infrastructure(object_counts),
                "suspicious_activities": threat_count > 0,
                "personnel_count": object_counts.get("person", 0),
                "vehicle_count": vehicle_count
            }
            
        except Exception as e:
            logger.error(f"Error generating assessment: {str(e)}")
            return {
                "error": str(e),
                "threat_level": "UNKNOWN",
                "tactical_value": "UNKNOWN",
                "intelligence_value": "UNKNOWN"
            }
    
    def _assess_infrastructure(self, object_counts: Dict[str, int]) -> str:
        """Evaluate infrastructure density based on detected objects"""
        infrastructure_objects = {
            "traffic light", "fire hydrant", "stop sign", "parking meter",
            "bench", "chair", "couch", "bed", "dining table", "toilet",
            "tvmonitor", "laptop", "cell phone", "microwave", "oven",
            "refrigerator", "clock", "vase"
        }
        
        infra_count = sum(object_counts.get(obj, 0) for obj in infrastructure_objects)
        
        if infra_count == 0:
            return "UNDEVELOPED"
        elif infra_count < 5:
            return "MINIMAL"
        elif infra_count < 10:
            return "MODERATE"
        else:
            return "DEVELOPED" 