"""
Single Camera Gun Person Tracking System
Tracks a person with a gun and maintains their ID even when they leave/return to view
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import json
import os
import torchvision.models as models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Person:
    """Represents a tracked person"""
    id: int
    embedding: np.ndarray
    last_seen: float
    appearance_count: int
    gun_detections: int
    trajectory: deque
    confidence_history: deque

@dataclass
class Detection:
    """Represents a detection (person or gun)"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    embedding: Optional[np.ndarray] = None

class ReIDSystem:
    """High-performance ReID system with state-of-the-art models and CUDA acceleration"""
    
    def __init__(self, similarity_threshold: float = 0.7, max_people: int = 50):
        self.similarity_threshold = similarity_threshold
        self.max_people = max_people
        self.people_database: List[Person] = []
        self.next_id = 0
        
        # Setup CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load powerful pre-trained models
        self.models = {}
        self.load_powerful_models()
        
        # High-quality preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((384, 192)),  # Higher resolution for better features
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load existing database if available
        self.load_database()
    
    def load_powerful_models(self):
        """Load state-of-the-art models for ReID with consistent output sizes"""
        try:
            # 1. ResNet-50 (reliable, consistent 2048 features)
            resnet50 = models.resnet50(pretrained=True)
            resnet50.fc = torch.nn.Identity()  # Remove classification head
            resnet50 = resnet50.to(self.device)
            resnet50.eval()
            self.models['resnet50'] = resnet50
            
            # 2. ResNet-101 (more powerful, same 2048 features)
            resnet101 = models.resnet101(pretrained=True)
            resnet101.fc = torch.nn.Identity()  # Remove classification head
            resnet101 = resnet101.to(self.device)
            resnet101.eval()
            self.models['resnet101'] = resnet101
            
            # 3. DenseNet-121 (consistent 1024 features, add projection layer)
            densenet = models.densenet121(pretrained=True)
            densenet.classifier = torch.nn.Identity()  # Remove classification head
            # Add projection layer to match ResNet output size
            densenet.projection = torch.nn.Linear(1024, 2048).to(self.device)
            densenet = densenet.to(self.device)
            densenet.eval()
            self.models['densenet'] = densenet
            
            logger.info(f"Loaded {len(self.models)} models with consistent 2048-dim outputs: {list(self.models.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            # Fallback to basic ResNet-50 only
            resnet50 = models.resnet50(pretrained=True)
            resnet50.fc = torch.nn.Identity()
            resnet50 = resnet50.to(self.device)
            resnet50.eval()
            self.models['resnet50'] = resnet50
    
    def get_embedding(self, img_bgr: np.ndarray) -> np.ndarray:
        """Extract high-quality ensemble embedding with CUDA acceleration"""
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
            
            embeddings = []
            
            # Get embeddings from all models with CUDA acceleration
            with torch.no_grad():
                for model_name, model in self.models.items():
                    try:
                        # Extract features
                        if 'densenet' in model_name.lower():
                            # DenseNet needs projection layer
                            features = model(img_tensor)
                            features = model.projection(features)
                        else:
                            # ResNet models
                            features = model(img_tensor)
                        
                        # L2 normalize for better similarity computation
                        features = F.normalize(features, p=2, dim=1)
                        embeddings.append(features.cpu().numpy().squeeze())
                        
                    except Exception as e:
                        logger.warning(f"Failed to get embedding from {model_name}: {e}")
            
            if not embeddings:
                logger.error("No embeddings extracted from any model")
                return np.zeros(2048)  # Consistent fallback size
            
            # Smart ensemble combination
            if len(embeddings) == 1:
                return embeddings[0]
            
            # Weight models by their typical performance
            model_weights = {
                'resnet101': 0.4,
                'resnet50': 0.35,
                'densenet': 0.25
            }
            
            # Calculate weighted average
            combined_embedding = np.zeros_like(embeddings[0])
            total_weight = 0
            
            for i, (model_name, model) in enumerate(self.models.items()):
                if i < len(embeddings):
                    weight = model_weights.get(model_name, 0.1)
                    combined_embedding += weight * embeddings[i]
                    total_weight += weight
            
            # Normalize final embedding
            if total_weight > 0:
                combined_embedding /= total_weight
                combined_embedding = combined_embedding / (np.linalg.norm(combined_embedding) + 1e-8)
            
            return combined_embedding
            
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return np.zeros(2048)  # Consistent fallback size
    
    def find_best_match(self, embedding: np.ndarray, bbox: Tuple[int, int, int, int] = None, min_confidence: float = 0.5) -> Optional[int]:
        """Find best matching person with optimized similarity computation"""
        if len(self.people_database) == 0:
            return None
        
        best_match_id = None
        best_score = -1
        
        # Convert to tensor for efficient computation
        embedding_tensor = torch.from_numpy(embedding).float().to(self.device)
        
        for person in self.people_database:
            # Fast cosine similarity using PyTorch
            person_embedding_tensor = torch.from_numpy(person.embedding).float().to(self.device)
            cosine_sim = F.cosine_similarity(embedding_tensor.unsqueeze(0), person_embedding_tensor.unsqueeze(0)).item()
            
            # Temporal bonus (simplified)
            time_since_seen = time.time() - person.last_seen
            time_bonus = 1.0
            if time_since_seen < 10:  # 10 seconds
                time_bonus = 1.3
            elif time_since_seen < 30:  # 30 seconds
                time_bonus = 1.1
            
            # Appearance bonus
            appearance_bonus = 1.0 + min(person.appearance_count * 0.01, 0.2)
            
            # Calculate final score
            final_score = cosine_sim * time_bonus * appearance_bonus
            
            if final_score > best_score and final_score > min_confidence:
                best_score = final_score
                best_match_id = person.id
        
        # Higher threshold for better accuracy
        if best_score > self.similarity_threshold:
            return best_match_id
        
        return None
    
    def add_person(self, embedding: np.ndarray, is_gun_person: bool = False) -> int:
        """Add new person to database"""
        person_id = self.next_id
        self.next_id += 1
        
        person = Person(
            id=person_id,
            embedding=embedding,
            last_seen=time.time(),
            appearance_count=1,
            gun_detections=1 if is_gun_person else 0,
            trajectory=deque(maxlen=50),  # Keep last 50 positions
            confidence_history=deque(maxlen=20)
        )
        
        self.people_database.append(person)
        
        # Limit database size
        if len(self.people_database) > self.max_people:
            # Remove oldest person
            oldest_person = min(self.people_database, key=lambda p: p.last_seen)
            self.people_database.remove(oldest_person)
        
        logger.info(f"Added new person with ID {person_id}")
        return person_id
    
    def update_person(self, person_id: int, embedding: np.ndarray, bbox: Tuple[int, int, int, int], 
                     is_gun_person: bool = False, confidence: float = 1.0):
        """Update existing person with new information"""
        for person in self.people_database:
            if person.id == person_id:
                # Update embedding with exponential moving average
                alpha = 0.1  # Learning rate
                person.embedding = (1 - alpha) * person.embedding + alpha * embedding
                
                # Update metadata
                person.last_seen = time.time()
                person.appearance_count += 1
                if is_gun_person:
                    person.gun_detections += 1
                
                # Update trajectory
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                person.trajectory.append((center_x, center_y, time.time()))
                person.confidence_history.append(confidence)
                
                break
    
    def get_person_info(self, person_id: int) -> Optional[Person]:
        """Get person information by ID"""
        for person in self.people_database:
            if person.id == person_id:
                return person
        return None
    
    def save_database(self):
        """Save person database to file"""
        try:
            data = []
            for person in self.people_database:
                data.append({
                    'id': person.id,
                    'embedding': person.embedding.tolist(),
                    'last_seen': person.last_seen,
                    'appearance_count': person.appearance_count,
                    'gun_detections': person.gun_detections,
                    'trajectory': list(person.trajectory),
                    'confidence_history': list(person.confidence_history)
                })
            
            with open('person_database.json', 'w') as f:
                json.dump(data, f)
            
            logger.info(f"Saved {len(data)} people to database")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def load_database(self):
        """Load person database from file"""
        try:
            if os.path.exists('person_database.json'):
                with open('person_database.json', 'r') as f:
                    data = json.load(f)
                
                for item in data:
                    embedding = np.array(item['embedding'])
                    
                    # Handle old database with different embedding sizes
                    if len(embedding) != 2048:
                        logger.warning(f"Old embedding size {len(embedding)} detected, creating new database")
                        # Clear old database and start fresh
                        self.people_database = []
                        self.next_id = 0
                        if os.path.exists('person_database.json'):
                            os.remove('person_database.json')
                        return
                    
                    person = Person(
                        id=item['id'],
                        embedding=embedding,
                        last_seen=item['last_seen'],
                        appearance_count=item['appearance_count'],
                        gun_detections=item['gun_detections'],
                        trajectory=deque(item['trajectory'], maxlen=50),
                        confidence_history=deque(item['confidence_history'], maxlen=20)
                    )
                    self.people_database.append(person)
                
                # Update next_id
                if self.people_database:
                    self.next_id = max(p.id for p in self.people_database) + 1
                
                logger.info(f"Loaded {len(data)} people from database")
        except Exception as e:
            logger.error(f"Failed to load database: {e}")

class GunPersonTracker:
    """High-performance gun person tracking system with CUDA acceleration"""
    
    def __init__(self, gun_model_path: str = "best3.pt", person_model_path: str = "yolov8n.pt"):
        # Setup CUDA device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Tracker using device: {self.device}")
        
        # Initialize models with CUDA
        self.gun_model = YOLO(gun_model_path)
        self.person_model = YOLO(person_model_path)
        
        # Initialize high-performance ReID system
        self.reid_system = ReIDSystem()
        
        # Much more accurate detection parameters
        self.gun_conf_threshold = 0.4  # Higher threshold for gun detection
        self.person_conf_threshold = 0.5  # Higher threshold for person detection
        self.gun_person_distance_threshold = 80  # Maximum distance between gun and person centers
        self.gun_person_overlap_threshold = 0.1  # Minimum overlap between gun and person boxes
        
        # Tracking state
        self.current_gun_person_id = None
        self.gun_person_last_seen = 0
        self.gun_person_timeout = 15  # seconds
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        """Detect guns and people in frame with CUDA acceleration"""
        # Detect guns with CUDA
        gun_results = self.gun_model.predict(frame, conf=self.gun_conf_threshold, device=self.device, verbose=False)[0]
        gun_detections = []
        
        if gun_results.boxes is not None:
            for box, conf in zip(gun_results.boxes.xyxy, gun_results.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                gun_detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_name="gun"
                ))
        
        # Detect people with CUDA
        person_results = self.person_model.predict(frame, conf=self.person_conf_threshold, device=self.device, verbose=False)[0]
        person_detections = []
        
        if person_results.boxes is not None:
            for box, conf, cls in zip(person_results.boxes.xyxy, person_results.boxes.conf, person_results.boxes.cls):
                if int(cls) == 0:  # COCO class 0 is person
                    x1, y1, x2, y2 = map(int, box)
                    person_detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(conf),
                        class_name="person"
                    ))
        
        return gun_detections, person_detections
    
    def match_gun_to_person(self, gun_detections: List[Detection], person_detections: List[Detection]) -> List[Tuple[Detection, Detection]]:
        """Much more accurate gun-to-person matching with strict criteria"""
        matches = []
        
        for gun in gun_detections:
            best_person = None
            best_score = 0
            
            for person in person_detections:
                # Calculate center distance
                gun_center = ((gun.bbox[0] + gun.bbox[2]) / 2, (gun.bbox[1] + gun.bbox[3]) / 2)
                person_center = ((person.bbox[0] + person.bbox[2]) / 2, (person.bbox[1] + person.bbox[3]) / 2)
                distance = np.sqrt((gun_center[0] - person_center[0])**2 + (gun_center[1] - person_center[1])**2)
                
                # Calculate IoU
                iou = self.calculate_iou(gun.bbox, person.bbox)
                
                # Check if gun is inside person bounding box
                gun_inside_person = self.is_gun_inside_person(gun.bbox, person.bbox)
                
                # Calculate score based on multiple criteria
                score = 0
                
                # Primary criterion: Gun must be very close to person
                if distance <= self.gun_person_distance_threshold:
                    # Distance score (closer is better)
                    distance_score = 1.0 - (distance / self.gun_person_distance_threshold)
                    
                    # IoU score (overlap is good)
                    iou_score = min(iou * 2, 1.0)  # Scale IoU
                    
                    # Inside person bonus
                    inside_bonus = 1.5 if gun_inside_person else 1.0
                    
                    # Combined score
                    score = (distance_score * 0.6 + iou_score * 0.4) * inside_bonus
                
                # Only consider if score is high enough
                if score > best_score and score > 0.3:  # Higher threshold for accuracy
                    best_score = score
                    best_person = person
            
            # Only match if we have a very confident match
            if best_person and best_score > 0.5:  # Much higher threshold
                matches.append((gun, best_person))
                logger.info(f"GUN-PERSON MATCH: Distance={self.calculate_distance(gun.bbox, best_person.bbox):.1f}px, "
                           f"IoU={self.calculate_iou(gun.bbox, best_person.bbox):.3f}, Score={best_score:.3f}")
        
        return matches
    
    def calculate_distance(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate distance between centers of two bounding boxes"""
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def is_gun_inside_person(self, gun_bbox: Tuple[int, int, int, int], person_bbox: Tuple[int, int, int, int]) -> bool:
        """Check if gun is inside person bounding box"""
        gx1, gy1, gx2, gy2 = gun_bbox
        px1, py1, px2, py2 = person_bbox
        
        # Check if gun center is inside person box
        gun_center_x = (gx1 + gx2) / 2
        gun_center_y = (gy1 + gy2) / 2
        
        return (px1 <= gun_center_x <= px2 and py1 <= gun_center_y <= py2)
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with improved gun-person detection logic"""
        # Detect objects
        gun_detections, person_detections = self.detect_objects(frame)
        
        # Match guns to people with strict criteria
        gun_person_matches = self.match_gun_to_person(gun_detections, person_detections)
        
        annotated_frame = frame.copy()
        current_gun_person_detected = False
        
        # Draw all person detections first
        for person in person_detections:
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Person {person.confidence:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw gun detections
        for gun in gun_detections:
            x1, y1, x2, y2 = gun.bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"GUN {gun.confidence:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # ONLY process ReID for people who are matched to guns
        for gun, person in gun_person_matches:
            x1, y1, x2, y2 = person.bbox
            
            # Extract person ROI for ReID
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size > 0:
                person_roi_resized = cv2.resize(person_roi, (128, 256))
                embedding = self.reid_system.get_embedding(person_roi_resized)
                
                # Find or assign person ID (this person is confirmed to have a gun)
                person_id = self.reid_system.find_best_match(embedding, person.bbox, min_confidence=0.5)
                
                if person_id is None:
                    # New gun person - add to database
                    person_id = self.reid_system.add_person(embedding, is_gun_person=True)
                    logger.info(f"NEW GUN PERSON DETECTED: ID {person_id}")
                else:
                    # Update existing gun person
                    self.reid_system.update_person(person_id, embedding, person.bbox, is_gun_person=True, confidence=person.confidence)
                
                # Update gun person tracking
                self.current_gun_person_id = person_id
                self.gun_person_last_seen = time.time()
                current_gun_person_detected = True
                
                # Draw gun person with special styling
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)  # Thick red border
                cv2.putText(annotated_frame, f"GUN PERSON ID: {person_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw connection line between gun and person
                gun_center = ((gun.bbox[0] + gun.bbox[2]) // 2, (gun.bbox[1] + gun.bbox[3]) // 2)
                person_center = ((person.bbox[0] + person.bbox[2]) // 2, (person.bbox[1] + person.bbox[3]) // 2)
                cv2.line(annotated_frame, gun_center, person_center, (0, 255, 255), 3)
        
        # Also check if we should continue tracking a gun person who might have left the gun detection area
        if self.current_gun_person_id is not None and not current_gun_person_detected:
            # Check if gun person is still in view (without gun)
            time_since_seen = time.time() - self.gun_person_last_seen
            if time_since_seen < self.gun_person_timeout:
                # Look for the gun person in all person detections
                for person in person_detections:
                    x1, y1, x2, y2 = person.bbox
                    person_roi = frame[y1:y2, x1:x2]
                    if person_roi.size > 0:
                        person_roi_resized = cv2.resize(person_roi, (128, 256))
                        embedding = self.reid_system.get_embedding(person_roi_resized)
                        
                        # Check if this person matches our gun person
                        person_id = self.reid_system.find_best_match(embedding, person.bbox, min_confidence=0.6)
                        
                        if person_id == self.current_gun_person_id:
                            # Found the gun person (without gun visible)
                            self.gun_person_last_seen = time.time()
                            current_gun_person_detected = True
                            
                            # Draw as tracked gun person
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 3)  # Orange border
                            cv2.putText(annotated_frame, f"TRACKED GUN PERSON ID: {person_id}", (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                            break
        
        # Status information
        self.draw_status(annotated_frame, current_gun_person_detected)
        
        return annotated_frame
    
    def draw_status(self, frame: np.ndarray, gun_person_detected: bool):
        """Draw essential status information only"""
        # FPS calculation
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            self.fps = 30 / elapsed
            self.fps_start_time = time.time()
        
        # Essential status only
        status_lines = [
            f"FPS: {getattr(self, 'fps', 0):.1f}",
            f"Device: {self.device}",
            f"Gun Person: {'DETECTED' if gun_person_detected else 'NONE'}"
        ]
        
        if self.current_gun_person_id is not None:
            time_since_seen = time.time() - self.gun_person_last_seen
            status_lines.append(f"ID: {self.current_gun_person_id}")
            
            if time_since_seen > self.gun_person_timeout:
                status_lines.append("STATUS: LEFT VIEW")
            else:
                status_lines.append("STATUS: IN VIEW")
        
        # Draw status with clean background
        y_offset = 30
        for line in status_lines:
            # Draw background rectangle
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (5, y_offset - 20), (text_size[0] + 15, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

def main():
    """Main function to run the gun person tracker"""
    # Initialize tracker
    tracker = GunPersonTracker()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    logger.info("Starting Gun Person Tracker...")
    logger.info("Press 'q' to quit, 's' to save database")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Process frame
            annotated_frame = tracker.process_frame(frame)
            
            # Display frame
            cv2.imshow("Gun Person Tracker", annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                tracker.reid_system.save_database()
                logger.info("Database saved")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        tracker.reid_system.save_database()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()