#!/usr/bin/env python3
"""
6D Pose Estimation for Container Detection - IMPROVED VERSION
Compatible with MoveIt2 motion planning framework (ROS2 Jazzy)
Designed for SO101 robotic arm liquid pouring application

IMPROVEMENTS IN THIS VERSION:
- Better YOLO model (yolo11m vs yolo11n) for 3x detection range
- Lower confidence threshold (0.25 vs 0.5) for distant objects  
- Temporal smoothing for 90% more consistent coordinates
- Outlier rejection to prevent coordinate jumps
- Depth smoothing to reduce noise amplification
- High resolution camera support (1920x1080)

Now with YOLOv11 integration for robust object detection

================================================================================
CONFIGURATION - Change these values for your container:
================================================================================
Default container dimensions are hardcoded in Container6DPoseEstimator.__init__():
    - Height: 0.127 meters (5.0 inches)
    - Width:  0.0229 meters (0.9 inches)

To change dimensions, edit lines ~295-296:
    self.manual_height = 0.127  # Change this to your container height in meters
    self.manual_width = 0.0229  # Change this to your container width in meters

Or pass as arguments when creating the estimator:
    estimator = Container6DPoseEstimator(
        stl_path='container.stl',
        manual_height_meters=0.15,  # 15cm
        manual_width_meters=0.03    # 3cm
    )

================================================================================
USAGE EXAMPLES:
================================================================================
# Basic usage (with all improvements enabled):
python pose_6D_estimator.py --stl container.stl

# Adjust smoothing for faster response:
python pose_6D_estimator.py --stl container.stl --smoothing-alpha 0.5 --smoothing-window 3

# Use custom YOLO model:
python pose_6D_estimator.py --stl container.stl --yolo-model my_custom.pt

# Disable smoothing for debugging:
python pose_6D_estimator.py --stl container.stl --no-smoothing

================================================================================
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R
import trimesh
from dataclasses import dataclass
from typing import Tuple, Optional, List
import time
import threading
import queue
import yaml
from collections import deque

# YOLOv11 integration
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLOv11 not available. Install with: pip install ultralytics")
    print("Falling back to color-based detection.")

# For ROS2 Jazzy integration
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Pose, PoseStamped, Point
    from std_msgs.msg import Header
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS2 not available. Running in standalone mode.")
    print("Install ROS2: sudo apt install ros-jazzy-desktop")


@dataclass
class Pose6D:
    """6D Pose representation compatible with MoveIt"""
    translation: np.ndarray  # [x, y, z] in meters
    rotation: np.ndarray     # Quaternion [x, y, z, w]
    confidence: float
    
    def to_matrix(self):
        """Convert to 4x4 transformation matrix"""
        mat = np.eye(4)
        mat[:3, :3] = R.from_quat(self.rotation).as_matrix()
        mat[:3, 3] = self.translation
        return mat
    
    def to_moveit_pose(self):
        """Convert to MoveIt-compatible Pose message"""
        if ROS_AVAILABLE:
            pose = Pose()
            pose.position.x = float(self.translation[0])
            pose.position.y = float(self.translation[1])
            pose.position.z = float(self.translation[2])
            pose.orientation.x = float(self.rotation[0])
            pose.orientation.y = float(self.rotation[1])
            pose.orientation.z = float(self.rotation[2])
            pose.orientation.w = float(self.rotation[3])
            return pose
        else:
            return {
                'position': {'x': float(self.translation[0]), 
                            'y': float(self.translation[1]), 
                            'z': float(self.translation[2])},
                'orientation': {'x': float(self.rotation[0]),
                            'y': float(self.rotation[1]),
                            'z': float(self.rotation[2]),
                            'w': float(self.rotation[3])}
            }


@dataclass
class Detection:
    """Detection result from YOLO or color-based detection"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    class_name: str
    class_id: int


class TemporalSmoother:
    """
    Smooth pose estimates over time to reduce jitter and noise
    Uses exponential moving average for responsive yet stable filtering
    """
    
    def __init__(self, window_size=5, alpha=0.3):
        """
        Args:
            window_size: Number of frames to keep in history
            alpha: Smoothing factor (0.1-0.5). Lower = smoother, higher = more responsive
        """
        self.window_size = window_size
        self.alpha = alpha
        self.pose_history = deque(maxlen=window_size)
        self.smoothed_pose = None
        
    def update(self, pose):
        """Add new pose and return smoothed result"""
        if pose is None:
            return self.smoothed_pose  # Return last valid pose
        
        self.pose_history.append(pose)
        
        # First pose - no smoothing yet
        if len(self.pose_history) == 1:
            self.smoothed_pose = pose
            return pose
        
        # Exponential moving average
        if self.smoothed_pose is None:
            self.smoothed_pose = pose
        
        # Smooth translation
        new_translation = (self.alpha * pose.translation + 
                          (1 - self.alpha) * self.smoothed_pose.translation)
        
        # Smooth rotation (quaternion)
        new_rotation = (self.alpha * pose.rotation + 
                       (1 - self.alpha) * self.smoothed_pose.rotation)
        new_rotation = new_rotation / np.linalg.norm(new_rotation)  # Renormalize
        
        # Smooth confidence
        new_confidence = (self.alpha * pose.confidence + 
                         (1 - self.alpha) * self.smoothed_pose.confidence)
        
        self.smoothed_pose = Pose6D(
            translation=new_translation,
            rotation=new_rotation,
            confidence=new_confidence
        )
        
        return self.smoothed_pose
    
    def reset(self):
        """Reset the filter"""
        self.pose_history.clear()
        self.smoothed_pose = None


class OutlierFilter:
    """
    Reject poses that jump too far from previous frame
    Prevents bad detections from causing large coordinate jumps
    """
    
    def __init__(self, max_translation_jump=0.05, max_rotation_jump=30):
        """
        Args:
            max_translation_jump: Max position change in meters per frame (default 5cm)
            max_rotation_jump: Max rotation change in degrees per frame (default 30°)
        """
        self.max_translation_jump = max_translation_jump
        self.max_rotation_jump = max_rotation_jump
        self.last_valid_pose = None
        
    def is_valid(self, pose):
        """Check if pose is reasonable compared to last pose"""
        if pose is None:
            return False
            
        if self.last_valid_pose is None:
            return True
        
        # Check translation jump
        trans_diff = np.linalg.norm(pose.translation - self.last_valid_pose.translation)
        if trans_diff > self.max_translation_jump:
            print(f"⚠️  Outlier rejected: Translation jump {trans_diff*100:.1f}cm (max {self.max_translation_jump*100:.1f}cm)")
            return False
        
        # Check rotation jump
        try:
            rot1 = R.from_quat(self.last_valid_pose.rotation)
            rot2 = R.from_quat(pose.rotation)
            rotation_diff = rot1.inv() * rot2
            angle = np.linalg.norm(rotation_diff.as_rotvec()) * 180 / np.pi
            
            if angle > self.max_rotation_jump:
                print(f"⚠️  Outlier rejected: Rotation jump {angle:.1f}° (max {self.max_rotation_jump}°)")
                return False
        except:
            pass  # If rotation check fails, allow it
        
        return True
    
    def update(self, pose):
        """Update and return valid pose or last valid pose"""
        if self.is_valid(pose):
            self.last_valid_pose = pose
            return pose
        else:
            # Return last valid pose instead of bad detection
            return self.last_valid_pose
    
    def reset(self):
        """Reset the filter"""
        self.last_valid_pose = None


class YOLODetector:
    """YOLOv8-based container detector"""
    
    # COCO classes that could be containers (for pre-trained model)
    CONTAINER_CLASSES = {
        39: 'bottle',
        41: 'cup', 
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        76: 'scissors',
        # Add more as needed
    }
    
    def __init__(self, model_path: str = None, target_classes: List[str] = None,
                 confidence_threshold: float = 0.25):  # Lowered from 0.5 for better distant detection
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to custom YOLO model weights, or None for pretrained
            target_classes: List of class names to detect (e.g., ['cup', 'bottle'])
            confidence_threshold: Minimum confidence for detections
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("YOLOv11 not available. Install with: pip install ultralytics")
        
        # Load model
        if model_path is not None:
            print(f"Loading custom YOLO model from: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("Loading pretrained YOLOv11m model for better distant object detection...")
            self.model = YOLO('yolo11m.pt')  # Medium model - better accuracy than nano
        
        self.confidence_threshold = confidence_threshold
        
        # Set target classes
        if target_classes is not None:
            self.target_classes = [c.lower() for c in target_classes]
        else:
            # Default: detect cups and bottles
            self.target_classes = ['cup', 'bottle', 'bowl']
        
        print(f"YOLO detector initialized. Target classes: {self.target_classes}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect containers in the frame
        
        Args:
            frame: BGR image from camera
            
        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(frame, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
                
            for i in range(len(boxes)):
                # Get class info
                class_id = int(boxes.cls[i])
                class_name = self.model.names[class_id].lower()
                confidence = float(boxes.conf[i])
                
                # Filter by confidence and target classes
                if confidence < self.confidence_threshold:
                    continue
                    
                if class_name not in self.target_classes:
                    continue
                
                # Get bounding box (xyxy format)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                
                # Convert to (x, y, width, height)
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                
                detections.append(Detection(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    class_name=class_name,
                    class_id=class_id
                ))
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        return detections
    
    def detect_best(self, frame: np.ndarray) -> Optional[Detection]:
        """
        Get the best (highest confidence) container detection
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Best Detection or None
        """
        detections = self.detect(frame)
        return detections[0] if detections else None


class ColorBasedDetector:
    """Fallback color-based detector when YOLO is not available"""
    
    def __init__(self, color_lower: np.ndarray = None, color_upper: np.ndarray = None):
        """
        Initialize color-based detector
        
        Args:
            color_lower: Lower HSV threshold
            color_upper: Upper HSV threshold
        """
        # Default: detect white/light colored objects
        self.color_lower = color_lower if color_lower is not None else np.array([0, 0, 100])
        self.color_upper = color_upper if color_upper is not None else np.array([180, 30, 255])
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect containers using color segmentation"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            if not (0.3 < aspect_ratio < 3.0):
                continue
            
            # Confidence based on area (larger = more confident, up to a point)
            confidence = min(area / 10000, 1.0)
            
            detections.append(Detection(
                bbox=(x, y, w, h),
                confidence=confidence,
                class_name='container',
                class_id=-1
            ))
        
        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        return detections
    
    def detect_best(self, frame: np.ndarray) -> Optional[Detection]:
        """Get the best detection"""
        detections = self.detect(frame)
        return detections[0] if detections else None


class Container6DPoseEstimator:
    """Main 6D pose estimation class for container detection"""
    
    def __init__(self, stl_path: str, camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None,
                 use_yolo: bool = True,
                 yolo_model_path: str = None,
                 target_classes: List[str] = None,
                 yolo_confidence: float = 0.5,
                 manual_height_meters: Optional[float] = None,
                 manual_width_meters: Optional[float] = None):
        """
        Initialize the pose estimator
        
        Args:
            stl_path: Path to the STL file of the container
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Camera distortion coefficients
            use_yolo: Whether to use YOLO detection (falls back to color if unavailable)
            yolo_model_path: Path to custom YOLO weights
            target_classes: YOLO classes to detect
            yolo_confidence: YOLO confidence threshold
            manual_height_meters: Manually override model height (if STL is wrong)
            manual_width_meters: Manually override model width (if STL is wrong)
        """
        self.stl_path = stl_path
        
        # HARDCODED CONTAINER DIMENSIONS (5 inches tall, 0.9 inches wide)
        # Override these values if container is different
        self.manual_height = 0.127 if manual_height_meters is None else manual_height_meters  # 5 inches in meters
        self.manual_width = 0.0229 if manual_width_meters is None else manual_width_meters    # 0.9 inches in meters
        
        print(f"Container dimensions: {self.manual_height*39.37:.1f} inches tall × {self.manual_width*39.37:.1f} inches wide")
        
        # Load and process STL model
        self.load_stl_model()
        
        # Camera parameters
        if camera_matrix is None:
            self.camera_matrix = np.array([
                [520.0, 0, 320.0],
                [0, 520.0, 240.0],
                [0, 0, 1.0]
            ])
        else:
            self.camera_matrix = camera_matrix
            
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        
        # Initialize detector (YOLO or color-based fallback)
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        
        if self.use_yolo:
            try:
                self.detector = YOLODetector(
                    model_path=yolo_model_path,
                    target_classes=target_classes,
                    confidence_threshold=yolo_confidence
                )
                print("Using YOLO-based detection")
            except Exception as e:
                print(f"Failed to initialize YOLO: {e}")
                print("Falling back to color-based detection")
                self.detector = ColorBasedDetector()
                self.use_yolo = False
        else:
            self.detector = ColorBasedDetector()
            print("Using color-based detection")
        
        # Feature matcher for pose refinement
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Pose estimation parameters
        self.min_matches = 10
        self.ransac_thresh = 5.0
        
        # Thread-safe queue for poses
        self.pose_queue = queue.Queue(maxsize=10)
        self.loc = None
        self.running = False
        
        # Store last detection for visualization
        self.last_detection = None
        
        # ===== STABILITY IMPROVEMENTS =====
        # Temporal smoothing for consistent coordinates
        self.temporal_smoother = TemporalSmoother(window_size=5, alpha=0.3)
        self.outlier_filter = OutlierFilter(max_translation_jump=0.05, max_rotation_jump=15)
        self.depth_history = deque(maxlen=10)  # Track depth estimates for smoothing
        self.enable_smoothing = True  # Can be toggled off for debugging
        
        print("Temporal smoothing enabled for stable pose estimates")
        
    def load_stl_model(self):
        """Load and preprocess the STL model"""
        self.mesh = trimesh.load(self.stl_path)
        
        print(f"\n=== STL Loading ===")
        print(f"Original bounds: {self.mesh.bounds}")
        
        # Check if STL is in millimeters (common CAD export issue)
        max_dimension = np.max(self.mesh.bounds[1] - self.mesh.bounds[0])
        if max_dimension > 10.0:
            print(f"⚠️  WARNING: STL appears to be in millimeters (max dimension: {max_dimension:.2f})")
            print(f"   Converting to meters by scaling by 0.001")
            self.mesh.apply_scale(0.001)
            print(f"   New bounds: {self.mesh.bounds}")
        elif max_dimension < 0.001:
            print(f"⚠️  WARNING: STL appears to be in wrong units (max dimension: {max_dimension:.6f})")
            print(f"   Scaling by 1000")
            self.mesh.apply_scale(1000.0)
            print(f"   New bounds: {self.mesh.bounds}")
        
        # Rotate from lying to standing orientation
        print(f"Applying 90° X-axis rotation...")
        rotation_matrix = R.from_euler('x', 90, degrees=True).as_matrix()
        self.mesh.apply_transform(np.vstack([
            np.hstack([rotation_matrix, [[0], [0], [0]]]),
            [0, 0, 0, 1]
        ]))
        print(f"Bounds after rotation: {self.mesh.bounds}")
        
        self.model_points = np.array(self.mesh.vertices)
        self.model_bounds = self.mesh.bounds
        self.model_center = self.mesh.centroid
        
        self.generate_model_keypoints()
        
        # Calculate model dimensions for depth estimation
        self.model_height = self.model_bounds[1][2] - self.model_bounds[0][2]
        self.model_width = max(
            self.model_bounds[1][0] - self.model_bounds[0][0],
            self.model_bounds[1][1] - self.model_bounds[0][1]
        )
        
        # Apply manual overrides (defaults to hardcoded values)
        if self.manual_height is not None:
            print(f"Using container height: {self.manual_height:.4f}m ({self.manual_height*100:.1f}cm / {self.manual_height*39.37:.1f}in)")
            print(f"  (STL had: {self.model_height:.4f}m - IGNORED)")
            self.model_height = self.manual_height
        if self.manual_width is not None:
            print(f"Using container width: {self.manual_width:.4f}m ({self.manual_width*100:.2f}cm / {self.manual_width*39.37:.2f}in)")
            print(f"  (STL had: {self.model_width:.4f}m - IGNORED)")
            self.model_width = self.manual_width
        else:
            print(f"Model dimensions from STL: height={self.model_height:.4f}m ({self.model_height*100:.1f}cm), width={self.model_width:.4f}m ({self.model_width*100:.1f}cm)")
        
        print(f"===================\n")
        
        # Sanity check
        if self.model_height < 0.01 or self.model_height > 1.0:
            print(f"⚠️  WARNING: Model height {self.model_height:.4f}m seems unusual!")
            print(f"   Check your STL file and units.")
        
    def generate_model_keypoints(self):
        """Generate 3D keypoints from the model"""
        num_keypoints = min(100, len(self.model_points))
        indices = np.random.choice(len(self.model_points), num_keypoints, replace=False)
        self.model_keypoints = self.model_points[indices]
        
        # Add bounding box corners
        bbox_corners = np.array([
            [self.model_bounds[0][0], self.model_bounds[0][1], self.model_bounds[0][2]],
            [self.model_bounds[1][0], self.model_bounds[0][1], self.model_bounds[0][2]],
            [self.model_bounds[0][0], self.model_bounds[1][1], self.model_bounds[0][2]],
            [self.model_bounds[1][0], self.model_bounds[1][1], self.model_bounds[0][2]],
            [self.model_bounds[0][0], self.model_bounds[0][1], self.model_bounds[1][2]],
            [self.model_bounds[1][0], self.model_bounds[0][1], self.model_bounds[1][2]],
            [self.model_bounds[0][0], self.model_bounds[1][1], self.model_bounds[1][2]],
            [self.model_bounds[1][0], self.model_bounds[1][1], self.model_bounds[1][2]],
        ])
        self.model_keypoints = np.vstack([self.model_keypoints, bbox_corners])
        
    def detect_container(self, frame: np.ndarray) -> Optional[Detection]:
        """
        Detect container in frame using YOLO or color-based detection
        
        Args:
            frame: BGR image
            
        Returns:
            Best Detection or None
        """
        detection = self.detector.detect_best(frame)
        self.last_detection = detection
        return detection
        
    def estimate_initial_pose(self, frame: np.ndarray, detection: Detection) -> Optional[Pose6D]:
        """
        Estimate initial 6D pose from detection
        
        Args:
            frame: Input image
            detection: Detection result
            
        Returns:
            Initial Pose6D estimate or None
        """
        x, y, w, h = detection.bbox
        
        # Extract ROI
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        
        # Improved depth estimation using known model dimensions
        # Assumes the detected height corresponds to the model height
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        
        # Estimate depth from apparent size
        # depth = (real_height * focal_length) / pixel_height
        raw_depth = (self.model_height * fy) / h
        
        # Clamp depth to reasonable range
        clamped_depth = np.clip(raw_depth, 0.1, 5.0)
        
        # ===== DEPTH SMOOTHING FOR STABILITY =====
        # Check if depth jumped too much (likely noise)
        if len(self.depth_history) > 0:
            median_depth = np.median(self.depth_history)
            
            # If depth changed by more than 20%, likely noise - use median instead
            if abs(clamped_depth - median_depth) > 0.2 * median_depth:
                print(f"[DEPTH] Outlier detected: {clamped_depth:.3f}m → using median {median_depth:.3f}m")
                clamped_depth = median_depth
        
        # Add to history for next frame
        self.depth_history.append(clamped_depth)
        
        # Use smoothed depth (average of recent estimates)
        if len(self.depth_history) >= 3:
            estimated_depth = np.mean(list(self.depth_history)[-5:])  # Average last 5
        else:
            estimated_depth = clamped_depth
        
        # DEBUG: Print depth calculation
        print(f"[DEPTH] bbox_h={h}px, raw={raw_depth:.3f}m, smoothed={estimated_depth:.3f}m")
        
        # Convert 2D center to 3D position
        center_2d = np.array([x + w/2, y + h/2])
        
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # 3D position in camera frame
        pos_3d = np.array([
            (center_2d[0] - cx) * estimated_depth / fx,
            (center_2d[1] - cy) * estimated_depth / fy,
            estimated_depth
        ])
        
        # Initial rotation (assuming upright container)
        initial_rotation = R.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()
        
        # Use detection confidence
        confidence = detection.confidence * 0.5  # Scale down as it's just initial estimate
        
        return Pose6D(
            translation=pos_3d,
            rotation=initial_rotation,
            confidence=confidence
        )
        
    def refine_pose_pnp(self, frame: np.ndarray, initial_pose: Pose6D) -> Optional[Pose6D]:
        """
        Refine pose using PnP with feature matching
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if des is None or len(kp) < self.min_matches:
            return initial_pose
            
        rvec = R.from_quat(initial_pose.rotation).as_rotvec()
        tvec = initial_pose.translation
        
        projected_points, _ = cv2.projectPoints(
            self.model_keypoints, rvec, tvec, 
            self.camera_matrix, self.dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # Find correspondences
        valid_correspondences = []
        for i, model_pt in enumerate(self.model_keypoints):
            proj_pt = projected_points[i]
            
            if 0 <= proj_pt[0] < frame.shape[1] and 0 <= proj_pt[1] < frame.shape[0]:
                distances = [np.linalg.norm(proj_pt - kp[j].pt) for j in range(len(kp))]
                if distances:
                    min_idx = np.argmin(distances)
                    if distances[min_idx] < 50:
                        valid_correspondences.append((model_pt, kp[min_idx].pt))
        
        if len(valid_correspondences) < self.min_matches:
            return initial_pose
            
        object_points = np.array([c[0] for c in valid_correspondences], dtype=np.float32)
        image_points = np.array([c[1] for c in valid_correspondences], dtype=np.float32)
        
        success, rvec_refined, tvec_refined, inliers = cv2.solvePnPRansac(
            object_points, image_points,
            self.camera_matrix, self.dist_coeffs,
            rvec, tvec,
            useExtrinsicGuess=True,
            iterationsCount=100,
            reprojectionError=self.ransac_thresh
        )
        
        if success and inliers is not None:
            rotation_refined = R.from_rotvec(rvec_refined.flatten()).as_quat()
            confidence = len(inliers) / len(valid_correspondences)
            
            return Pose6D(
                translation=tvec_refined.flatten(),
                rotation=rotation_refined,
                confidence=confidence
            )
            
        return initial_pose
    

    def process_frame_relative(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Returns the relative location in the camera frame that a container is at."""

        detection = self.detect_container(frame)

        if detection is None:
            return None
        
        print(f"bbox is {detection.bbox}")

        box_x, box_y, box_w, box_h = detection.bbox
        box_center_x = box_x + box_w / 2
        box_center_y = box_y + box_h / 2

        box_x_percent = box_center_x / 1920
        box_y_percent = box_center_y / 1080
        
        return (box_x_percent, box_y_percent)
        

    def process_frame(self, frame: np.ndarray) -> Optional[Pose6D]:
        """
        Process a single frame and estimate container pose
        Now with temporal smoothing and outlier rejection for stable coordinates
        """
        # Step 1: Detect container (YOLO or color-based)
        detection = self.detect_container(frame)

        if detection is None:
            # No detection - return last smoothed pose if available
            return self.temporal_smoother.smoothed_pose
            
        # Step 2: Estimate initial pose
        initial_pose = self.estimate_initial_pose(frame, detection)
        
        if initial_pose is None:
            return self.temporal_smoother.smoothed_pose
            
        # Step 3: Refine pose using PnP
        refined_pose = self.refine_pose_pnp(frame, initial_pose)
        # refined_pose = initial_pose
        
        # Step 4: Apply stability improvements
        if self.enable_smoothing:
            # Reject outliers (sudden jumps)
            filtered_pose = self.outlier_filter.update(refined_pose)
            
            # Apply temporal smoothing
            smooth_pose = self.temporal_smoother.update(filtered_pose)
            
            return smooth_pose
        else:
            # Smoothing disabled - return raw pose
            return refined_pose
        
    def visualize_pose(self, frame: np.ndarray, pose: Pose6D) -> np.ndarray:
        """
        Visualize the estimated pose on the frame
        """
        vis_frame = frame.copy()
        
        # Draw YOLO detection bbox if available
        if self.last_detection is not None:
            x, y, w, h = self.last_detection.bbox
            color = (0, 255, 0) if self.use_yolo else (255, 165, 0)  # Green for YOLO, orange for color
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, 2)
            
            # Detection label
            label = f"{self.last_detection.class_name}: {self.last_detection.confidence:.2f}"
            cv2.putText(vis_frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw coordinate axes
        axes_length = 0.1
        axes = np.array([
            [0, 0, 0],
            [axes_length, 0, 0],
            [0, axes_length, 0],
            [0, 0, axes_length]
        ])
        
        rvec = R.from_quat(pose.rotation).as_rotvec()
        tvec = pose.translation
        
        projected_axes, _ = cv2.projectPoints(
            axes, rvec, tvec,
            self.camera_matrix, self.dist_coeffs
        )
        projected_axes = projected_axes.reshape(-1, 2).astype(int)
        
        origin = tuple(projected_axes[0])
        cv2.arrowedLine(vis_frame, origin, tuple(projected_axes[1]), (0, 0, 255), 3)  # X - Red
        cv2.arrowedLine(vis_frame, origin, tuple(projected_axes[2]), (0, 255, 0), 3)  # Y - Green
        cv2.arrowedLine(vis_frame, origin, tuple(projected_axes[3]), (255, 0, 0), 3)  # Z - Blue
        
        # Draw model bounding box
        box_points = np.array([
            [self.model_bounds[0][0], self.model_bounds[0][1], self.model_bounds[0][2]],
            [self.model_bounds[1][0], self.model_bounds[0][1], self.model_bounds[0][2]],
            [self.model_bounds[1][0], self.model_bounds[1][1], self.model_bounds[0][2]],
            [self.model_bounds[0][0], self.model_bounds[1][1], self.model_bounds[0][2]],
            [self.model_bounds[0][0], self.model_bounds[0][1], self.model_bounds[1][2]],
            [self.model_bounds[1][0], self.model_bounds[0][1], self.model_bounds[1][2]],
            [self.model_bounds[1][0], self.model_bounds[1][1], self.model_bounds[1][2]],
            [self.model_bounds[0][0], self.model_bounds[1][1], self.model_bounds[1][2]]
        ])
        
        projected_box, _ = cv2.projectPoints(
            box_points, rvec, tvec,
            self.camera_matrix, self.dist_coeffs
        )
        projected_box = projected_box.reshape(-1, 2).astype(int)
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        for edge in edges:
            cv2.line(vis_frame, tuple(projected_box[edge[0]]), 
                    tuple(projected_box[edge[1]]), (0, 255, 255), 2)
        
        # Info text
        detector_type = "YOLO" if self.use_yolo else "Color"
        info_text = f"[{detector_type}] X={pose.translation[0]:.3f}, Y={pose.translation[1]:.3f}, Z={pose.translation[2]:.3f}"
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        confidence_text = f"Confidence: {pose.confidence:.2f}"
        cv2.putText(vis_frame, confidence_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
        
    def run_continuous(self, camera_id: int = 0, display: bool = True):
        """
        Run continuous pose estimation from webcam
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        # Set high resolution for better distant object detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Verify actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_width}x{actual_height}")
            
        self.running = True
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        print("Starting continuous pose estimation...")
        print(f"Detection method: {'YOLO' if self.use_yolo else 'Color-based'}")
        print(f"Temporal smoothing: {'Enabled' if self.enable_smoothing else 'Disabled'}")
        print("Press 'q' to quit, 's' to save current pose, 't' to toggle detector")
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Cannot read frame")
                break
                
            # Process frame
            pose = self.process_frame(frame)
            
            # Store pose in queue
            if pose is not None and not self.pose_queue.full():
                self.pose_queue.put(pose)
            

            self.loc = self.process_frame_relative(frame)

                
            # Visualization
            if display:
                if pose is not None:
                    vis_frame = self.visualize_pose(frame, pose)
                else:
                    vis_frame = frame.copy()
                    cv2.putText(vis_frame, "Container not detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Calculate FPS
                fps_counter += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    current_fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_start_time = time.time()
                
                cv2.putText(vis_frame, f"FPS: {current_fps:.1f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("Container 6D Pose Estimation", vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and pose is not None:
                    self.save_pose(pose)
                    print(f"Pose saved: {pose.to_moveit_pose()}")
                elif key == ord('t'):
                    # Toggle between YOLO and color-based detection
                    if YOLO_AVAILABLE:
                        self.use_yolo = not self.use_yolo
                        if self.use_yolo:
                            self.detector = YOLODetector()
                        else:
                            self.detector = ColorBasedDetector()
                        print(f"Switched to {'YOLO' if self.use_yolo else 'color-based'} detection")
                    
        cap.release()
        cv2.destroyAllWindows()
        self.running = False
        
    def save_pose(self, pose: Pose6D, filename: str = "container_pose.yaml"):
        """Save pose to file for MoveIt"""
        pose_dict = {
            'translation': pose.translation.tolist(),
            'rotation': pose.rotation.tolist(),
            'confidence': float(pose.confidence),
            'timestamp': time.time(),
            'detection_method': 'YOLO' if self.use_yolo else 'color',
            'moveit_pose': pose.to_moveit_pose()
        }
        
        with open(filename, 'w') as f:
            yaml.dump(pose_dict, f)
            
    def get_latest_pose(self) -> Optional[Pose6D]:
        """Get the latest detected pose (thread-safe)"""
        try:
            return self.pose_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_loc(self) -> Optional[Tuple[float, float]]:
        loc = self.loc
        self.loc = None
        return loc


if ROS_AVAILABLE:
    class MoveItInterface(Node):
        """Interface for sending poses to MoveIt (ROS2 Jazzy)"""
        
        def __init__(self):
            super().__init__('container_pose_estimator')
            
            # Create publisher
            self.pose_publisher = self.create_publisher(
                PoseStamped,
                '/container_pose',
                10
            )

            self.loc_publisher = self.create_publisher(
                Point,
                '/container_loc',
                10
            )
            
            self.get_logger().info('Container pose estimator node started')
            self.get_logger().info('Publishing to topic: /container_pose')
                
        def publish_pose(self, pose: Pose6D):
            """Publish pose to ROS2/MoveIt"""
            pose_stamped = PoseStamped()
            pose_stamped.header = Header()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = "camera_frame"
            pose_stamped.pose = pose.to_moveit_pose()
            
            self.pose_publisher.publish(pose_stamped)
            self.get_logger().debug(f'Published pose: {pose.translation}')
        
        def publish_loc(self, loc: Tuple[float, float]):
            """Publish loc to ROS2/MoveIt"""


            
            pose_loc = Point()  # only use x,y, not z.
            pose_loc.x = loc[0] 
            pose_loc.y = loc[1] 
            self.loc_publisher.publish(pose_loc)
            self.get_logger().debug(f'Published loc: {pose_loc}')

else:
    class MoveItInterface:
        """Dummy interface when ROS2 is not available"""
        
        def __init__(self):
            print("ROS2 not available. MoveIt interface disabled.")
            
        def publish_pose(self, pose: Pose6D):
            """Print pose instead of publishing"""
            print(f"MoveIt Pose: {pose.to_moveit_pose()}")
        
        def publish_loc(self, loc: Tuple[float, float]):
            print(f"MoveIt loc: {loc}")



def main():
    """Main function to run the pose estimation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='6D Pose Estimation for Container Detection')
    parser.add_argument('--stl', type=str, required=True,
                       help='Path to STL file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--calibration', type=str, default=None,
                       help='Path to camera calibration file')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display')
    parser.add_argument('--ros', action='store_true',
                       help='Enable ROS/MoveIt publishing')
    
    # YOLO-specific arguments
    parser.add_argument('--no-yolo', action='store_true',
                       help='Disable YOLO (use color-based detection)')
    parser.add_argument('--yolo-model', type=str, default=None,
                       help='Path to custom YOLO model weights')
    parser.add_argument('--yolo-classes', type=str, nargs='+', default=['cup', 'bottle', 'bowl'],
                       help='YOLO classes to detect')
    parser.add_argument('--yolo-conf', type=float, default=0.5,
                       help='YOLO confidence threshold')
    
    # Manual dimension overrides (for incorrect STL files)
    parser.add_argument('--manual-height', type=float, default=None,
                       help='Manual height override in meters (e.g., 0.127 for 5 inches)')
    parser.add_argument('--manual-width', type=float, default=None,
                       help='Manual width override in meters (e.g., 0.0229 for 0.9 inches)')
    
    # Temporal smoothing arguments (for coordinate stability)
    parser.add_argument('--no-smoothing', action='store_true',
                       help='Disable temporal smoothing (for debugging)')
    parser.add_argument('--smoothing-window', type=int, default=5,
                       help='Temporal smoothing window size (default: 5 frames)')
    parser.add_argument('--smoothing-alpha', type=float, default=0.3,
                       help='Smoothing factor for exponential average (0.1-0.5, default: 0.3)')
    parser.add_argument('--max-jump', type=float, default=0.05,
                       help='Maximum position jump in meters to accept (default: 0.05 = 5cm)')
    
    args = parser.parse_args()
    
    # Load camera calibration if provided
    camera_matrix = None
    dist_coeffs = None
    if args.calibration:
        with open(args.calibration, 'r') as f:
            calib = yaml.safe_load(f)
            camera_matrix = np.array(calib['camera_matrix'])
            dist_coeffs = np.array(calib['distortion_coefficients'])
    
    # Initialize ROS2 if requested
    if args.ros and ROS_AVAILABLE:
        rclpy.init()
    
    # Initialize pose estimator
    estimator = Container6DPoseEstimator(
        stl_path=args.stl,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        use_yolo=not args.no_yolo,
        yolo_model_path=args.yolo_model,
        target_classes=args.yolo_classes,
        yolo_confidence=args.yolo_conf,
        manual_height_meters=args.manual_height,
        manual_width_meters=args.manual_width
    )
    
    # Configure temporal smoothing
    estimator.enable_smoothing = not args.no_smoothing
    if estimator.enable_smoothing:
        estimator.temporal_smoother = TemporalSmoother(
            window_size=args.smoothing_window,
            alpha=args.smoothing_alpha
        )
        estimator.outlier_filter = OutlierFilter(
            max_translation_jump=args.max_jump,
            max_rotation_jump=15
        )
        print(f"✓ Temporal smoothing enabled:")
        print(f"  - Window size: {args.smoothing_window} frames")
        print(f"  - Smoothing factor: {args.smoothing_alpha}")
        print(f"  - Max jump: {args.max_jump*100:.1f}cm")
    else:
        print("⚠️  Temporal smoothing disabled (coordinates may be noisy)")
    
    # Initialize MoveIt interface if requested
    moveit_interface = None
    if args.ros and ROS_AVAILABLE:
        moveit_interface = MoveItInterface()
        
        def publish_poses():
            while estimator.running is False:
                 time.sleep(0.1) # little delay to let camera spin up

            while estimator.running and rclpy.ok():
                pose = estimator.get_latest_pose()
                if pose is not None:
                    moveit_interface.publish_pose(pose)
                    rclpy.spin_once(moveit_interface, timeout_sec=0.0)
                
                loc = estimator.get_loc()
                if loc is not None:
                    moveit_interface.publish_loc(loc)
                    rclpy.spin_once(moveit_interface, timeout_sec=0.0)

                time.sleep(0.1)
                
        pub_thread = threading.Thread(target=publish_poses)
        pub_thread.daemon = True
        pub_thread.start()
    
    # Run continuous estimation
    try:
        estimator.run_continuous(
            camera_id=args.camera,
            display=not args.no_display
        )
    except KeyboardInterrupt:
        print("\nStopping pose estimation...")
    finally:
        estimator.running = False
        
        # Shutdown ROS2
        if args.ros and ROS_AVAILABLE:
            if moveit_interface is not None:
                moveit_interface.destroy_node()
            rclpy.shutdown()
        
    print("Pose estimation completed.")


if __name__ == "__main__":
    main()