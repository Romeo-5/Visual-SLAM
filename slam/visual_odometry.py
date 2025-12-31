import cv2
import numpy as np
from core.camera import Camera
from core.frame import Frame
from core.feature import FeatureExtractor
from core.pose_estimator import PoseEstimator

class VisualOdometry:
    def __init__(self, camera_matrix):
        """
        Simple Visual Odometry system
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
        """
        self.camera = Camera(camera_matrix)
        self.feature_extractor = FeatureExtractor(n_features=2000)
        self.pose_estimator = PoseEstimator(camera_matrix)
        
        self.frames = []
        self.current_pose = np.eye(4)  # Start at origin
        
    def process_frame(self, image):
        """
        Process a new frame and estimate camera motion
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            success: Boolean indicating if pose was estimated
        """
        # Create frame
        frame = Frame(image, self.camera)
        
        # Extract features
        frame.keypoints, frame.descriptors = self.feature_extractor.detect_and_compute(image)
        
        # First frame - just store it
        if len(self.frames) == 0:
            frame.set_pose(np.eye(3), np.zeros((3, 1)))
            self.frames.append(frame)
            return True
            
        # Match with previous frame
        prev_frame = self.frames[-1]
        matches = self.feature_extractor.match_features(
            prev_frame.descriptors,
            frame.descriptors
        )
        
        if len(matches) < 8:
            print(f"Warning: Only {len(matches)} matches found")
            return False
            
        # Get matched points
        pts1 = np.float32([prev_frame.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([frame.keypoints[m.trainIdx].pt for m in matches])
        
        # Estimate relative pose
        R, t, mask = self.pose_estimator.estimate_pose_essential(pts1, pts2)
        
        # Update global pose (chain transformations)
        R_prev, t_prev = prev_frame.get_pose()
        R_new = R_prev @ R
        t_new = R_prev @ t + t_prev.reshape(3, 1)
        
        frame.set_pose(R_new, t_new)
        self.current_pose = frame.T_w_c
        
        self.frames.append(frame)
        return True
    
    def get_trajectory(self):
        """
        Get camera trajectory as array of positions
        
        Returns:
            trajectory: Nx3 array of camera positions
        """
        trajectory = []
        for frame in self.frames:
            pos = frame.get_camera_center()
            trajectory.append(pos)
        return np.array(trajectory)