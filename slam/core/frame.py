import numpy as np

class Frame:
    """
    Represents a single camera frame in SLAM
    """
    # Class variable for frame ID counter
    next_id = 0
    
    def __init__(self, image, camera, timestamp=None):
        """
        Args:
            image: Grayscale or RGB image
            camera: Camera object with intrinsics
            timestamp: Optional timestamp
        """
        self.id = Frame.next_id
        Frame.next_id += 1
        
        self.image = image
        self.camera = camera
        self.timestamp = timestamp
        
        # Pose (4x4 transformation matrix, world to camera)
        self.T_w_c = np.eye(4)  # Initially at origin
        
        # Features
        self.keypoints = None
        self.descriptors = None
        self.map_points = None  # 3D points corresponding to keypoints
        
        # Is this a keyframe?
        self.is_keyframe = False
        
    def set_pose(self, R, t):
        """
        Set camera pose (world to camera transform)
        
        Args:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
        """
        self.T_w_c[:3, :3] = R
        self.T_w_c[:3, 3] = t.flatten()
        
    def get_pose(self):
        """
        Returns:
            R: 3x3 rotation
            t: 3x1 translation
        """
        return self.T_w_c[:3, :3], self.T_w_c[:3, 3]
    
    def get_camera_center(self):
        """
        Get camera center in world coordinates
        """
        R, t = self.get_pose()
        # Camera center: C = -R^T * t
        return -R.T @ t