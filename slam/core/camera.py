import numpy as np
import cv2


class Camera:
    """
    Pinhole camera model with optional distortion
    """
    
    def __init__(self, camera_matrix, dist_coeffs=None, image_size=None):
        """
        Initialize camera with intrinsic parameters
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix K
                [[fx, 0, cx],
                 [0, fy, cy],
                 [0,  0,  1]]
            dist_coeffs: Distortion coefficients (k1, k2, p1, p2, k3) or None
            image_size: (width, height) tuple or None
        """
        self.K = np.array(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64) if dist_coeffs is not None else None
        self.image_size = image_size
        
        # Extract intrinsic parameters
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        
        # Inverse of camera matrix (precomputed for efficiency)
        self.K_inv = np.linalg.inv(self.K)
        
    @classmethod
    def from_params(cls, fx, fy, cx, cy, dist_coeffs=None, image_size=None):
        """
        Create camera from individual parameters
        
        Args:
            fx, fy: Focal lengths in pixels
            cx, cy: Principal point coordinates
            dist_coeffs: Optional distortion coefficients
            image_size: Optional (width, height) tuple
        """
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        return cls(K, dist_coeffs, image_size)
    
    @classmethod
    def from_yaml(cls, yaml_path):
        """
        Load camera parameters from YAML configuration file
        
        Args:
            yaml_path: Path to YAML file
            
        Returns:
            Camera object
        """
        import yaml
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Handle different YAML formats
        if 'camera_matrix' in config:
            K = np.array(config['camera_matrix']).reshape(3, 3)
        elif 'Camera.fx' in config:
            # ORB-SLAM style config
            K = np.array([
                [config['Camera.fx'], 0, config['Camera.cx']],
                [0, config['Camera.fy'], config['Camera.cy']],
                [0, 0, 1]
            ])
        else:
            # Try extracting from nested structure
            cam = config.get('camera', config)
            K = np.array([
                [cam['fx'], 0, cam['cx']],
                [0, cam['fy'], cam['cy']],
                [0, 0, 1]
            ])
        
        dist_coeffs = config.get('dist_coeffs', config.get('distortion_coefficients', None))
        if dist_coeffs is not None:
            dist_coeffs = np.array(dist_coeffs).flatten()
            
        image_size = config.get('image_size', None)
        if image_size is not None:
            image_size = tuple(image_size)
            
        return cls(K, dist_coeffs, image_size)
    
    def project(self, points_3d):
        """
        Project 3D points to 2D image coordinates
        
        Args:
            points_3d: Nx3 array of 3D points in camera coordinates
            
        Returns:
            points_2d: Nx2 array of 2D image points
        """
        points_3d = np.asarray(points_3d, dtype=np.float64)
        
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)
            
        # Perspective projection
        x = points_3d[:, 0] / points_3d[:, 2]
        y = points_3d[:, 1] / points_3d[:, 2]
        
        # Apply distortion if available
        if self.dist_coeffs is not None:
            points_2d = self._apply_distortion(x, y)
        else:
            points_2d = np.column_stack([x, y])
        
        # Apply camera matrix
        u = self.fx * points_2d[:, 0] + self.cx
        v = self.fy * points_2d[:, 1] + self.cy
        
        return np.column_stack([u, v])
    
    def _apply_distortion(self, x, y):
        """
        Apply radial and tangential distortion
        
        Args:
            x, y: Normalized camera coordinates
            
        Returns:
            Distorted coordinates
        """
        k1, k2, p1, p2 = self.dist_coeffs[:4]
        k3 = self.dist_coeffs[4] if len(self.dist_coeffs) > 4 else 0
        
        r2 = x*x + y*y
        r4 = r2 * r2
        r6 = r4 * r2
        
        # Radial distortion
        radial = 1 + k1*r2 + k2*r4 + k3*r6
        
        # Tangential distortion
        x_dist = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        y_dist = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
        
        return np.column_stack([x_dist, y_dist])
    
    def unproject(self, points_2d, depth=1.0):
        """
        Unproject 2D image points to 3D rays or points
        
        Args:
            points_2d: Nx2 array of 2D image points
            depth: Scalar or Nx1 array of depths (default 1.0 for unit rays)
            
        Returns:
            points_3d: Nx3 array of 3D points in camera coordinates
        """
        points_2d = np.asarray(points_2d, dtype=np.float64)
        
        if points_2d.ndim == 1:
            points_2d = points_2d.reshape(1, 2)
            
        # Undistort if needed
        if self.dist_coeffs is not None:
            points_2d = cv2.undistortPoints(
                points_2d.reshape(-1, 1, 2),
                self.K,
                self.dist_coeffs,
                P=self.K
            ).reshape(-1, 2)
        
        # Convert to normalized camera coordinates
        x = (points_2d[:, 0] - self.cx) / self.fx
        y = (points_2d[:, 1] - self.cy) / self.fy
        
        # Scale by depth
        if np.isscalar(depth):
            depth = np.full(len(x), depth)
        else:
            depth = np.asarray(depth).flatten()
            
        X = x * depth
        Y = y * depth
        Z = depth
        
        return np.column_stack([X, Y, Z])
    
    def undistort_image(self, image):
        """
        Remove lens distortion from image
        
        Args:
            image: Input distorted image
            
        Returns:
            Undistorted image
        """
        if self.dist_coeffs is None:
            return image
            
        return cv2.undistort(image, self.K, self.dist_coeffs)
    
    def undistort_points(self, points_2d):
        """
        Remove distortion from 2D points
        
        Args:
            points_2d: Nx2 array of distorted image points
            
        Returns:
            Nx2 array of undistorted points
        """
        if self.dist_coeffs is None:
            return points_2d
            
        points_2d = np.asarray(points_2d, dtype=np.float64)
        
        undistorted = cv2.undistortPoints(
            points_2d.reshape(-1, 1, 2),
            self.K,
            self.dist_coeffs,
            P=self.K
        )
        
        return undistorted.reshape(-1, 2)
    
    def get_projection_matrix(self, R=None, t=None):
        """
        Get 3x4 projection matrix P = K[R|t]
        
        Args:
            R: 3x3 rotation matrix (default: identity)
            t: 3x1 translation vector (default: zeros)
            
        Returns:
            P: 3x4 projection matrix
        """
        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros((3, 1))
            
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).reshape(3, 1)
        
        Rt = np.hstack([R, t])
        P = self.K @ Rt
        
        return P
    
    def is_in_frame(self, points_2d, margin=0):
        """
        Check if 2D points are within image bounds
        
        Args:
            points_2d: Nx2 array of 2D points
            margin: Border margin in pixels
            
        Returns:
            Boolean mask of valid points
        """
        if self.image_size is None:
            raise ValueError("Image size not set")
            
        points_2d = np.asarray(points_2d)
        width, height = self.image_size
        
        valid = (
            (points_2d[:, 0] >= margin) &
            (points_2d[:, 0] < width - margin) &
            (points_2d[:, 1] >= margin) &
            (points_2d[:, 1] < height - margin)
        )
        
        return valid
    
    def resize(self, scale):
        """
        Create a new camera with scaled intrinsics
        
        Args:
            scale: Scale factor (e.g., 0.5 for half resolution)
            
        Returns:
            New Camera object with scaled parameters
        """
        K_scaled = self.K.copy()
        K_scaled[0, 0] *= scale  # fx
        K_scaled[1, 1] *= scale  # fy
        K_scaled[0, 2] *= scale  # cx
        K_scaled[1, 2] *= scale  # cy
        
        image_size = None
        if self.image_size is not None:
            image_size = (int(self.image_size[0] * scale), 
                         int(self.image_size[1] * scale))
        
        return Camera(K_scaled, self.dist_coeffs, image_size)
    
    def __repr__(self):
        return (f"Camera(fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f})")


class StereoCamera:
    """
    Stereo camera model for stereo SLAM
    """
    
    def __init__(self, left_camera, right_camera, baseline):
        """
        Initialize stereo camera
        
        Args:
            left_camera: Camera object for left camera
            right_camera: Camera object for right camera
            baseline: Distance between cameras in meters
        """
        self.left = left_camera
        self.right = right_camera
        self.baseline = baseline
        
        # For stereo, we typically use left camera intrinsics
        self.K = left_camera.K
        self.fx = left_camera.fx
        self.bf = baseline * self.fx  # baseline * focal length
        
    @classmethod
    def from_kitti(cls, calib_path):
        """
        Load stereo camera from KITTI calibration file
        
        Args:
            calib_path: Path to calib.txt file
            
        Returns:
            StereoCamera object
        """
        with open(calib_path, 'r') as f:
            lines = f.readlines()
        
        # Parse P0 (left camera projection matrix)
        P0_line = [l for l in lines if l.startswith('P0:')][0]
        P0 = np.array([float(x) for x in P0_line.split()[1:]]).reshape(3, 4)
        
        # Parse P1 (right camera projection matrix)
        P1_line = [l for l in lines if l.startswith('P1:')][0]
        P1 = np.array([float(x) for x in P1_line.split()[1:]]).reshape(3, 4)
        
        # Extract intrinsics from P0
        K = P0[:, :3]
        
        # Baseline from translation difference
        # P1 = K[R|t] where t = [-baseline, 0, 0]
        baseline = -P1[0, 3] / K[0, 0]
        
        left_cam = Camera(K)
        right_cam = Camera(K)
        
        return cls(left_cam, right_cam, baseline)
    
    def compute_depth(self, disparity):
        """
        Compute depth from stereo disparity
        
        Args:
            disparity: Disparity map or single value
            
        Returns:
            Depth in meters
        """
        # depth = baseline * fx / disparity
        with np.errstate(divide='ignore'):
            depth = self.bf / disparity
        return depth
    
    def triangulate_from_disparity(self, u, v, disparity):
        """
        Triangulate 3D point from left image coordinates and disparity
        
        Args:
            u, v: Left image coordinates
            disparity: Disparity value
            
        Returns:
            3D point in left camera coordinates
        """
        depth = self.compute_depth(disparity)
        
        X = (u - self.left.cx) * depth / self.left.fx
        Y = (v - self.left.cy) * depth / self.left.fy
        Z = depth
        
        return np.array([X, Y, Z])