import numpy as np
import cv2
from collections import deque
from threading import Thread, Lock
import time

from core.camera import Camera
from core.frame import Frame
from core.feature import FeatureExtractor
from core.pose_estimator import PoseEstimator
from core.map import Map
from core.map_point import MapPoint
from core.triangulation import Triangulator
from core.bundle_adjustment import BundleAdjustment
from core.loop_closure import LoopClosureDetector


class SLAMSystem:
    """
    Complete Visual SLAM system with tracking, mapping, and optional loop closure
    """
    
    # System states
    STATE_NOT_INITIALIZED = 0
    STATE_INITIALIZING = 1
    STATE_TRACKING = 2
    STATE_LOST = 3
    
    def __init__(self, camera_matrix, config=None):
        """
        Initialize SLAM system
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            config: Optional configuration dictionary
        """
        # Default configuration
        self.config = {
            'n_features': 2000,
            'min_matches_init': 100,
            'min_matches_tracking': 20,
            'keyframe_ratio': 0.9,  # New keyframe if <90% points tracked
            'min_keyframe_interval': 10,  # Minimum frames between keyframes
            'local_ba_size': 10,  # Number of keyframes for local BA
            'enable_loop_closure': True,
            'vocabulary_size': 1000,
            'loop_threshold': 0.7,
            'min_parallax': 1.0,  # Minimum parallax for triangulation (degrees)
        }
        if config:
            self.config.update(config)
        
        # Camera
        self.K = np.array(camera_matrix, dtype=np.float64)
        self.camera = Camera(camera_matrix)
        
        # Core components
        self.feature_extractor = FeatureExtractor(n_features=self.config['n_features'])
        self.pose_estimator = PoseEstimator(self.K)
        self.triangulator = Triangulator()
        self.bundle_adjuster = BundleAdjustment(self.K)
        self.map = Map()
        
        # Loop closure
        self.loop_detector = None
        if self.config['enable_loop_closure']:
            self.loop_detector = LoopClosureDetector(
                vocabulary_size=self.config['vocabulary_size']
            )
        
        # State
        self.state = self.STATE_NOT_INITIALIZED
        self.current_frame = None
        self.reference_frame = None
        self.last_keyframe = None
        self.frames_since_keyframe = 0
        
        # All frames (for trajectory)
        self.frames = []
        
        # Statistics
        self.stats = {
            'n_frames': 0,
            'n_keyframes': 0,
            'n_map_points': 0,
            'n_loop_closures': 0,
            'tracking_time': [],
            'mapping_time': [],
        }
        
        # Thread safety
        self.map_lock = Lock()
        
    def process_frame(self, image, timestamp=None):
        """
        Process a new frame
        
        Args:
            image: Input image (grayscale or RGB)
            timestamp: Optional timestamp
            
        Returns:
            success: Boolean indicating if frame was processed successfully
        """
        start_time = time.time()
        
        # Create frame
        frame = Frame(image, self.camera, timestamp)
        
        # Extract features
        frame.keypoints, frame.descriptors = self.feature_extractor.detect_and_compute(image)
        
        if frame.descriptors is None or len(frame.keypoints) < 50:
            print(f"Warning: Only {len(frame.keypoints) if frame.keypoints else 0} features detected")
            return False
        
        self.current_frame = frame
        self.stats['n_frames'] += 1
        
        # State machine
        if self.state == self.STATE_NOT_INITIALIZED:
            success = self._initialize(frame)
        elif self.state == self.STATE_INITIALIZING:
            success = self._try_initialize(frame)
        elif self.state == self.STATE_TRACKING:
            success = self._track(frame)
        elif self.state == self.STATE_LOST:
            success = self._relocalize(frame)
        else:
            success = False
            
        if success:
            self.frames.append(frame)
            
        self.stats['tracking_time'].append(time.time() - start_time)
        
        return success
    
    def _initialize(self, frame):
        """
        Start initialization with first frame
        """
        # Store as reference frame
        self.reference_frame = frame
        frame.set_pose(np.eye(3), np.zeros((3, 1)))
        
        self.state = self.STATE_INITIALIZING
        print("Initialization started - waiting for second frame with sufficient parallax")
        
        return True
    
    def _try_initialize(self, frame):
        """
        Try to complete initialization with second frame
        """
        # Match with reference frame
        matches = self.feature_extractor.match_features(
            self.reference_frame.descriptors,
            frame.descriptors,
            ratio_test=0.7
        )
        
        if len(matches) < self.config['min_matches_init']:
            print(f"Not enough matches for initialization: {len(matches)}")
            return False
        
        # Get matched points
        pts1 = np.float32([self.reference_frame.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([frame.keypoints[m.trainIdx].pt for m in matches])
        
        # Estimate relative pose
        R, t, mask = self.pose_estimator.estimate_pose_essential(pts1, pts2)
        
        if R is None:
            return False
        
        # Check parallax
        parallax = self._compute_parallax(pts1, pts2, mask)
        if parallax < self.config['min_parallax']:
            print(f"Insufficient parallax for initialization: {parallax:.2f}°")
            return False
        
        # Set frame poses
        frame.set_pose(R, t)
        
        # Triangulate initial map points
        n_points = self._create_initial_map(
            self.reference_frame, frame, matches, mask.flatten().astype(bool)
        )
        
        if n_points < 50:
            print(f"Not enough triangulated points: {n_points}")
            return False
        
        # Add keyframes
        self.map.add_keyframe(self.reference_frame)
        self.map.add_keyframe(frame)
        self.last_keyframe = frame
        self.stats['n_keyframes'] = 2
        
        # Build vocabulary if loop closure enabled
        if self.loop_detector is not None:
            # Collect descriptors for vocabulary
            all_descs = np.vstack([
                self.reference_frame.descriptors,
                frame.descriptors
            ])
            self.loop_detector.build_vocabulary(all_descs)
            self.loop_detector.add_frame(self.reference_frame)
            self.loop_detector.add_frame(frame)
        
        self.state = self.STATE_TRACKING
        print(f"Initialization complete! {n_points} map points created")
        
        return True
    
    def _compute_parallax(self, pts1, pts2, mask):
        """
        Compute median parallax angle between point correspondences
        """
        if mask is None:
            mask = np.ones(len(pts1), dtype=bool)
        else:
            mask = mask.flatten().astype(bool)
            
        pts1_inlier = pts1[mask]
        pts2_inlier = pts2[mask]
        
        # Compute angle from pixel displacement
        # This is a simplified approximation
        displacement = np.linalg.norm(pts2_inlier - pts1_inlier, axis=1)
        focal = (self.K[0, 0] + self.K[1, 1]) / 2
        angles = np.arctan(displacement / focal) * 180 / np.pi
        
        return np.median(angles)
    
    def _create_initial_map(self, frame1, frame2, matches, inlier_mask):
        """
        Create initial map from two-view reconstruction
        """
        # Get projection matrices
        R1, t1 = frame1.get_pose()
        R2, t2 = frame2.get_pose()
        
        P1 = self.camera.get_projection_matrix(R1, t1)
        P2 = self.camera.get_projection_matrix(R2, t2)
        
        # Get inlier matches
        inlier_matches = [m for m, inlier in zip(matches, inlier_mask) if inlier]
        
        pts1 = np.float32([frame1.keypoints[m.queryIdx].pt for m in inlier_matches])
        pts2 = np.float32([frame2.keypoints[m.trainIdx].pt for m in inlier_matches])
        
        # Triangulate
        points_3d = self.triangulator.triangulate_points(pts1, pts2, P1, P2)
        
        # Validate triangulation
        valid_mask = self.triangulator.check_triangulation(
            points_3d, P1, P2, pts1, pts2, threshold=2.0
        )
        
        # Initialize map point arrays for frames
        frame1.map_points = [None] * len(frame1.keypoints)
        frame2.map_points = [None] * len(frame2.keypoints)
        
        # Create map points
        n_created = 0
        for i, (match, valid) in enumerate(zip(inlier_matches, valid_mask)):
            if not valid:
                continue
                
            # Create map point
            mp = MapPoint(
                position=points_3d[i],
                descriptor=frame1.descriptors[match.queryIdx]
            )
            
            # Add observations
            mp.add_observation(frame1, match.queryIdx)
            mp.add_observation(frame2, match.trainIdx)
            
            # Link to frames
            frame1.map_points[match.queryIdx] = mp
            frame2.map_points[match.trainIdx] = mp
            
            # Add to map
            with self.map_lock:
                self.map.add_map_point(mp)
            
            n_created += 1
        
        self.stats['n_map_points'] = n_created
        
        return n_created
    
    def _track(self, frame):
        """
        Track camera pose in current frame
        """
        # First, try to track using map points (more robust)
        success = self._track_with_map(frame)
        
        if not success:
            # Fall back to tracking with reference frame
            success = self._track_with_reference(frame)
        
        if not success:
            self.state = self.STATE_LOST
            print("Tracking lost!")
            return False
        
        # Update reference frame
        self.reference_frame = frame
        self.frames_since_keyframe += 1
        
        # Check if we need a new keyframe
        if self._need_new_keyframe(frame):
            self._create_keyframe(frame)
        
        return True
    
    def _track_with_map(self, frame):
        """
        Track using existing map points (PnP)
        """
        # Get local map points
        with self.map_lock:
            local_points = self.map.get_local_map_points(frame, n_keyframes=5)
        
        if len(local_points) < 20:
            return False
        
        # Project map points to current frame and match
        pts_3d = []
        pts_2d = []
        point_indices = []
        
        for i, mp in enumerate(local_points):
            # Match map point descriptor to frame features
            if mp.descriptor is None or frame.descriptors is None:
                continue
                
            # Simple descriptor matching
            distances = np.linalg.norm(
                frame.descriptors.astype(np.float32) - mp.descriptor.astype(np.float32),
                axis=1
            )
            best_idx = np.argmin(distances)
            
            if distances[best_idx] < 50:  # Hamming distance threshold
                pts_3d.append(mp.position)
                pts_2d.append(frame.keypoints[best_idx].pt)
                point_indices.append((i, best_idx))
        
        if len(pts_3d) < self.config['min_matches_tracking']:
            return False
        
        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)
        
        # Estimate pose using PnP
        R, t, inlier_mask = self.pose_estimator.estimate_pose_pnp(pts_2d, pts_3d)
        
        if R is None:
            return False
        
        frame.set_pose(R, t)
        
        # Update map point observations
        frame.map_points = [None] * len(frame.keypoints)
        for (mp_idx, kp_idx), is_inlier in zip(point_indices, inlier_mask):
            if is_inlier:
                mp = local_points[mp_idx]
                mp.add_observation(frame, kp_idx)
                frame.map_points[kp_idx] = mp
        
        n_inliers = np.sum(inlier_mask)
        print(f"PnP tracking: {n_inliers}/{len(pts_3d)} inliers")
        
        return n_inliers >= self.config['min_matches_tracking']
    
    def _track_with_reference(self, frame):
        """
        Track relative to reference frame using Essential matrix
        """
        if self.reference_frame is None:
            return False
        
        # Match features
        matches = self.feature_extractor.match_features(
            self.reference_frame.descriptors,
            frame.descriptors
        )
        
        if len(matches) < self.config['min_matches_tracking']:
            return False
        
        # Get matched points
        pts1 = np.float32([self.reference_frame.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([frame.keypoints[m.trainIdx].pt for m in matches])
        
        # Estimate relative pose
        R_rel, t_rel, mask = self.pose_estimator.estimate_pose_essential(pts1, pts2)
        
        if R_rel is None:
            return False
        
        # Chain with reference frame pose
        R_ref, t_ref = self.reference_frame.get_pose()
        R_new = R_ref @ R_rel
        t_new = R_ref @ t_rel + t_ref.reshape(3, 1)
        
        frame.set_pose(R_new, t_new)
        
        print(f"Reference tracking: {np.sum(mask)}/{len(matches)} inliers")
        
        return True
    
    def _need_new_keyframe(self, frame):
        """
        Check if current frame should become a keyframe
        """
        if self.frames_since_keyframe < self.config['min_keyframe_interval']:
            return False
        
        # Count tracked map points
        if frame.map_points is None:
            return True
        
        n_tracked = sum(1 for mp in frame.map_points if mp is not None)
        n_reference = sum(1 for mp in self.last_keyframe.map_points if mp is not None) if self.last_keyframe.map_points else 0
        
        if n_reference == 0:
            return True
        
        tracking_ratio = n_tracked / n_reference
        
        return tracking_ratio < self.config['keyframe_ratio']
    
    def _create_keyframe(self, frame):
        """
        Add frame as keyframe and create new map points
        """
        start_time = time.time()
        
        with self.map_lock:
            self.map.add_keyframe(frame)
        
        self.stats['n_keyframes'] += 1
        self.frames_since_keyframe = 0
        
        # Triangulate new map points with previous keyframe
        if self.last_keyframe is not None:
            n_new = self._triangulate_new_points(self.last_keyframe, frame)
            print(f"Created {n_new} new map points")
        
        # Add to loop closure database
        if self.loop_detector is not None:
            self.loop_detector.add_frame(frame)
            
            # Check for loop closure
            loop_id, similarity = self.loop_detector.detect_loop(
                frame,
                similarity_threshold=self.config['loop_threshold']
            )
            
            if loop_id is not None:
                print(f"Loop closure detected! Frame {frame.id} -> Frame {loop_id} (sim={similarity:.2f})")
                self.stats['n_loop_closures'] += 1
                # TODO: Implement pose graph optimization
        
        # Local bundle adjustment
        if len(self.map.keyframes) > 2:
            self._run_local_ba()
        
        self.last_keyframe = frame
        
        self.stats['mapping_time'].append(time.time() - start_time)
    
    def _triangulate_new_points(self, kf1, kf2):
        """
        Triangulate new map points between two keyframes
        """
        # Match untracked features
        matches = self.feature_extractor.match_features(
            kf1.descriptors,
            kf2.descriptors,
            ratio_test=0.7
        )
        
        # Filter out already tracked points
        new_matches = []
        for m in matches:
            if kf1.map_points is None or kf1.map_points[m.queryIdx] is None:
                if kf2.map_points is None or kf2.map_points[m.trainIdx] is None:
                    new_matches.append(m)
        
        if len(new_matches) < 10:
            return 0
        
        # Get projection matrices
        R1, t1 = kf1.get_pose()
        R2, t2 = kf2.get_pose()
        P1 = self.camera.get_projection_matrix(R1, t1)
        P2 = self.camera.get_projection_matrix(R2, t2)
        
        # Get matched points
        pts1 = np.float32([kf1.keypoints[m.queryIdx].pt for m in new_matches])
        pts2 = np.float32([kf2.keypoints[m.trainIdx].pt for m in new_matches])
        
        # Triangulate
        points_3d = self.triangulator.triangulate_points(pts1, pts2, P1, P2)
        
        # Validate
        valid_mask = self.triangulator.check_triangulation(
            points_3d, P1, P2, pts1, pts2, threshold=2.0
        )
        
        # Ensure map_points arrays exist
        if kf1.map_points is None:
            kf1.map_points = [None] * len(kf1.keypoints)
        if kf2.map_points is None:
            kf2.map_points = [None] * len(kf2.keypoints)
        
        # Create new map points
        n_created = 0
        for i, (match, valid) in enumerate(zip(new_matches, valid_mask)):
            if not valid:
                continue
            
            mp = MapPoint(
                position=points_3d[i],
                descriptor=kf1.descriptors[match.queryIdx]
            )
            
            mp.add_observation(kf1, match.queryIdx)
            mp.add_observation(kf2, match.trainIdx)
            
            kf1.map_points[match.queryIdx] = mp
            kf2.map_points[match.trainIdx] = mp
            
            with self.map_lock:
                self.map.add_map_point(mp)
            
            n_created += 1
        
        self.stats['n_map_points'] += n_created
        
        return n_created
    
    def _run_local_ba(self):
        """
        Run local bundle adjustment on recent keyframes
        """
        with self.map_lock:
            # Get recent keyframes
            n_kf = min(self.config['local_ba_size'], len(self.map.keyframes))
            keyframes = self.map.keyframes[-n_kf:]
            
            # Get map points observed by these keyframes
            map_points = set()
            for kf in keyframes:
                if kf.map_points is not None:
                    for mp in kf.map_points:
                        if mp is not None and not mp.is_bad():
                            map_points.add(mp)
            
            map_points = list(map_points)
        
        if len(map_points) < 10:
            return
        
        # Run optimization
        success = self.bundle_adjuster.optimize_local(
            keyframes, map_points, fix_first_pose=True
        )
        
        if success:
            # Cull bad map points
            with self.map_lock:
                self.map.cull_map_points()
    
    def _relocalize(self, frame):
        """
        Try to relocalize after tracking is lost
        """
        # Try to match with recent keyframes
        for kf in reversed(self.map.keyframes[-10:]):
            matches = self.feature_extractor.match_features(
                kf.descriptors,
                frame.descriptors,
                ratio_test=0.7
            )
            
            if len(matches) < 30:
                continue
            
            # Get 3D-2D correspondences
            pts_3d = []
            pts_2d = []
            
            for m in matches:
                if kf.map_points is not None and kf.map_points[m.queryIdx] is not None:
                    mp = kf.map_points[m.queryIdx]
                    pts_3d.append(mp.position)
                    pts_2d.append(frame.keypoints[m.trainIdx].pt)
            
            if len(pts_3d) < 20:
                continue
            
            pts_3d = np.array(pts_3d, dtype=np.float32)
            pts_2d = np.array(pts_2d, dtype=np.float32)
            
            # Try PnP
            R, t, inliers = self.pose_estimator.estimate_pose_pnp(pts_2d, pts_3d)
            
            if R is not None and np.sum(inliers) >= 20:
                frame.set_pose(R, t)
                self.reference_frame = frame
                self.state = self.STATE_TRACKING
                print(f"Relocalization successful! Matched with keyframe {kf.id}")
                return True
        
        print("Relocalization failed")
        return False
    
    def get_current_pose(self):
        """
        Get current camera pose
        
        Returns:
            T: 4x4 transformation matrix (world to camera)
        """
        if self.current_frame is None:
            return np.eye(4)
        return self.current_frame.T_w_c.copy()
    
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
        return np.array(trajectory) if trajectory else np.array([]).reshape(0, 3)
    
    def get_poses(self):
        """
        Get all camera poses
        
        Returns:
            poses: List of 4x4 transformation matrices
        """
        return [frame.T_w_c.copy() for frame in self.frames]
    
    def get_point_cloud(self):
        """
        Get current 3D point cloud
        
        Returns:
            points: Nx3 array of 3D points
        """
        with self.map_lock:
            return self.map.get_point_cloud()
    
    def get_keyframe_poses(self):
        """
        Get keyframe poses only
        
        Returns:
            poses: List of 4x4 transformation matrices
        """
        with self.map_lock:
            return [kf.T_w_c.copy() for kf in self.map.keyframes]
    
    def get_statistics(self):
        """
        Get system statistics
        
        Returns:
            dict with statistics
        """
        stats = self.stats.copy()
        stats['state'] = self.state
        stats['avg_tracking_time'] = np.mean(stats['tracking_time']) if stats['tracking_time'] else 0
        stats['avg_mapping_time'] = np.mean(stats['mapping_time']) if stats['mapping_time'] else 0
        return stats
    
    def save_trajectory(self, filepath, format='tum'):
        """
        Save trajectory to file
        
        Args:
            filepath: Output file path
            format: 'tum' or 'kitti'
        """
        with open(filepath, 'w') as f:
            for i, frame in enumerate(self.frames):
                R, t = frame.get_pose()
                pos = frame.get_camera_center()
                
                if format == 'tum':
                    # TUM format: timestamp tx ty tz qx qy qz qw
                    from scipy.spatial.transform import Rotation
                    quat = Rotation.from_matrix(R.T).as_quat()  # x, y, z, w
                    timestamp = frame.timestamp if frame.timestamp else i
                    f.write(f"{timestamp} {pos[0]} {pos[1]} {pos[2]} "
                           f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}\n")
                           
                elif format == 'kitti':
                    # KITTI format: 3x4 transformation matrix (row-major)
                    T = np.eye(4)
                    T[:3, :3] = R.T  # World to camera -> Camera to world
                    T[:3, 3] = pos
                    f.write(' '.join(map(str, T[:3].flatten())) + '\n')
    
    def reset(self):
        """
        Reset SLAM system
        """
        self.state = self.STATE_NOT_INITIALIZED
        self.current_frame = None
        self.reference_frame = None
        self.last_keyframe = None
        self.frames_since_keyframe = 0
        self.frames = []
        
        self.map = Map()
        
        if self.loop_detector:
            self.loop_detector = LoopClosureDetector(
                vocabulary_size=self.config['vocabulary_size']
            )
        
        # Reset frame IDs
        Frame.next_id = 0
        MapPoint.next_id = 0
        
        self.stats = {
            'n_frames': 0,
            'n_keyframes': 0,
            'n_map_points': 0,
            'n_loop_closures': 0,
            'tracking_time': [],
            'mapping_time': [],
        }
        
        print("SLAM system reset")