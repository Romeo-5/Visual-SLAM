import numpy as np
import cv2
from scipy.optimize import least_squares

class BundleAdjustment:
    """
    Local Bundle Adjustment for pose and structure refinement
    """
    
    def __init__(self, camera_matrix):
        self.K = camera_matrix
        
    def optimize_local(self, keyframes, map_points, fix_first_pose=True):
        """
        Local Bundle Adjustment: optimize recent keyframes and their map points
        
        Args:
            keyframes: List of Frame objects to optimize
            map_points: List of MapPoint objects to optimize
            fix_first_pose: Whether to fix the first keyframe pose
            
        Returns:
            success: Boolean indicating if optimization converged
        """
        # Build parameter vector
        params = self._pack_parameters(keyframes, map_points, fix_first_pose)
        
        # Run optimization
        result = least_squares(
            self._reprojection_error,
            params,
            args=(keyframes, map_points, fix_first_pose),
            method='trf',  # Trust Region Reflective
            verbose=0,
            max_nfev=50  # Max function evaluations (keep it fast)
        )
        
        # Unpack optimized parameters
        self._unpack_parameters(result.x, keyframes, map_points, fix_first_pose)
        
        return result.success
    
    def _pack_parameters(self, keyframes, map_points, fix_first_pose):
        """
        Pack poses and 3D points into a single parameter vector
        
        Returns:
            params: 1D numpy array [poses, points]
        """
        params = []
        
        # Pack camera poses (6-DOF: rotation + translation)
        start_idx = 1 if fix_first_pose else 0
        for kf in keyframes[start_idx:]:
            R, t = kf.get_pose()
            # Convert rotation to Rodrigues (3-vector)
            rvec, _ = cv2.Rodrigues(R)
            params.extend(rvec.flatten())
            params.extend(t.flatten())
            
        # Pack 3D points (3-DOF each)
        for mp in map_points:
            params.extend(mp.position)
            
        return np.array(params)
    
    def _unpack_parameters(self, params, keyframes, map_points, fix_first_pose):
        """
        Unpack parameter vector back into poses and 3D points
        """
        idx = 0
        
        # Unpack poses
        start_kf = 1 if fix_first_pose else 0
        for kf in keyframes[start_kf:]:
            rvec = params[idx:idx+3]
            idx += 3
            t = params[idx:idx+3]
            idx += 3
            
            # Convert Rodrigues to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            kf.set_pose(R, t.reshape(3, 1))
            
        # Unpack 3D points
        for mp in map_points:
            mp.position = params[idx:idx+3]
            idx += 3
    
    def _reprojection_error(self, params, keyframes, map_points, fix_first_pose):
        """
        Compute reprojection errors for all observations
        
        Returns:
            residuals: 1D array of reprojection errors
        """
        # Unpack current parameters
        self._unpack_parameters(params, keyframes, map_points, fix_first_pose)
        
        residuals = []
        
        # For each map point
        for mp in map_points:
            point_3d = mp.position
            
            # For each observation of this point
            for frame_id, kp_idx in mp.observations.items():
                # Find the corresponding keyframe
                kf = next((f for f in keyframes if f.id == frame_id), None)
                if kf is None:
                    continue
                    
                # Get observed 2D point
                observed_pt = np.array(kf.keypoints[kp_idx].pt)
                
                # Project 3D point to this camera
                R, t = kf.get_pose()
                point_camera = R @ point_3d + t.flatten()
                
                # Project to image plane
                point_proj = self.K @ point_camera
                point_proj = point_proj[:2] / point_proj[2]
                
                # Compute residual
                error = observed_pt - point_proj
                residuals.extend(error)
                
        return np.array(residuals)