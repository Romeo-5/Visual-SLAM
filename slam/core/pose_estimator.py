import cv2
import numpy as np

class PoseEstimator:
    def __init__(self, camera_matrix):
        """
        Estimate camera motion between frames
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix K
        """
        self.K = camera_matrix
        
    def estimate_pose_essential(self, pts1, pts2):
        """
        Estimate relative pose using Essential matrix decomposition
        
        Args:
            pts1, pts2: Matched 2D points (Nx2 arrays)
            
        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector (unit scale)
            inliers_mask: Boolean mask of inlier points
        """
        # Compute Essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, 
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        # Recover pose from Essential matrix
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        
        return R, t, mask
    
    def estimate_pose_pnp(self, pts_2d, pts_3d):
        """
        Estimate absolute pose using PnP (Perspective-n-Point)
        Used when we have 3D map points
        
        Args:
            pts_2d: 2D image points (Nx2)
            pts_3d: Corresponding 3D world points (Nx3)
            
        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
            inliers_mask: Boolean mask of inlier points
        """
        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d,
            self.K, None,
            reprojectionError=8.0,
            confidence=0.99
        )
        
        if not success:
            return None, None, None
            
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec
        
        inliers_mask = np.zeros(len(pts_2d), dtype=bool)
        if inliers is not None:
            inliers_mask[inliers.flatten()] = True
            
        return R, t, inliers_mask