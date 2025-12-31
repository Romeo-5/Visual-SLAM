import numpy as np
import cv2

class Triangulator:
    """
    Reconstruct 3D points from 2D correspondences
    """
    
    @staticmethod
    def triangulate_points(pts1, pts2, P1, P2):
        """
        Triangulate 3D points from two views
        
        Args:
            pts1, pts2: Matched 2D points (Nx2)
            P1, P2: 3x4 projection matrices [K[R|t]]
            
        Returns:
            points_3d: Nx3 array of 3D points in world coordinates
        """
        # Reshape points for cv2.triangulatePoints
        pts1_h = pts1.T  # 2xN
        pts2_h = pts2.T  # 2xN
        
        # Triangulate (returns homogeneous coordinates)
        points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
        
        # Convert to 3D (divide by homogeneous coordinate)
        points_3d = points_4d[:3, :] / points_4d[3, :]
        
        return points_3d.T  # Nx3
    
    @staticmethod
    def check_triangulation(points_3d, P1, P2, pts1, pts2, threshold=1.0):
        """
        Validate triangulated points
        
        Returns:
            valid_mask: Boolean mask of valid points
        """
        n_points = len(points_3d)
        valid_mask = np.ones(n_points, dtype=bool)
        
        # Convert to homogeneous
        points_3d_h = np.hstack([points_3d, np.ones((n_points, 1))])
        
        # Project to both cameras
        pts1_proj = (P1 @ points_3d_h.T).T
        pts2_proj = (P2 @ points_3d_h.T).T
        
        # Normalize
        pts1_proj = pts1_proj[:, :2] / pts1_proj[:, 2:]
        pts2_proj = pts2_proj[:, :2] / pts2_proj[:, 2:]
        
        # Compute reprojection error
        error1 = np.linalg.norm(pts1 - pts1_proj, axis=1)
        error2 = np.linalg.norm(pts2 - pts2_proj, axis=1)
        
        # Check if points are in front of both cameras and reprojection error is low
        valid_mask &= (points_3d[:, 2] > 0)  # Positive depth in camera 1
        valid_mask &= (error1 < threshold)
        valid_mask &= (error2 < threshold)
        
        return valid_mask