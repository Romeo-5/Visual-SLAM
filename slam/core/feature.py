import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, n_features=2000):
        """
        ORB feature extractor optimized for SLAM
        """
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
    def detect_and_compute(self, image):
        """
        Detect keypoints and compute descriptors
        
        Args:
            image: Grayscale image
            
        Returns:
            keypoints: List of cv2.KeyPoint
            descriptors: numpy array of shape (n_features, 32)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect and compute
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2, ratio_test=0.75):
        """
        Match features between two frames using Lowe's ratio test
        
        Args:
            desc1, desc2: Feature descriptors
            ratio_test: Ratio for Lowe's test (lower = more strict)
            
        Returns:
            good_matches: List of cv2.DMatch objects
        """
        # KNN matching (k=2 for ratio test)
        matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_test * n.distance:
                    good_matches.append(m)
                    
        return good_matches