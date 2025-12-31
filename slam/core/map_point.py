import numpy as np

class MapPoint:
    """
    Represents a 3D point in the map
    """
    next_id = 0
    
    def __init__(self, position, descriptor=None):
        """
        Args:
            position: 3D position (x, y, z)
            descriptor: Feature descriptor for matching
        """
        self.id = MapPoint.next_id
        MapPoint.next_id += 1
        
        self.position = np.array(position, dtype=np.float32)
        self.descriptor = descriptor
        
        # Observations: frames that see this point
        self.observations = {}  # {frame_id: keypoint_idx}
        
        # Quality tracking
        self.n_found = 0  # Times successfully matched
        self.n_visible = 0  # Times it should be visible
        
    def add_observation(self, frame, keypoint_idx):
        """
        Add a frame that observes this point
        """
        self.observations[frame.id] = keypoint_idx
        self.n_found += 1
        self.n_visible += 1
        
    def get_found_ratio(self):
        """
        Get ratio of times found vs visible (quality metric)
        """
        if self.n_visible == 0:
            return 0.0
        return self.n_found / self.n_visible
    
    def is_bad(self, min_observations=2, min_ratio=0.25):
        """
        Check if this is a bad point (should be removed)
        """
        if len(self.observations) < min_observations:
            return True
        if self.get_found_ratio() < min_ratio:
            return True
        return False