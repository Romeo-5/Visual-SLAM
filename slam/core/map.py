import numpy as np
from core.map_point import MapPoint

class Map:
    """
    Manages the 3D map of the environment
    """
    def __init__(self):
        self.map_points = []  # List of MapPoint objects
        self.keyframes = []   # List of keyframes
        
    def add_map_point(self, map_point):
        """
        Add a new map point
        """
        self.map_points.append(map_point)
        
    def add_keyframe(self, frame):
        """
        Add a keyframe
        """
        frame.is_keyframe = True
        self.keyframes.append(frame)
        
    def get_local_map_points(self, current_frame, n_keyframes=10):
        """
        Get map points visible from recent keyframes
        
        Args:
            current_frame: Current frame
            n_keyframes: Number of recent keyframes to consider
            
        Returns:
            local_points: List of MapPoint objects
        """
        # Get recent keyframes
        recent_kfs = self.keyframes[-n_keyframes:]
        
        # Collect map points seen by these keyframes
        local_points = set()
        for kf in recent_kfs:
            if kf.map_points is not None:
                for mp in kf.map_points:
                    if mp is not None and not mp.is_bad():
                        local_points.add(mp)
                        
        return list(local_points)
    
    def cull_map_points(self):
        """
        Remove bad map points from the map
        """
        self.map_points = [mp for mp in self.map_points if not mp.is_bad()]
        
    def get_point_cloud(self):
        """
        Get all map points as Nx3 array
        """
        return np.array([mp.position for mp in self.map_points if not mp.is_bad()])