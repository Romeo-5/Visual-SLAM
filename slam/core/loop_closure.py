import numpy as np
from sklearn.cluster import KMeans

class LoopClosureDetector:
    """
    Detect loop closures using Bag of Words (BoW) approach
    """
    
    def __init__(self, vocabulary_size=1000):
        """
        Args:
            vocabulary_size: Number of visual words in vocabulary
        """
        self.vocabulary_size = vocabulary_size
        self.vocabulary = None  # Will be KMeans model
        self.database = []  # List of (frame_id, bow_vector) tuples
        
    def build_vocabulary(self, all_descriptors):
        """
        Build visual vocabulary from training descriptors
        
        Args:
            all_descriptors: Nx32 array of ORB descriptors
        """
        print(f"Building vocabulary with {len(all_descriptors)} descriptors...")
        
        # Cluster descriptors into visual words
        self.vocabulary = KMeans(
            n_clusters=self.vocabulary_size,
            random_state=0,
            n_init=3,
            max_iter=100
        )
        self.vocabulary.fit(all_descriptors)
        
        print("Vocabulary built successfully")
        
    def compute_bow_vector(self, descriptors):
        """
        Compute Bag of Words vector for a frame
        
        Args:
            descriptors: Nx32 array of descriptors
            
        Returns:
            bow_vector: Histogram of visual word occurrences
        """
        if self.vocabulary is None:
            raise ValueError("Vocabulary not built yet!")
            
        # Assign each descriptor to nearest visual word
        word_ids = self.vocabulary.predict(descriptors)
        
        # Create histogram
        bow_vector = np.bincount(word_ids, minlength=self.vocabulary_size)
        
        # Normalize
        bow_vector = bow_vector.astype(np.float32)
        if bow_vector.sum() > 0:
            bow_vector /= bow_vector.sum()
            
        return bow_vector
    
    def add_frame(self, frame):
        """
        Add a keyframe to the database
        """
        bow_vector = self.compute_bow_vector(frame.descriptors)
        self.database.append((frame.id, bow_vector))
        
    def detect_loop(self, query_frame, similarity_threshold=0.7, temporal_gap=30):
        """
        Detect if query frame closes a loop
        
        Args:
            query_frame: Frame to query
            similarity_threshold: Minimum similarity to consider a loop
            temporal_gap: Minimum frame gap to consider (avoid recent frames)
            
        Returns:
            loop_frame_id: ID of matched frame, or None if no loop
            similarity: Similarity score
        """
        if len(self.database) < temporal_gap:
            return None, 0.0
            
        # Compute query BoW
        query_bow = self.compute_bow_vector(query_frame.descriptors)
        
        # Search database (excluding recent frames)
        best_match_id = None
        best_similarity = 0.0
        
        for frame_id, db_bow in self.database[:-temporal_gap]:
            # Compute similarity (cosine similarity or L1 score)
            similarity = np.dot(query_bow, db_bow)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = frame_id
                
        if best_similarity >= similarity_threshold:
            return best_match_id, best_similarity
        
        return None, best_similarity