import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import json
import re

class Image3DMapper:
    def __init__(self, obj_file_path, image_path):
        """
        Initialize the 2D-3D mapping system
        
        Args:
            obj_file_path (str): Path to the 3D model (.obj file)
            image_path (str): Path to the 2D image (.jpg file)
        """
        self.obj_file_path = obj_file_path
        self.image_path = image_path
        self.mesh = None
        self.point_cloud = None
        self.image = None
        self.keypoints_2d = None
        self.descriptors_2d = None
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        
    def load_3d_model(self):
        """
        Load the 3D model from OBJ file and extract vertices and faces
        """
        try:
            vertices = []
            faces = []
            
            with open(self.obj_file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('v '):
                        # Parse vertex line: v x y z
                        parts = line.split()
                        if len(parts) >= 4:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            vertices.append([x, y, z])
                    elif line.startswith('f '):
                        # Parse face line: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                        parts = line.split()[1:]  # Skip 'f'
                        if len(parts) >= 3:
                            face_vertices = []
                            for part in parts[:3]:  # Take only first 3 vertices for triangles
                                # Extract vertex index (before first slash)
                                vertex_idx = int(part.split('/')[0]) - 1  # OBJ indices are 1-based
                                face_vertices.append(vertex_idx)
                            faces.append(face_vertices)
            
            if len(vertices) == 0:
                raise ValueError("No vertices found in OBJ file")
            
            # Convert to numpy arrays
            self.vertices = np.array(vertices)
            self.faces = np.array(faces) if faces else None
            
            # Sample vertices and faces if too many
            if len(self.vertices) > 10000:
                indices = np.random.choice(len(self.vertices), 10000, replace=False)
                self.vertices = self.vertices[indices]
                # Update face indices to match sampled vertices
                if self.faces is not None:
                    # Create mapping from old to new indices
                    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
                    # Filter faces that reference sampled vertices
                    valid_faces = []
                    for face in self.faces:
                        if all(v_idx in old_to_new for v_idx in face):
                            new_face = [old_to_new[v_idx] for v_idx in face]
                            valid_faces.append(new_face)
                    self.faces = np.array(valid_faces) if valid_faces else None
            
            print(f"Loaded 3D model with {len(self.vertices)} vertices and {len(self.faces) if self.faces is not None else 0} faces")
            return True
            
        except Exception as e:
            print(f"Error loading 3D model: {e}")
            return False
    
    def load_image(self):
        """
        Load and preprocess the 2D image
        """
        try:
            self.image = cv2.imread(self.image_path)
            if self.image is None:
                raise ValueError("Failed to load image")
            
            print(f"Loaded image with shape: {self.image.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def extract_2d_features(self):
        """
        Extract SIFT features from the 2D image
        """
        try:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.keypoints_2d, self.descriptors_2d = self.sift.detectAndCompute(gray, None)
            
            print(f"Extracted {len(self.keypoints_2d)} 2D features")
            return True
            
        except Exception as e:
            print(f"Error extracting 2D features: {e}")
            return False
    
    def estimate_camera_pose(self):
        """
        Estimate camera pose using PnP algorithm
        This is a simplified version - in practice, you'd need known correspondences
        """
        # Default camera parameters (should be calibrated for real use)
        height, width = self.image.shape[:2]
        focal_length = max(width, height)
        
        self.camera_matrix = np.array([
            [focal_length, 0, width/2],
            [0, focal_length, height/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4, 1))
        
        print("Camera parameters estimated")
        return True
    
    def project_3d_to_2d(self, points_3d, rvec, tvec):
        """
        Project 3D points to 2D image coordinates
        
        Args:
            points_3d: 3D points to project
            rvec: Rotation vector
            tvec: Translation vector
            
        Returns:
            2D projected points
        """
        points_2d, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3),
            rvec, tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        return points_2d.reshape(-1, 2)
    
    def find_correspondences(self):
        """
        Find correspondences between 2D image features and 3D points
        This uses a simplified approach based on spatial proximity
        """
        correspondences = []
        
        # Get 3D points
        points_3d = self.vertices
        
        # Estimate initial camera pose (simplified)
        # In practice, this would use more sophisticated methods
        rvec = np.array([0.1, 0.1, 0.0])  # Small rotation
        tvec = np.array([0.0, 0.0, -5.0])  # Move camera back
        
        # Project all 3D points to 2D
        projected_2d = self.project_3d_to_2d(points_3d, rvec, tvec)
        
        # Find nearest 2D features for each projected point
        kp_coords = np.array([kp.pt for kp in self.keypoints_2d])
        
        # Use KNN to find closest 2D features
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(kp_coords)
        
        for i, proj_pt in enumerate(projected_2d):
            # Check if projection is within image bounds
            if (0 <= proj_pt[0] < self.image.shape[1] and 
                0 <= proj_pt[1] < self.image.shape[0]):
                
                distances, indices = nbrs.kneighbors([proj_pt])
                
                # Only accept close matches
                if distances[0][0] < 50:  # Threshold in pixels
                    correspondences.append({
                        '3d_point': points_3d[i].tolist(),
                        '2d_point': kp_coords[indices[0][0]].tolist(),
                        'distance': float(distances[0][0])
                    })
        
        print(f"Found {len(correspondences)} correspondences")
        return correspondences
    
    def map_pixel_to_3d(self, pixel_x, pixel_y):
        """
        Map a specific pixel coordinate to 3D space
        
        Args:
            pixel_x, pixel_y: 2D pixel coordinates
            
        Returns:
            Closest 3D point or None if no good match
        """
        correspondences = self.find_correspondences()
        
        if not correspondences:
            return None
        
        # Find the closest correspondence to the given pixel
        min_dist = float('inf')
        closest_3d = None
        
        for corr in correspondences:
            dist = np.sqrt((corr['2d_point'][0] - pixel_x)**2 + 
                          (corr['2d_point'][1] - pixel_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_3d = corr['3d_point']
        
        return closest_3d if min_dist < 100 else None  # 100 pixel threshold
    
    def process(self):
        """
        Main processing pipeline
        """
        print("Starting 2D-3D mapping process...")
        
        # Load data
        if not self.load_3d_model():
            return False
        
        if not self.load_image():
            return False
        
        # Extract features
        if not self.extract_2d_features():
            return False
        
        # Estimate camera parameters
        if not self.estimate_camera_pose():
            return False
        
        # Find correspondences
        correspondences = self.find_correspondences()
        
        print("Processing complete!")
        return correspondences
    
    def save_correspondences(self, correspondences, output_path):
        """
        Save correspondences to JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(correspondences, f, indent=2)
        print(f"Correspondences saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Example paths - replace with actual file paths
    obj_path = "model.obj"
    img_path = "image.jpg"
    
    mapper = Image3DMapper(obj_path, img_path)
    correspondences = mapper.process()
    
    if correspondences:
        mapper.save_correspondences(correspondences, "correspondences.json")
        
        # Example: Map a specific pixel to 3D
        pixel_3d = mapper.map_pixel_to_3d(100, 200)
        if pixel_3d:
            print(f"Pixel (100, 200) maps to 3D point: {pixel_3d}")
        else:
            print("No 3D correspondence found for pixel (100, 200)")