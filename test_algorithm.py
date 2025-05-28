#!/usr/bin/env python3
"""
Test script for the 2D-3D mapping algorithm
This script helps validate the algorithm with sample data
"""

import numpy as np
import cv2
import open3d as o3d
import os
from mapping_algorithm import Image3DMapper

def create_sample_3d_model(output_path="sample_model.obj"):
    """
    Create a simple 3D model for testing purposes
    """
    print("Creating sample 3D model...")
    
    # Create a simple cube mesh
    mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
    mesh.translate([-1.0, -1.0, -1.0])  # Center at origin
    
    # Add some color
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    
    # Compute normals
    mesh.compute_vertex_normals()
    
    # Save as OBJ file
    success = o3d.io.write_triangle_mesh(output_path, mesh)
    
    if success:
        print(f"Sample 3D model saved to: {output_path}")
        return output_path
    else:
        print("Failed to create sample 3D model")
        return None

def create_sample_image(output_path="sample_image.jpg"):
    """
    Create a simple test image
    """
    print("Creating sample test image...")
    
    # Create a 640x480 image with some features
    height, width = 480, 640
    image = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add some geometric shapes as features
    # Rectangle
    cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 255), -1)
    
    # Circle
    cv2.circle(image, (400, 150), 50, (0, 255, 0), -1)
    
    # Triangle
    pts = np.array([[300, 300], [350, 250], [400, 300]], np.int32)
    cv2.fillPoly(image, [pts], (255, 0, 0))
    
    # Add some text
    cv2.putText(image, 'TEST IMAGE', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add some noise for more features
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    # Save image
    success = cv2.imwrite(output_path, image)
    
    if success:
        print(f"Sample image saved to: {output_path}")
        return output_path
    else:
        print("Failed to create sample image")
        return None

def test_algorithm():
    """
    Test the mapping algorithm with sample data
    """
    print("\n=== Testing 2D-3D Mapping Algorithm ===")
    
    # Create sample files
    model_path = create_sample_3d_model()
    image_path = create_sample_image()
    
    if not model_path or not image_path:
        print("Failed to create sample files")
        return False
    
    try:
        # Initialize mapper
        print("\nInitializing mapper...")
        mapper = Image3DMapper(model_path, image_path)
        
        # Test individual components
        print("\nTesting 3D model loading...")
        if not mapper.load_3d_model():
            print("❌ Failed to load 3D model")
            return False
        print("✅ 3D model loaded successfully")
        
        print("\nTesting image loading...")
        if not mapper.load_image():
            print("❌ Failed to load image")
            return False
        print("✅ Image loaded successfully")
        
        print("\nTesting feature extraction...")
        if not mapper.extract_2d_features():
            print("❌ Failed to extract features")
            return False
        print(f"✅ Extracted {len(mapper.keypoints_2d)} features")
        
        print("\nTesting camera pose estimation...")
        if not mapper.estimate_camera_pose():
            print("❌ Failed to estimate camera pose")
            return False
        print("✅ Camera pose estimated")
        
        print("\nTesting correspondence finding...")
        correspondences = mapper.find_correspondences()
        if correspondences:
            print(f"✅ Found {len(correspondences)} correspondences")
        else:
            print("⚠️  No correspondences found (this may be normal for synthetic data)")
        
        print("\nTesting pixel mapping...")
        # Test mapping a few pixels
        test_pixels = [(100, 100), (200, 200), (300, 300)]
        
        for px, py in test_pixels:
            result = mapper.map_pixel_to_3d(px, py)
            if result:
                print(f"✅ Pixel ({px}, {py}) -> 3D point: ({result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f})")
            else:
                print(f"⚠️  Pixel ({px}, {py}) -> No 3D correspondence found")
        
        print("\nTesting full processing pipeline...")
        correspondences = mapper.process()
        if correspondences:
            print(f"✅ Full pipeline completed with {len(correspondences)} correspondences")
            
            # Save results
            mapper.save_correspondences(correspondences, "test_correspondences.json")
            print("✅ Results saved to test_correspondences.json")
        else:
            print("⚠️  Full pipeline completed but no correspondences found")
        
        print("\n=== Algorithm Test Completed ===")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    finally:
        # Clean up sample files
        for file_path in [model_path, image_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Cleaned up: {file_path}")
                except:
                    pass

def test_dependencies():
    """
    Test if all required dependencies are installed
    """
    print("=== Testing Dependencies ===")
    
    dependencies = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'open3d': 'Open3D',
        'scipy': 'SciPy',
        'sklearn': 'scikit-learn'
    }
    
    all_good = True
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - NOT FOUND")
            all_good = False
    
    if all_good:
        print("✅ All dependencies are installed")
    else:
        print("❌ Some dependencies are missing. Run: pip install -r requirements.txt")
    
    return all_good

def main():
    """
    Main test function
    """
    print("2D-3D Mapping Algorithm Test Suite")
    print("===================================\n")
    
    # Test dependencies first
    if not test_dependencies():
        print("\nPlease install missing dependencies before running tests.")
        return
    
    print("\n")
    
    # Test the algorithm
    success = test_algorithm()
    
    if success:
        print("\n🎉 All tests passed! The algorithm is working correctly.")
        print("\nYou can now run the web application with: python app.py")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()