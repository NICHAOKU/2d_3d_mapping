# 2D-3D Mapping Visualizer

A web-based application that maps 2D image pixels to 3D point cloud coordinates using computer vision techniques. This tool is designed for mapping close-up photos to 3D models constructed from photogrammetry.

## Features

- **2D-3D Correspondence Mapping**: Map pixels from 2D images to 3D point cloud coordinates
- **Interactive Visualization**: Web-based interface with 2D image viewer and 3D model renderer
- **Real-time Mapping**: Click on 2D image pixels to see corresponding 3D coordinates
- **Feature Detection**: Uses SIFT (Scale-Invariant Feature Transform) for robust feature matching
- **Multiple File Format Support**: Supports .jpg/.png images and .obj 3D models

## Algorithm Overview

The mapping algorithm works through several key steps:

1. **Feature Extraction**: Extract SIFT features from the 2D image
2. **3D Model Processing**: Load and sample points from the 3D mesh (.obj file)
3. **Camera Pose Estimation**: Estimate camera parameters and pose
4. **3D-to-2D Projection**: Project 3D points to 2D image coordinates
5. **Correspondence Finding**: Match 2D features with projected 3D points using nearest neighbor search
6. **Pixel Mapping**: For any clicked pixel, find the closest correspondence and return the 3D coordinate

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the project**:
   ```bash
   cd /path/to/2d_3d_mapping
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

1. **Run the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

### Using the Interface

1. **Upload Files**:
   - Upload your 2D image (.jpg, .png) - this should be a close-up photo of the building
   - Upload your 3D model (.obj) - this should be the photogrammetry-generated model

2. **Process Files**:
   - Click "Process Files" to run the mapping algorithm
   - Wait for processing to complete (may take a few moments)

3. **Interactive Mapping**:
   - Click anywhere on the 2D image to map that pixel to 3D coordinates
   - The corresponding 3D point will be highlighted in red on the 3D model
   - View mapping results in the information panel

4. **3D Model Navigation**:
   - Use mouse to rotate, zoom, and pan the 3D model
   - Left click + drag: Rotate
   - Right click + drag: Pan
   - Scroll wheel: Zoom

## File Structure

```
2d_3d_mapping/
├── app.py                 # Flask web application
├── mapping_algorithm.py   # Core 2D-3D mapping algorithm
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Web interface template
└── uploads/              # Directory for uploaded files (created automatically)
```

## Technical Details

### Algorithm Components

#### Feature Detection
- **SIFT (Scale-Invariant Feature Transform)**: Robust to scale, rotation, and illumination changes
- Extracts keypoints and descriptors from the 2D image

#### 3D Model Processing
- Loads .obj files using Open3D library
- Samples uniform points from mesh surface for correspondence matching
- Estimates surface normals for improved matching

#### Camera Pose Estimation
- Uses simplified camera model with estimated focal length
- In production, camera calibration would provide more accurate parameters
- Supports PnP (Perspective-n-Point) algorithm for pose refinement

#### Correspondence Matching
- Projects 3D points to 2D using estimated camera parameters
- Uses K-nearest neighbors to find closest 2D features
- Applies distance thresholds to filter poor matches

### Limitations and Improvements

**Current Limitations**:
- Simplified camera calibration (assumes basic pinhole model)
- Limited to single image mapping
- Requires manual parameter tuning for different scenarios

**Potential Improvements**:
- Implement proper camera calibration
- Add support for multiple images
- Integrate bundle adjustment for better accuracy
- Add machine learning-based feature matching
- Support for more 3D file formats (.ply, .pcd)

## Dependencies

- **Flask**: Web framework for the user interface
- **OpenCV**: Computer vision library for feature detection and image processing
- **Open3D**: 3D data processing library for handling .obj files
- **NumPy**: Numerical computing for array operations
- **SciPy**: Scientific computing for spatial operations
- **scikit-learn**: Machine learning library for nearest neighbor search

## Troubleshooting

### Common Issues

1. **"No correspondences found"**:
   - Ensure the 2D image shows the same building as the 3D model
   - Try images with more distinctive features
   - Check that the 3D model has sufficient detail

2. **"Failed to load 3D model"**:
   - Verify the .obj file is valid and not corrupted
   - Ensure the file contains vertex data

3. **Poor mapping accuracy**:
   - The algorithm uses simplified camera parameters
   - For better results, implement proper camera calibration
   - Ensure good overlap between 2D image and 3D model viewpoints

### Performance Tips

- Use images with resolution between 1000-2000 pixels for optimal performance
- Ensure 3D models have reasonable polygon counts (10K-100K vertices)
- Close the browser tab when not in use to free up GPU resources

## Contributing

This is a research/educational project. Contributions are welcome for:
- Improved camera calibration methods
- Better feature matching algorithms
- Support for additional file formats
- Performance optimizations
- UI/UX improvements

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or issues, please refer to the code comments and documentation within the source files.