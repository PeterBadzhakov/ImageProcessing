# Image Morphology Toolkit
A simple tool that performs basic morphological operations on grayscale images. Perfect for exploring how these operations affect images - from X-rays to technical drawings.

## What it does
Takes a grayscale image and applies four fundamental morphological operations, each serving a different purpose in image enhancement and analysis:

### Dilation
Makes bright regions expand outward. In our X-ray example, this made the bone structures appear slightly thicker and more prominent, enhancing the visibility of the rib cage's mesh pattern. Useful when you need to:
- Make features more visible
- Connect broken parts
- Highlight bright structures

### Erosion
The opposite of dilation - shrinks bright regions. In the X-ray, this thinned out the bone structures and emphasized the darker regions, making fine mesh patterns more distinct. Great for:
- Removing small bright noise
- Separating connected features
- Finding minimal structure boundaries

### Opening (Erosion → Dilation)
Removes small bright spots while preserving the overall shape of larger bright regions. In our chest X-ray, this cleaned up noise in the mesh pattern while maintaining the important bone structure. Perfect for:
- Cleaning up noisy images
- Removing small bright artifacts
- Smoothing object boundaries

### Closing (Dilation → Erosion)
Fills in small dark holes and gaps. In the X-ray, this created more continuous bone edges by filling tiny gaps in the mesh pattern without significantly altering the anatomical features. Useful for:
- Filling small holes
- Connecting nearby features
- Smoothing boundaries

## Quick Example
Input a chest X-ray, get 5 images back in the output folder:
- original.jpg (your input)
- dilation.jpg
- erosion.jpg
- opening.jpg
- closing.jpg

The operations use a standard 3x3 kernel - a good balance between visible effect and detail preservation. All processing maintains the original grayscale values, ensuring subtle and realistic results.

## Getting Started
1. Make sure you have Python 3.x installed

2. Install the requirements:
> pip install numpy opencv-python
3. Run it 
python main.py your_image.jpg

## Results will appear in an 'output' folder where you ran the script.

## Notes
- Works best with grayscale images like X-rays, technical drawings, fingerprints
- Processes images while preserving grayscale values - no harsh binary conversion
- Includes error checking for image loading and processing
- Uses symmetric padding to handle image edges properly
