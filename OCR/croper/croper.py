import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def parse_annotation_file(annot_path):
    """
    Parse annotation file with format: x_center, y_center, width, height, rotation, text
    """
    croppings = []
    try:
        with open(annot_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Split the line by spaces
                parts = line.split()
                if len(parts) < 5:  # Need at least 5 values (x,y,w,h,rotation)
                    continue
                
                try:
                    # Extract coordinates and rotation
                    x_center = float(parts[0])
                    y_center = float(parts[1])
                    width = float(parts[2])
                    height = float(parts[3])
                    rotation = float(parts[4])
                    
                    # Extract text (remaining parts)
                    text = ' '.join(parts[5:]) if len(parts) > 5 else ""
                    
                    # Calculate bounding box coordinates
                    x_min = x_center - (width / 2)
                    y_min = y_center - (height / 2)
                    x_max = x_center + (width / 2)
                    y_max = y_center + (height / 2)
                    
                    croppings.append({
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'rotation': rotation,
                        'text': text,
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max
                    })
                    
                except ValueError as e:
                    print(f"Error parsing line {line_num} in {annot_path}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error reading {annot_path}: {e}")
    
    return croppings

def rotate_image(image, angle, center):
    """
    Rotate image around a center point
    """
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC)
    return rotated

def crop_with_rotation(image, annotation):
    """
    Crop image region with optional rotation correction
    """
    x_min = int(annotation['x_min'])
    y_min = int(annotation['y_min'])
    x_max = int(annotation['x_max'])
    y_max = int(annotation['y_max'])
    
    # Ensure coordinates are within image bounds
    height, width = image.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)
    
    w = x_max - x_min
    h = y_max - y_min
    
    if w <= 0 or h <= 0:
        print(f"Invalid crop dimensions: {w}x{h}")
        return None
    
    # Crop the region
    cropped = image[y_min:y_max, x_min:x_max].copy()
    
    # Apply rotation if needed (significant rotation)
    if abs(annotation['rotation']) > 1.0:
        center = (w // 2, h // 2)
        cropped = rotate_image(cropped, annotation['rotation'], center)
    
    return cropped

def process_folder(folder_path, output_folder="cropped_regions"):
    """
    Process all images in folder that have annotation files
    """
    folder = Path(folder_path)
    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
    
    # CSV data storage
    csv_data = []
    
    processed_count = 0
    total_crops = 0
    
    for image_path in image_files:
        # Find corresponding annotation file
        annot_path = folder / f"{image_path.stem}-annot.txt"
        
        if not annot_path.exists():
            print(f"No annotation file found for {image_path.name}")
            continue
        
        print(f"Processing {image_path.name}...")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            continue
        
        # Parse annotation file
        annotations = parse_annotation_file(annot_path)
        if not annotations:
            print(f"No valid annotations in {annot_path.name}")
            continue
        
        # Create subfolder for this image's crops
        image_output_dir = output_dir / image_path.stem
        image_output_dir.mkdir(exist_ok=True)
        
        # Process each annotation
        for i, annot in enumerate(annotations):
            cropped_image = crop_with_rotation(image, annot)
            
            if cropped_image is not None:
                # Save cropped image
                crop_filename = f"crop_{i+1:03d}.jpg"
                crop_path = image_output_dir / crop_filename
                cv2.imwrite(str(crop_path), cropped_image)
                
                # Add to CSV data
                csv_data.append({
                    'image_name': crop_filename,
                    'original_image': image_path.name,
                    'expected_text': annot['text'],
                    'x_center': annot['x_center'],
                    'y_center': annot['y_center'],
                    'width': annot['width'],
                    'height': annot['height'],
                    'rotation': annot['rotation'],
                    'crop_width': cropped_image.shape[1],
                    'crop_height': cropped_image.shape[0]
                })
                
                total_crops += 1
                print(f"  Saved crop {i+1}: {crop_filename} - Text: '{annot['text']}'")
        
        processed_count += 1
    
    # Create CSV file
    if csv_data:
        csv_path = output_dir / "annotations.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\nCSV file created: {csv_path}")
    
    print(f"\nProcessing complete!")
    print(f"Processed {processed_count} images")
    print(f"Created {total_crops} cropped regions")
    print(f"Output folder: {output_dir}")
    
    return csv_data

def preview_annotations(image_path, annot_path, max_previews=3):
    """
    Preview the annotations on the original image
    """
    image = cv2.imread(image_path)
    annotations = parse_annotation_file(annot_path)
    
    print(f"Preview for {Path(image_path).name}:")
    print(f"Found {len(annotations)} annotations")
    
    # Draw bounding boxes on original image
    preview_image = image.copy()
    for i, annot in enumerate(annotations[:max_previews]):
        x_min = int(annot['x_min'])
        y_min = int(annot['y_min'])
        x_max = int(annot['x_max'])
        y_max = int(annot['y_max'])
        
        # Draw rectangle
        cv2.rectangle(preview_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Draw text
        label = f"{i+1}: {annot['text']}" if annot['text'] else f"{i+1}"
        cv2.putText(preview_image, label, (x_min, y_min-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"  Annotation {i+1}: ({x_min},{y_min})-({x_max},{y_max}) - Text: '{annot['text']}'")
    
    # Show preview
    cv2.imshow('Annotations Preview', preview_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    # Process entire folder
    folder_path = "/home/armo38/PCB-defects-detector/OCR/Dataset/cvl_pcb_dslr_1/pcb1"  # Change this to your folder path
    csv_data = process_folder(folder_path)
    
    # For testing a single file
    # preview_annotations("rec1.jpg", "rec1_annot.txt")