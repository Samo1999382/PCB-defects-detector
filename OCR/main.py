import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def text_only_ocr_visualization(image_path, output_path="text_only_ocr.jpg"):
    # Initialize model
    model = ocr_predictor(det_arch='linknet_resnet50', reco_arch='parseq', pretrained=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Get dimensions
    original_height, original_width = image.shape[:2]
    
    # Create a copy for drawing
    annotated_image = image.copy()
    
    # Perform OCR
    doc = DocumentFile.from_images(image_path)
    result = model(doc)
    
    all_detections = []
    
    # Process results and draw text overlays only
    for page_num, page in enumerate(result.pages):
        for block_num, block in enumerate(page.blocks):
            for line_num, line in enumerate(block.lines):
                for word_num, word in enumerate(line.words):
                    # Get word center position for text placement
                    bbox = word.geometry
                    x_center = int((bbox[0][0] + bbox[1][0]) / 2 * original_width)
                    y_center = int((bbox[0][1] + bbox[1][1]) / 2 * original_height)
                    
                    predicted_text = word.value
                    confidence = float(word.confidence)
                    
                    # Choose color based on confidence
                    if confidence > 0.8:
                        color = (0, 255, 0)  # Green for high confidence
                    elif confidence > 0.5:
                        color = (0, 255, 255)  # Yellow for medium confidence
                    else:
                        color = (0, 0, 255)  # Red for low confidence
                    
                    # Prepare text label (text only, no confidence)
                    label = f"{predicted_text}"
                    
                    # Calculate text size and position
                    font_scale = 0.6
                    thickness = 2
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    
                    # Calculate text position (centered on word)
                    text_x = x_center - text_size[0] // 2
                    text_y = y_center + text_size[1] // 2
                    
                    # Ensure text stays within image bounds
                    text_x = max(5, min(text_x, original_width - text_size[0] - 5))
                    text_y = max(text_size[1] + 5, min(text_y, original_height - 5))
                    
                    # Draw text with background for better readability
                    bg_padding = 5
                    cv2.rectangle(annotated_image,
                                (text_x - bg_padding, text_y - text_size[1] - bg_padding),
                                (text_x + text_size[0] + bg_padding, text_y + bg_padding),
                                color, -1)
                    
                    # Draw the text
                    cv2.putText(annotated_image, label, (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
                    
                    # Store detection info
                    all_detections.append({
                        'text': predicted_text,
                        'confidence': confidence,
                        'position': (x_center, y_center),
                        'page': page_num,
                        'block': block_num,
                        'line': line_num,
                        'word': word_num
                    })
    
    # Save the result
    cv2.imwrite(output_path, annotated_image)
    print(f"Text-only visualization saved to: {output_path}")
    
    # Print summary
    print(f"\n=== OCR SUMMARY ===")
    print(f"Total words detected: {len(all_detections)}")
    if all_detections:
        confidences = [d['confidence'] for d in all_detections]
        print(f"Average confidence: {np.mean(confidences):.3f}")
    
    # Display the image
    cv2.imshow('OCR Text Overlay', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return all_detections, result

# Usage
detections, result = text_only_ocr_visualization(
    "/home/armo38/Pictures/Screenshots/Screenshot From 2025-10-26 14-18-20.png",
    "text_only_ocr_result.jpg"
)

# Print extracted text
print("\n=== EXTRACTED TEXT ===")
full_text = result.render()
print(full_text)