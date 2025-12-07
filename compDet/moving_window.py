import argparse
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from collections import defaultdict

def multi_scale_sliding_window(model, image, window_sizes=[320, 640, 1280], overlap=0.25, conf_threshold=0.25):
    """
    Perform multi-scale sliding window inference on large images
    Fixed to handle edges and implement proper overlapping in both directions
    """
    h, w = image.shape[:2]
    all_detections = []
    
    for window_size in window_sizes:
        print(f"Processing with window size: {window_size}")
        stride = int(window_size * (1 - overlap))
        
        # Ensure we cover the entire image, including edges
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Calculate window coordinates, handling edges
                y1 = y
                y2 = min(y + window_size, h)
                x1 = x
                x2 = min(x + window_size, w)
                
                # Skip if window is too small (less than 50% of target size)
                actual_height = y2 - y1
                actual_width = x2 - x1
                if actual_height < window_size * 0.5 or actual_width < window_size * 0.5:
                    continue
                
                # Extract window (may be smaller than window_size at edges)
                window = image[y1:y2, x1:x2]
                
                # If window is smaller than target, pad it
                if window.shape[0] != window_size or window.shape[1] != window_size:
                    padded_window = np.zeros((window_size, window_size, 3), dtype=window.dtype)
                    padded_window[0:window.shape[0], 0:window.shape[1]] = window
                    window = padded_window
                
                # Run inference on window
                results = model(window, imgsz=window_size, conf=conf_threshold, verbose=False)
                
                # Process detections and adjust coordinates to original image space
                for result in results:
                    if hasattr(result, 'obb') and result.obb is not None:
                        for detection in result.obb:
                            xywhr = detection.xywhr[0].clone()
                            # Ensure tensor is on CPU for consistent processing
                            xywhr = xywhr.cpu()
                            # Adjust coordinates to original image space
                            xywhr[0] += x1  # x_center
                            xywhr[1] += y1  # y_center
                            
                            # If we padded the window, adjust for actual detection area
                            if actual_width < window_size:
                                # Scale x coordinates back to original image space
                                scale_x = actual_width / window_size
                                xywhr[0] = x1 + (xywhr[0] - x1) * scale_x
                                xywhr[2] *= scale_x
                            
                            if actual_height < window_size:
                                # Scale y coordinates back to original image space
                                scale_y = actual_height / window_size
                                xywhr[1] = y1 + (xywhr[1] - y1) * scale_y
                                xywhr[3] *= scale_y
                            
                            detection_data = {
                                'xywhr': xywhr,
                                'conf': detection.conf.item(),
                                'cls': detection.cls.item(),
                                'angle': xywhr[4].item(),
                                'window_size': window_size,
                                'width': xywhr[2].item(),
                                'height': xywhr[3].item()
                            }
                            all_detections.append(detection_data)
    
    return all_detections

def calculate_angle_similarity(angle1, angle2):
    """
    Calculate similarity between two angles (0-1, where 1 is identical)
    Handles angle wrap-around (0° and 360° are the same)
    """
    # Convert to degrees and normalize to 0-180 range (since 0° and 180° are the same for rectangles)
    deg1 = np.degrees(angle1) % 180
    deg2 = np.degrees(angle2) % 180
    
    # Calculate absolute difference (considering wrap-around)
    diff = min(abs(deg1 - deg2), 180 - abs(deg1 - deg2))
    
    # Convert to similarity score (1.0 when identical, 0.0 when 90° apart)
    similarity = 1.0 - (diff / 90.0)
    return max(0.0, similarity)

def calculate_size_similarity(size1, size2):
    """
    Calculate similarity between two sizes (0-1, where 1 is identical)
    Uses geometric mean to be scale-invariant
    """
    w1, h1 = size1
    w2, h2 = size2
    
    # Calculate area ratio (handles division by zero)
    area1 = w1 * h1
    area2 = w2 * h2
    
    if area1 == 0 or area2 == 0:
        return 0.0
    
    # Use geometric mean of width and height ratios
    width_ratio = min(w1 / w2, w2 / w1)  # Always <= 1
    height_ratio = min(h1 / h2, h2 / h1)  # Always <= 1
    
    # Geometric mean of ratios
    similarity = np.sqrt(width_ratio * height_ratio)
    return similarity

def calculate_class_similarity(cls1, cls2):
    """
    Calculate similarity between two classes (1.0 if same, 0.0 if different)
    """
    return 1.0 if cls1 == cls2 else 0.0

def get_window_size_advantage(window_size, advantage_smallest=True):
    """
    Calculate advantage factor based on window size
    Smaller window sizes get higher advantage for precision
    """
    if advantage_smallest:
        # Inverse relationship - smallest window gets highest advantage
        if window_size == 320:
            return 1.3  # 30% advantage for smallest window
        elif window_size == 640:
            return 1.1  # 10% advantage for medium window
        else:  # 1280
            return 1.0  # No advantage for largest window
    else:
        return 1.0  # No advantage

def smart_oriented_nms(detections, iou_threshold=0.5, angle_weight=0.3, size_weight=0.3, class_weight=0.2, conf_weight=0.2, advantage_smallest=True):
    """
    Enhanced NMS that considers angle, size, class similarity and confidence
    Gives advantage to smallest window size detections
    """
    if len(detections) == 0:
        return []
    
    # Calculate average angle and size for common components
    class_stats = defaultdict(list)
    for det in detections:
        class_stats[det['cls']].append({
            'angle': det['angle'],
            'width': det['width'],
            'height': det['height']
        })
    
    # Calculate class-wise averages
    class_averages = {}
    for cls, stats_list in class_stats.items():
        if len(stats_list) > 0:
            angles = [s['angle'] for s in stats_list]
            widths = [s['width'] for s in stats_list]
            heights = [s['height'] for s in stats_list]
            
            # Calculate circular mean for angles
            angle_sin = np.mean(np.sin(angles))
            angle_cos = np.mean(np.cos(angles))
            avg_angle = np.arctan2(angle_sin, angle_cos)
            
            class_averages[cls] = {
                'angle': avg_angle,
                'width': np.median(widths),  # Use median for robustness to outliers
                'height': np.median(heights)
            }
    
    def calculate_detection_score(det, class_avg):
        """
        Calculate a combined score considering confidence, angle similarity, size similarity, and window size advantage
        """
        conf_score = det['conf']
        
        if class_avg:
            angle_sim = calculate_angle_similarity(det['angle'], class_avg['angle'])
            size_sim = calculate_size_similarity(
                (det['width'], det['height']), 
                (class_avg['width'], class_avg['height'])
            )
            class_sim = 1.0  # Same class by definition
        else:
            # No class average available, use neutral values
            angle_sim = 0.5
            size_sim = 0.5
            class_sim = 1.0
        
        # Get window size advantage factor
        window_advantage = get_window_size_advantage(det['window_size'], advantage_smallest)
        
        # Combined score with weights and window advantage
        base_score = (
            conf_weight * conf_score +
            angle_weight * angle_sim +
            size_weight * size_sim +
            class_weight * class_sim
        )
        
        # Apply window size advantage
        final_score = base_score * window_advantage
        
        return final_score
    
    # Calculate scores for all detections
    scored_detections = []
    for det in detections:
        class_avg = class_averages.get(det['cls'])
        score = calculate_detection_score(det, class_avg)
        scored_detections.append((score, det))
    
    # Sort by combined score (highest first)
    scored_detections.sort(key=lambda x: x[0], reverse=True)
    
    keep = []
    
    while len(scored_detections) > 0:
        # Take the highest scored detection
        best_score, best_det = scored_detections.pop(0)
        keep.append(best_det)
        
        # Remove overlapping detections
        remaining_detections = []
        for score, det in scored_detections:
            iou = calculate_obb_iou(best_det['xywhr'], det['xywhr'])
            
            # Only remove if they significantly overlap
            if iou < iou_threshold:
                remaining_detections.append((score, det))
        
        scored_detections = remaining_detections
    
    return keep

def oriented_nms_with_size_advantage(detections, iou_threshold=0.5, advantage_smallest=True):
    """
    Standard NMS but with advantage given to smallest window sizes
    """
    if len(detections) == 0:
        return []
    
    # Apply window size advantage to confidence scores
    advantaged_detections = []
    for det in detections:
        advantage = get_window_size_advantage(det['window_size'], advantage_smallest)
        advantaged_conf = det['conf'] * advantage
        advantaged_detections.append((advantaged_conf, det))
    
    # Sort by advantaged confidence (highest first)
    advantaged_detections.sort(key=lambda x: x[0], reverse=True)
    
    keep = []
    
    while len(advantaged_detections) > 0:
        # Take the highest advantaged confidence detection
        best_adv_conf, best_det = advantaged_detections.pop(0)
        keep.append(best_det)
        
        # Remove overlapping detections
        advantaged_detections = [
            (adv_conf, det) for (adv_conf, det) in advantaged_detections 
            if calculate_obb_iou(best_det['xywhr'], det['xywhr']) < iou_threshold
        ]
    
    return keep

def calculate_obb_iou(box1, box2):
    """
    Calculate IoU for two oriented bounding boxes
    """
    # Ensure tensors are on CPU for calculation
    if isinstance(box1, torch.Tensor) and box1.is_cuda:
        box1 = box1.cpu()
    if isinstance(box2, torch.Tensor) and box2.is_cuda:
        box2 = box2.cpu()
        
    x1, y1, w1, h1, angle1 = box1.tolist()
    x2, y2, w2, h2, angle2 = box2.tolist()
    
    box1_aabb = [x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2]
    box2_aabb = [x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2]
    
    return calculate_iou(box1_aabb, box2_aabb)

def calculate_iou(box1, box2):
    """
    Calculate IoU for axis-aligned bounding boxes
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    intersection = inter_width * inter_height
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def draw_obb(image, xywhr, color=(0, 255, 0), thickness=2):
    """
    Draw oriented bounding box on image
    """
    # Ensure tensor is on CPU for drawing
    if isinstance(xywhr, torch.Tensor):
        if xywhr.is_cuda:
            xywhr = xywhr.cpu()
        x_center, y_center, width, height, angle = xywhr.tolist()
    else:
        x_center, y_center, width, height, angle = xywhr
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    w2 = width / 2
    h2 = height / 2
    
    corners = np.array([
        [-w2, -h2],
        [w2, -h2],
        [w2, h2],
        [-w2, h2]
    ])
    
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = corners @ rot_matrix.T
    
    rotated_corners[:, 0] += x_center
    rotated_corners[:, 1] += y_center
    
    rotated_corners = rotated_corners.astype(int)
    
    for i in range(4):
        pt1 = tuple(rotated_corners[i])
        pt2 = tuple(rotated_corners[(i + 1) % 4])
        cv2.line(image, pt1, pt2, color, thickness)
    
    return rotated_corners

def main():
    parser = argparse.ArgumentParser(description='YOLO OBB Detection with Small Window Size Advantage')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights (.pt file)')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='annotated_image.jpg', help='Output image path')
    parser.add_argument('--window-sizes', type=int, nargs='+', default=[320, 640, 1280], 
                       help='Window sizes for multi-scale sliding window')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--overlap', type=float, default=0.25, help='Overlap ratio between windows (0.0-1.0)')
    parser.add_argument('--nms-iou', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--angle-weight', type=float, default=0.3, help='Weight for angle similarity in smart NMS')
    parser.add_argument('--size-weight', type=float, default=0.3, help='Weight for size similarity in smart NMS')
    parser.add_argument('--class-weight', type=float, default=0.2, help='Weight for class similarity in smart NMS')
    parser.add_argument('--conf-weight', type=float, default=0.2, help='Weight for confidence in smart NMS')
    parser.add_argument('--use-smart-nms', action='store_true', help='Use smart NMS considering angle and size')
    parser.add_argument('--no-size-advantage', action='store_true', help='Disable small window size advantage')
    parser.add_argument('--show-class', action='store_true', help='Show class labels')
    parser.add_argument('--show-conf', action='store_true', help='Show confidence scores')
    parser.add_argument('--show-box', action='store_true', help='Show bounding boxes')
    parser.add_argument('--show-deg', action='store_true', help='Show angle in degrees')
    parser.add_argument('--show-scale', action='store_true', help='Show which scale detected the object')
    parser.add_argument('--all', action='store_true', help='Show all annotations')
    
    args = parser.parse_args()
    
    if args.all:
        args.show_class = True
        args.show_conf = True
        args.show_box = True
        args.show_deg = True
    
    model = YOLO(args.model)
    original_image = cv2.imread(args.image)
    
    if original_image is None:
        print(f"Error: Could not load image from {args.image}")
        return
    
    print(f"Original image size: {original_image.shape[1]}x{original_image.shape[0]}")
    print(f"Using window sizes: {args.window_sizes}")
    print(f"Small window advantage: {not args.no_size_advantage}")
    
    all_detections = multi_scale_sliding_window(
        model, original_image, 
        window_sizes=args.window_sizes,
        overlap=args.overlap, 
        conf_threshold=args.conf
    )
    
    print(f"Found {len(all_detections)} detections before NMS")
    
    scale_counts = {}
    for det in all_detections:
        scale = det['window_size']
        scale_counts[scale] = scale_counts.get(scale, 0) + 1
    
    for scale, count in scale_counts.items():
        print(f"  Window size {scale}: {count} detections")
    
    # Use appropriate NMS based on arguments
    advantage_smallest = not args.no_size_advantage
    
    if args.use_smart_nms:
        print("Using smart NMS with angle and size similarity...")
        filtered_detections = smart_oriented_nms(
            all_detections, 
            iou_threshold=args.nms_iou,
            angle_weight=args.angle_weight,
            size_weight=args.size_weight,
            class_weight=args.class_weight,
            conf_weight=args.conf_weight,
            advantage_smallest=advantage_smallest
        )
    else:
        print("Using standard NMS with window size advantage...")
        filtered_detections = oriented_nms_with_size_advantage(
            all_detections, 
            iou_threshold=args.nms_iou,
            advantage_smallest=advantage_smallest
        )
    
    print(f"Found {len(filtered_detections)} detections after NMS")
    
    # Calculate final statistics by window size
    final_scale_counts = {}
    for det in filtered_detections:
        scale = det['window_size']
        final_scale_counts[scale] = final_scale_counts.get(scale, 0) + 1
    
    print("Final detections by window size:")
    for scale in sorted(final_scale_counts.keys()):
        count = final_scale_counts[scale]
        original_count = scale_counts.get(scale, 0)
        percentage = (count / original_count * 100) if original_count > 0 else 0
        print(f"  Window size {scale}: {count} detections ({percentage:.1f}% of original)")
    
    output_image = original_image.copy()
    scale_colors = {320: (0, 255, 0), 640: (255, 255, 0), 1280: (255, 0, 0)}
    
    for i, detection in enumerate(filtered_detections):
        xywhr = detection['xywhr']
        # Ensure tensor is on CPU for processing
        if isinstance(xywhr, torch.Tensor) and xywhr.is_cuda:
            xywhr = xywhr.cpu()
        x_center, y_center, width, height, angle_rad = xywhr.tolist()
        angle_deg = np.degrees(angle_rad) % 180  # Normalize to 0-180
        confidence = detection['conf']
        class_id = detection['cls']
        window_size = detection['window_size']
        
        box_color = scale_colors.get(window_size, (0, 255, 255))
        
        if args.show_box:
            draw_obb(output_image, detection['xywhr'], color=box_color, thickness=2)
        
        texts = []
        if args.show_class:
            texts.append(f"Class: {class_id}")
        if args.show_conf:
            texts.append(f"Conf: {confidence:.3f}")
        if args.show_deg:
            texts.append(f"Angle: {angle_deg:.1f}°")
        if args.show_scale:
            texts.append(f"Scale: {window_size}")
        
        if texts:
            text = " | ".join(texts)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_color = box_color
            
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = int(x_center - text_size[0] // 2)
            text_y = int(y_center - height // 2 - 10)
            
            text_y = max(text_y, 15)
            text_x = max(text_x, 5)
            
            cv2.rectangle(output_image, 
                         (text_x - 2, text_y - text_size[1] - 2),
                         (text_x + text_size[0] + 2, text_y + 2),
                         (0, 0, 0), -1)
            
            cv2.putText(output_image, text, (text_x, text_y), 
                       font, font_scale, text_color, thickness)
        
        scale_info = f", Scale: {window_size}" if args.show_scale else ""
        print(f"Detection {i+1}: Class={class_id}, Angle={angle_deg:.1f}°, Confidence={confidence:.3f}{scale_info}")
    
    cv2.imwrite(args.output, output_image)
    print(f"Image saved to: {args.output}")
    print(f"Small window advantage: {advantage_smallest}")
    if args.use_smart_nms:
        print(f"Smart NMS weights - Angle: {args.angle_weight}, Size: {args.size_weight}, Class: {args.class_weight}, Conf: {args.conf_weight}")

if __name__ == "__main__":
    main()