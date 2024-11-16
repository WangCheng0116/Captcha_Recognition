import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import glob
from concurrent.futures import ThreadPoolExecutor
import warnings
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

def remove_lines(img):
    """
    Remove noisy lines from the image using advanced morphological operations
    """
    # Create horizontal and vertical kernels
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    
    # Detect lines
    horizontal_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combine lines
    lines = cv2.add(horizontal_lines, vertical_lines)
    
    # Remove lines from original image
    result = cv2.subtract(img, lines)
    
    return result



def preprocess_image(image_path, debug=False):
    """
    Enhanced preprocessing with debug visualization
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None
        
    if debug:
        print(f"Original image shape: {img.shape}")
        print(f"Value range: [{img.min()}, {img.max()}]")

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Apply adaptive thresholding with more lenient parameters
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 10  # Increased block size and C value
    )
    
    if debug:
        print(f"Threshold result - White pixels: {np.sum(thresh == 255)}")
    
    # Clean up using morphological operations
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def segment_characters_advanced(preprocessed_img, debug=False):
    """
    Improved character segmentation with more lenient parameters and debug info
    """
    if preprocessed_img is None:
        return []
        
    height, width = preprocessed_img.shape
    if debug:
        print(f"Input image shape: {preprocessed_img.shape}")
        print(f"Non-zero pixels: {np.count_nonzero(preprocessed_img)}")

    # Find contours directly
    contours, _ = cv2.findContours(
        preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if debug:
        print(f"Found {len(contours)} initial contours")

    # Filter and sort character regions
    char_regions = []
    min_area = height * width * 0.001  # Further reduced minimum area threshold
    max_area = height * width * 0.5    # Further increased maximum area threshold
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # More lenient filtering criteria
        aspect_ratio = w / h if h > 0 else 0
        if (min_area < area < max_area and 
            0.05 < aspect_ratio < 5.0 and  # Further lenient aspect ratio
            h > height * 0.1):           # Reduced minimum height requirement
            
            char_regions.append((x, y, w, h))
            
            if debug:
                print(f"Accepted region: x={x}, y={y}, w={w}, h={h}, "
                      f"area={area}, aspect_ratio={aspect_ratio:.2f}")

    if debug:
        print(f"Found {len(char_regions)} valid character regions")

    # Sort regions from left to right
    char_regions.sort(key=lambda x: x[0])

    # Extract characters
    characters = []
    for x, y, w, h in char_regions:
        # Add padding
        padding = 2
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(width, x + w + padding)
        y_end = min(height, y + h + padding)
        
        char_img = preprocessed_img[y_start:y_end, x_start:x_end]
        
        # Resize to fixed dimensions
        target_size = (32, 32)
        try:
            resized_char = cv2.resize(char_img, target_size, 
                                    interpolation=cv2.INTER_AREA)
            characters.append(resized_char)
        except Exception as e:
            if debug:
                print(f"Failed to resize character: {str(e)}")
            continue

    return characters
def extract_features(char_img):
    """
    Enhanced feature extraction with robust error handling
    """
    try:
        # Ensure correct image format
        if char_img.dtype == np.float32:
            char_img = (char_img * 255).astype(np.uint8)
        
        # Ensure correct dimensions
        if len(char_img.shape) == 2:
            char_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2BGR)
        
        # Resize to expected HOG dimensions if necessary
        if char_img.shape[:2] != (32, 32):
            char_img = cv2.resize(char_img, (32, 32), interpolation=cv2.INTER_LANCZOS4)
        
        # 1. HOG features with error checking
        win_size = (32, 32)
        cell_size = (8, 8)
        block_size = (16, 16)
        block_stride = (8, 8)
        num_bins = 9
        
        try:
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
            hog_features = hog.compute(char_img)
            if hog_features is None:
                hog_features = np.zeros((36,))  # Default size for 32x32 image
        except Exception as e:
            print(f"HOG extraction failed: {str(e)}")
            hog_features = np.zeros((36,))
        
        # 2. Pixel density features
        height, width = char_img.shape[:2]
        density_features = []
        regions = [0, 0.25, 0.5, 0.75, 1.0]
        
        for i in range(len(regions)-1):
            for j in range(len(regions)-1):
                y_start = int(regions[i] * height)
                y_end = int(regions[i+1] * height)
                x_start = int(regions[j] * width)
                x_end = int(regions[j+1] * width)
                
                region = char_img[y_start:y_end, x_start:x_end]
                if len(region.shape) == 3:
                    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                density = np.sum(region) / (255.0 * region.size) if region.size > 0 else 0
                density_features.append(density)
        
        # 3. Contour features with error handling
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY) if len(char_img.shape) == 3 else char_img
        try:
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            else:
                area, perimeter, circularity = 0, 0, 0
        except Exception as e:
            print(f"Contour extraction failed: {str(e)}")
            area, perimeter, circularity = 0, 0, 0
        
        contour_features = [
            area/(width*height) if width*height > 0 else 0,
            perimeter/(width+height) if width+height > 0 else 0,
            circularity
        ]
        
        # Combine all features
        all_features = np.concatenate([
            hog_features.flatten(),
            density_features,
            contour_features
        ])
        
        # Ensure features are valid
        all_features = np.nan_to_num(all_features)  # Replace NaN with 0
        return all_features
        
    except Exception as e:
        print(f"Feature extraction failed: {str(e)}")
        # Return a zero vector of expected size
        return np.zeros((36 + 16 + 3,))  # HOG + density + contour features

def process_image_file(image_path, debug=False):
    """Process a single image file with debug information"""
    try:
        # Extract label from filename
        label = os.path.basename(image_path).split('-')[0]
        if debug:
            print(f"\nProcessing {image_path}")
            print(f"Expected label: {label}")
        
        # Preprocess
        preprocessed = preprocess_image(image_path, debug)
        if preprocessed is None:
            print(f"Preprocessing failed for {image_path}")
            return None, None
            
        # Debug visualization if needed
        if debug:
            plt.figure(figsize=(12, 4))
            plt.subplot(121)
            plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
            plt.title('Original')
            plt.subplot(122)
            plt.imshow(preprocessed, cmap='gray')
            plt.title('Preprocessed')
            plt.show()
        
        # Segment characters
        characters = segment_characters_advanced(preprocessed, debug)
        if not characters:
            print(f"No characters found in {image_path}")
            return None, None
            
        if debug:
            print(f"Found {len(characters)} characters, expected {len(label)}")
            # Visualize segmented characters
            plt.figure(figsize=(15, 3))
            for i, char in enumerate(characters):
                plt.subplot(1, len(characters), i+1)
                plt.imshow(char, cmap='gray')
                if i < len(label):
                    plt.title(f'Char {i+1}: {label[i]}')
                plt.axis('off')
            plt.show()
        
        # Extract features
        features = []
        for char_img in characters:
            feat = extract_features(char_img)
            if feat is not None and feat.size > 0:
                features.append(feat)
                
        if len(features) != len(label):
            if debug:
                print(f"Mismatch: {len(features)} features vs {len(label)} labels")
            return None, None
            
        return features, list(label)
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None

# 首先修改导入语句,从 sklearn.svm import SVC 改为:
from sklearn.neighbors import KNeighborsClassifier

# train_model函数需要修改如下:
def train_model(train_paths):
    """Train KNN model with detailed progress information"""
    print("Processing training images...")
    X_train = []
    y_train = []
    
    # Process a single image with debug first
    if train_paths:
        print("\nDebug processing first image:")
        features, labels = process_image_file(train_paths[0], debug=True)
        if features is not None:
            X_train.extend(features)
            y_train.extend(labels)
    
    # Process remaining images
    successful_count = 0
    failed_paths = []
    
    for image_path in tqdm(train_paths[1:]):
        features, labels = process_image_file(image_path)
        if features is not None and labels is not None:
            X_train.extend(features)
            y_train.extend(labels)
            successful_count += 1
        else:
            failed_paths.append(image_path)
    
    print(f"\nProcessing summary:")
    print(f"Successfully processed: {successful_count+1}/{len(train_paths)} images")
    print(f"Failed images: {len(failed_paths)}")
    if failed_paths:
        print("First few failed images:")
        for path in failed_paths[:5]:
            print(f"  - {path}")
    
    if not X_train:
        raise ValueError("No valid features extracted from training images")
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"\nFeature matrix shape: {X_train.shape}")
    print(f"Labels shape: {y_train.shape}")
    
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Training KNN model...")
    # 将SVM替换为KNN,设置适当的参数
    knn = KNeighborsClassifier(
        n_neighbors=5,        # 设置K值
        weights='distance',   # 使用距离加权
        algorithm='auto',     # 自动选择最优算法
        n_jobs=-1            # 使用所有CPU核心
    )
    knn.fit(X_train_scaled, y_train)
    
    return knn, scaler


def evaluate_model(model, scaler, test_paths):
    """
    Evaluate model without skipping any test images.
    If segmentation produces wrong number of characters, predict what we can.
    """
    print("\nEvaluating model on test set...")
    char_correct = 0
    char_total = 0
    captcha_correct = 0
    captcha_total = 0
    segmentation_issues = 0
    
    for image_path in tqdm(test_paths):
        try:
            # Extract ground truth label
            true_label = os.path.basename(image_path).split('-')[0]
            
            # Preprocess image
            preprocessed = preprocess_image(image_path)
            characters = segment_characters_advanced(preprocessed)
            
            # Track segmentation issues but don't skip
            if len(characters) != len(true_label):
                segmentation_issues += 1
            
            # If no characters found, count as all incorrect
            if len(characters) == 0:
                char_total += len(true_label)
                captcha_total += 1
                continue
                
            # Extract features and scale
            features = [extract_features(char) for char in characters]
            features_scaled = scaler.transform(features)
            
            # Predict available characters
            predicted_labels = model.predict(features_scaled)
            
            # Compare as many characters as we can
            min_len = min(len(predicted_labels), len(true_label))
            char_correct += sum(p == t for p, t in zip(predicted_labels[:min_len], true_label[:min_len]))
            
            # For remaining characters in true_label, count as incorrect
            char_total += len(true_label)  # Count all expected characters
            
            # CAPTCHA is correct only if perfect match
            if len(predicted_labels) == len(true_label) and all(p == t for p, t in zip(predicted_labels, true_label)):
                captcha_correct += 1
            captcha_total += 1
            
        except Exception as e:
            print(f"\nError processing {image_path}: {str(e)}")
            # Count as incorrect but don't skip
            char_total += len(true_label)
            captcha_total += 1
            continue
    
    # Calculate metrics
    char_accuracy = char_correct / char_total * 100 if char_total > 0 else 0
    captcha_accuracy = captcha_correct / captcha_total * 100 if captcha_total > 0 else 0
    
    # Print detailed report
    print(f"\nEvaluation Results:")
    print(f"Total images in test set: {len(test_paths)}")
    print(f"Images with segmentation issues: {segmentation_issues}")
    print(f"Character-level accuracy: {char_accuracy:.2f}%")
    print(f"CAPTCHA-level accuracy: {captcha_accuracy:.2f}%")
    
    return {
        'char_accuracy': char_accuracy,
        'captcha_accuracy': captcha_accuracy,
        'segmentation_issues': segmentation_issues,
        'total_images': len(test_paths)
    }

def main():
    # Get file paths
    train_paths = glob.glob('train/*-0.png')
    test_paths = glob.glob('test/*-0.png')
    
    print(f"Found {len(train_paths)} training images and {len(test_paths)} test images")
    
    # Train KNN model instead of SVM
    model, scaler = train_model(train_paths)
    
    # Evaluate model
    evaluate_model(model, scaler, test_paths)

if __name__ == "__main__":
    main()
