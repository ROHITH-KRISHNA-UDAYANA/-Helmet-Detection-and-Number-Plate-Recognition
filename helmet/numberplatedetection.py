import cv2
import pytesseract
import os

# Configure the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def extract_number_plate_text(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours based on edges detected
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over contours to find a potential license plate region
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Assuming license plate has an aspect ratio between 2 and 6
        if 2 < aspect_ratio < 6:
            license_plate = gray[y:y + h, x:x + w]
            
            # Use Tesseract to do OCR on the selected region
            text = pytesseract.image_to_string(license_plate, config='--psm 8')
            return text
    
    return ""

def process_frames(frame_folder, output_folder):
    if not os.path.exists(frame_folder):
        print(f"Frame folder does not exist: {frame_folder}")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_files = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    for frame_file in frame_files:
        image = cv2.imread(frame_file)
        if image is None:
            print(f"Failed to load image: {frame_file}")
            continue

        # Add a red border around the entire image
        height, width, _ = image.shape
        cv2.rectangle(image, (0, 0), (width, height), (0, 0, 255), 10)
        
        # Extract number plate text
        number_plate_text = extract_number_plate_text(image)
        
        output_path = os.path.join(output_folder, os.path.basename(frame_file))
        cv2.imwrite(output_path, image)
        
        print(f"Processed frame: {frame_file}")
        if number_plate_text:
            print(f"Extracted number plate text: {number_plate_text}")
        else:
            print(f"No number plate text detected in frame: {frame_file}")

# Replace with the path to the folder where the frames are stored
frame_folder = r"D:\\helmet\\Photos"

# Replace with the path to the folder where the output images will be saved
output_folder = r"D:\\helmet\\Photos"

process_frames(frame_folder, output_folder)
