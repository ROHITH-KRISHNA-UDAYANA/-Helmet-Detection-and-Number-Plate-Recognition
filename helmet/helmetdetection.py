import torch
import cv2
import os

# Function to load the YOLOv5 model
def load_yolo_model():
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

# Function to detect helmets and annotate the image
def detect_helmets(image_path, model):
    # Perform inference
    results = model(image_path)
    
    # Get detection details
    detections = results.xyxy[0].numpy()
    
    # Load image with OpenCV
    image = cv2.imread(image_path)
    
    # Access class names
    class_names = model.names
    
    helmet_detected = False
    
    # Process detections
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        label = class_names[int(class_id)]
        
        if label == 'helmet' and confidence > 0.5:  # Adjust confidence threshold as needed
            helmet_detected = True
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green border
            print(f"Detected helmet with confidence {confidence:.2f}")
    
    if not helmet_detected:
        # Draw a red border around the entire image
        cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), 5)  # Red border
    
    return image

# Function to process all frames in a folder
def process_frames(frame_folder, model, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_files = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    for frame_file in frame_files:
        annotated_image = detect_helmets(frame_file, model)
        
        if annotated_image is not None:
            output_path = os.path.join(output_folder, os.path.basename(frame_file))
            cv2.imwrite(output_path, annotated_image)
            print(f"Processed {frame_file}")
        else:
            print(f"Skipped {frame_file}")

# Main function
if __name__ == "__main__":
    # Load YOLOv5 model
    model = load_yolo_model()
    
    # Replace with the path to the folder where the frames are stored
    frame_folder = r"D:\helmet\Photos"

    # Replace with the path to the folder where the output images will be saved
    output_folder = r"D:\helmet\Photos"

    # Process frames
    process_frames(frame_folder, model, output_folder)
