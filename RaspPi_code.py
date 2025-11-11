#!/usr/bin/env python3
import cv2
import numpy as np
import tensorflow.lite as tflite
import serial
import time
from collections import Counter


# -------- CONFIG --------
MODEL_PATH = "trashnet_model.tflite"
LABELS = ["glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = 128
CONF_THRESHOLD = 0.4
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
VOTE_WINDOW = 5.0  # seconds for averaging
CENTER_RATIO=0.4
# ------------------------

# Load model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Init serial
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
print("Serial connected to Arduino.")

# Init camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open camera")
print("Camera started. Press 'q' to quit.")

# State variables
class_buffer = []
window_start = time.time()
skip_processing_until = 0  # Timestamp until which to skip processing

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not available.")
        break
    
    h, w, _ = frame.shape
    ch, cw = int(h * CENTER_RATIO), int(w * CENTER_RATIO)
    y1 = (h - ch) // 2
    y2 = y1 + ch
    x1 = (w - cw) // 2
    x2 = x1 + cw
    

    # Draw the ROI box with bright green color
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Check if we should skip processing due to Arduino activity
    current_time = time.time()
    if current_time < skip_processing_until:
        # Skip all processing - Arduino is busy
        remaining_time = skip_processing_until - current_time
        cv2.putText(frame, f"Arduino busy - waiting {remaining_time:.1f}s", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.imshow("TrashNet Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Crop only the center region for inference
    center_crop = frame[y1:y2, x1:x2]

    # Check if center crop has too much blue color (skip inference)
    # Convert to HSV for better blue detection
    hsv_crop = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    
    # Define blue color range in HSV
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create mask for blue pixels
    blue_mask = cv2.inRange(hsv_crop, lower_blue, upper_blue)
    
    # Calculate percentage of blue pixels
    total_pixels = center_crop.shape[0] * center_crop.shape[1]
    blue_pixels = cv2.countNonZero(blue_mask)
    blue_percentage = (blue_pixels / total_pixels) * 100
    
    if blue_percentage > 95.0:
        # Skip inference - too much blue
        cv2.putText(frame, f"Place object in center", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        # Preprocess
        img = cv2.resize(center_crop, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
        idx = np.argmax(preds)
        conf = preds[idx]
        label = LABELS[idx]

        # Display classification info
        color = (0, 255, 0) if conf > CONF_THRESHOLD else (0, 0, 255)
        if label not in ["paper"]:
            cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        
    
    cv2.imshow("TrashNet Live", frame)

    # Collect detections for 3 seconds (only if inference was performed)
    if blue_percentage <= 80.0 and conf > CONF_THRESHOLD and label not in ["paper"]:
        class_buffer.append(label)

    # Check if 3-second window is over
    if time.time() - window_start >= VOTE_WINDOW and class_buffer:
        # Count most common class in the window
        common_label = Counter(class_buffer).most_common(1)[0][0]

        # Decide recyclable or trash
        if common_label in ["plastic", "glass", "metal"]:
            signal = "1\n"
        elif common_label in ["trash"]:
            signal = "2\n"
        else:
            signal = "0\n"
      
        ser.write(signal.encode('utf-8'))
        print(f"[VOTE RESULT] {common_label.upper()} -> Sent to { 'recycle' if signal.strip() == '1' else 'trash' }")
        # Skip processing for the next 5 seconds while Arduino is busy
        skip_processing_until = time.time() + 5.0

        # Reset window
        class_buffer.clear()
        window_start = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()


