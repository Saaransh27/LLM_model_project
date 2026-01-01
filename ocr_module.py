# ocr_module.py
import cv2
import numpy as np
import easyocr
from PIL import Image

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def preprocess_image(image):
    # Convert to grayscale
    image_array = np.array(image)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    elif len(image_array.shape) == 2:  # Image is already in grayscale
        gray = image_array
    else:
        raise ValueError("Unexpected number of channels in the input image")

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return adaptive_thresh

def detect_text(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Use EasyOCR to extract text
    results = reader.readtext(processed_image)

    # Extract and concatenate text from EasyOCR results
    ocr_text = ""
    for (bbox, text, prob) in results:
        ocr_text += text + " "

    return ocr_text.strip()
