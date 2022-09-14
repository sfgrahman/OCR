import numpy as np
import cv2
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Convert the image to grayscale and store the image to memory
invoice = cv2.imread('invoice.jpeg', cv2.IMREAD_GRAYSCALE)

# Show the image
#cv2.imshow('Invoice', invoice)
#cv2.waitKey(0)

# Apply contrast
(thresh, black_white_img) = cv2.threshold(invoice, 127, 255, cv2.THRESH_BINARY)

# Apply Gaussian blur for smoothing to rid the image of artifacts
blur = cv2.GaussianBlur(black_white_img,(9,9),0)

# Apply threshold to bring contours to prominence
th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

ret,thresh = cv2.threshold(th,127,255,0)

# Read our original invoice image to show the contours drawn in green
original_invoice_img = cv2.imread('invoice.jpeg')

# Find the contours in the image and store them in an array
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
invoice_contours = cv2.drawContours(original_invoice_img, contours, -1, (0, 255, 0), 3)

# Show the contours
#cv2.imshow('Contours', invoice_contours)
#cv2.waitKey(0)

# Iterate through our contours
index = 0
for contour in contours:
    index += 1
    # We want to find any contour with an area of 
    # greater than 28,800 pixels since we know that
    # the smallest contour we're looking for is at minimum
    # 320px wide x 90 px tall (the lower end of our
    # range for our smallest contour's dimensions)
    if cv2.contourArea(contour) > 28800:
        # Get the dimension of the contour
        x,y,w,h = cv2.boundingRect(contour)
        cropped_img_contours = original_invoice_img[y:y+h,x:x+w]
        cropped_img = invoice[y:y+h,x:x+w]
        
        # Identify the top right corner box by using an 'if' statement. We are looking for a contour with a width between 320px and 400px in width, and between 90px and 140px in height. 
        if w <= 400 and w > 320 and h <= 140 and h > 90:
            cv2.imshow('Top Right Corner Box Contours', cropped_img_contours)
            cv2.imshow('Top Right Corner Box', cropped_img)
            cv2.waitKey(0)
            cv2.imwrite('top_right_corner_box.jpeg', cropped_img)
            
        if w <= 1400 and w > 1300 and h <= 375 and h > 325: 
            cv2.imshow('Line Items Section Contours', cropped_img_contours)
            cv2.imshow('Line Items Section', cropped_img)
            cv2.waitKey(0)
            cv2.imwrite('line_items_section.jpeg', cropped_img)
       
        

# Function to sanitize text
def sanitize_text(image_path):
    # Store the extracted text from the image in a string.
    # Then, split the string into an array using the '\n' character as a delimiter
    raw_text_values = pytesseract.image_to_string(Image.open(image_path)).split('\n')
    # Remove any spaces or empty values
    for i in range(len(raw_text_values) - 1, -1, -1):
        if raw_text_values[i] == '' or raw_text_values[i] == ' ':
            del raw_text_values[i]
    return raw_text_values


top_right_corner_values = sanitize_text('top_right_corner_box.jpeg')
line_items_text = sanitize_text('line_items_section.jpeg')


# Print the text to the console to view our results
print(top_right_corner_values)
print(line_items_text)

