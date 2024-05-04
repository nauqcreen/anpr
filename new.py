import v8
import v5
import cv2
import math         
import easyocr
import numpy as np
from PIL import Image
import streamlit as st
from datetime import datetime
from deskew import determine_skew


# Set the langua is Anglais
reader = easyocr.Reader(['en'])

# Set the records of what recorded in the car park
parking_records = {}

# Image pre-processing function
def preprocess_image(img):
    # Upscale the image for better resolution
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Non-local Means Denoising
    img = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
    # Adaptive Thresholding for variable lighting
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    # Morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Smoothing the image
    smooth = cv2.GaussianBlur(img, (1, 1), 0)
    return smooth

# Image pre-processing function which has the angle
def deskew_plate(image):
    try:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(grayscale)
        # print(angle)
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1,2] += (width - old_width) / 2
        rot_mat[0,2] += (height - old_height) / 2
        return cv2.warpAffine(image,rot_mat,(int(round(height)),int(round(width))),borderValue=(0, 0, 0))
    except Exception as e:
        print(f"Error deskewing plate: {e}")
        # Return original image //handle error differently
        return image

# Function for read the characters -> output = processed_image
def text(img):
    try:
        text = ''
        for ele in reader.readtext(img, allowlist=".-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            text = text + str(ele[1])
        return text
    except:
        return ' '

# Objects detection function
def detect_objects(image, conf, model_type):
    if model_type == 'v5':
        model = v5.ANPR_V5("models/anpr_v5.pt")
    else:
        model = v8.ANPR_V8("models/anpr_v8.pt")

    plates, image = model.detect(image, conf)
    return plates, image

# Save the check-in time to 'time.csv' file
def check_in(plate_number):
    if plate_number not in parking_records:
        parking_records[plate_number] = {"check_in": datetime.now()}
    else:
        parking_records[plate_number]["check_in"] = datetime.now()
    
    # Save the check-in time to 'time.csv' file
    with open('time.csv', 'a') as f:
        f.write(f"{plate_number},{datetime.now()}\n")

# Save the check-out time with real if image appears 2 times
def check_out(plate_number):
    if plate_number in parking_records:
        parking_records[plate_number]["check_out"] = datetime.now()
    
    # Save the check-out time to 'time.csv' file
    with open('time.csv', 'r') as f:
        data = f.readlines()
    with open('time.csv', 'w') as f:
        for line in data:
            line = line.strip().split(',')
            if line[0] == plate_number and len(line) == 2:
                f.write(f"{line[0]},{line[1]},{datetime.now()}\n")
            else:
                f.write(','.join(line) + '\n')

# Compute the cost based on Check-out - Check-in time
def calculate_parking_fee(plate_number):
    #open the file 'time.csv' and read the data
    with open('time.csv', 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip().split(',')
            if line[0] == plate_number:
                check_in_time = datetime.strptime(line[1], '%Y-%m-%d %H:%M:%S.%f')
                check_out_time = datetime.strptime(line[2], '%Y-%m-%d %H:%M:%S.%f')
                parking_duration = (check_out_time - check_in_time).total_seconds() / 3600  # Convert to hours
                fee_per_hour = 15000  # VND
                return parking_duration * fee_per_hour

# Define local host by Streamlit
def main():
    st.set_page_config(page_title="ANPR using YOLO", page_icon="✨", layout="centered", initial_sidebar_state="expanded")
    st.title('Automatic Number Plate Recognition 🚘🚙')
    st.write('')
    # selected_type = st.sidebar.selectbox('Please select an activity type 🚀', ["Upload Image", "Live Video Feed"])
    st.sidebar.title('Settings 😎')
    conf = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.5, step=0.05)

    # Allow user to select model
    model_type = st.sidebar.selectbox("Select Model", ("v5", "v8"))
    top_image = Image.open('static/banner_top.png')
    top_image = st.sidebar.image(top_image, use_column_width='auto')
    
    # Allow user to select model
    st.write('')
    uploaded_file = st.file_uploader("Upload an image to process",
                                     type=["jpg", "jpeg", "png"],
                                     help="Supported image formats: JPG, JPEG, PNG")
    if uploaded_file is not None:
        top_image.empty()
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        st.sidebar.text("Uploaded Image:")
        st.sidebar.image(image, caption=" ", use_column_width=True)
        image = deskew_plate(image)
        plates, output_image = detect_objects(image, conf, model_type)
        if len(plates) > 0:
            for plate in plates:
                x1, y1, x2, y2, conf = plate
                crop_img = output_image[y1:y2, x1:x2]
                processed_img = preprocess_image(crop_img)  # 'smooth' of image pre-processing function
                plate_text = text(processed_img)

                # Handle parking logic based on plate text
                if plate_text not in parking_records:
                    check_in(plate_text)
                    st.sidebar.text(f"Check-in for {plate_text} recorded.")
                else:
                    check_out(plate_text)
                    fee = calculate_parking_fee(plate_text)
                    parking_duration = (parking_records[plate_text]['check_out'] - parking_records[plate_text]['check_in']).total_seconds() / 3600
                    st.sidebar.text(f"Check-out for {plate_text}. Parking duration: {parking_duration:.2f} hours. Fee: {fee:.2f} VND")
                    
                #display time check-in and check-out, fee from 'time.csv' file
                with open('time.csv', 'r') as f:
                    data = f.readlines()
                    for line in data:
                        line = line.strip().split(',')
                        if line[0] == plate_text:
                            st.sidebar.text(f"Check-in time: {line[1]}")
                        
                        # if image appears 2 times; display the check-out time
                        if line[0] == plate_text and len(line) == 3:
                            st.sidebar.text(f"Check-out time: {line[2]}")
                            st.sidebar.text(f"Total time: {line[2]} - {line[1]}")
                            st.sidebar.text(f"Total fee: {fee:.2f} VND")
                            
                            
                
                # Display plate and detection confidence
                st.sidebar.text('Number plates detected:')
                st.sidebar.image(processed_img, caption=f"Plate: {plate_text} - Detection Confidence: {conf}", use_column_width=True)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            st.subheader("No License Plates Detected")
        st.image(output_image, caption="OUTPUT IMAGE")
        
        


if __name__ == "__main__":
    main()
