import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import keras

# Custom CSS for styling
st.markdown("""
    <style>
        /* Set background color for the entire page */
        body {
            background-color: #121C59 !important;
            color: white;
            text-align: center;
        }
        .stApp {
            background-color: #121C59 !important;
            color: white !important;
        }
        /* Ensure the title has white text */
        .stTitle {
            color: white !important;
        }
        /* Styling for the uploaded image */
        .stImage {
            margin-top: 20px;
        }
        /* Style buttons with white text */
        .stButton > button {
            --bg: #12188a;
            --text-color: #fff;
            position: relative;
            width: 200px;
            border: none;
            background: var(--bg);
            color: var(--text-color);
            padding: -40px 60px;
            font-weight: bold;
            text-transform: normal;
            transition: 0.2s;
            border-radius: 5px;
            opacity: 0.8;
            letter-spacing: 1px;
            box-shadow: #5b61c7 2px 5px 2px, #000 0px 8px 5px;
        }
</style>
""", unsafe_allow_html=True)

# Add Logo
st.image("Logo.png", use_container_width=True)

# Load the Pretrained Model
model_path = "model_dl.h5"
model = tf.keras.models.load_model(model_path)

# Define Class Labels
class_labels = ["Full Stop", "Give Way", "Give Way for oncoming traffic", "Priority Over Oncoming Traffic", "Priority Road"]

# Custom CSS for the
st.markdown('<h1 style="font-family: \'Libre Baskerville\', serif; font-weight: bold; font-size: 50px; color: white;">Seek the innovation</h1>', unsafe_allow_html=True)

# 5️⃣ New text under the previous one, regular weight and smaller size
st.markdown('<h2 style="font-family: \'Libre Baskerville\', serif; font-weight: normal; font-size: 35px; color: white; margin-top: -20px">Explore the new image recognition system for traffic signs</h2>', unsafe_allow_html=True)

# Add horizontal line
st.markdown("<hr style='border: 1px solid white; width: 90%; margin: 20px auto;'>", unsafe_allow_html=True)

# 6️⃣ Add the "Take a picture" button (only one now) with a unique key
if st.button('📸 Take a Picture', key="camera_button"):
    st.session_state["take_picture"] = True

# "or" text in the middle
st.markdown('<h3 style="color: white;">or</h3>', unsafe_allow_html=True)

# Image uploader button with a unique key
uploaded_file = st.file_uploader("Choose a traffic sign image...", type=["jpg", "png", "jpeg"], key="upload_image")

# Initialize session state if it's not present
if "take_picture" not in st.session_state:
    st.session_state["take_picture"] = False
if "show_predictions" not in st.session_state:
    st.session_state["show_predictions"] = False
if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None
if "camera_image_taken" not in st.session_state:  # Add this state
    st.session_state["camera_image_taken"] = False  # Default to False

# 7️⃣ Camera input page
if st.session_state["take_picture"]:
    # Allow the user to take a picture from the camera
    camera_image = st.camera_input("Take a picture of a traffic sign")

    if camera_image is not None:
            # Process the image for prediction
            original_camera_image = Image.open(camera_image).convert("RGB")
            st.image(original_camera_image, caption="Captured Image", use_container_width=True)

            expected_input_camera_shape = (128,128)
            resized_camera_image = original_camera_image.resize(expected_input_camera_shape)

            # Convert resized image to NumPy array and normalize
            img_array = np.array(resized_camera_image, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)

            # Load EfficientNetB0 for feature extraction
            base_model_efficientnet = keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(128, 128, 3))
            
            # Preprocess image using EfficientNetB0 preprocessing
            preprocessed_image = keras.applications.efficientnet.preprocess_input(img_array)

            # Extract features using the EfficientNet base model
            extracted_features = base_model_efficientnet.predict(preprocessed_image)
    
            # Make prediction
            predictions = model.predict(extracted_features)
            predicted_class = np.argmax(predictions)

            # Set the state to true when the image is captured
            st.session_state["camera_image_taken"] = True

            # Display prediction result with specific directions
            st.markdown(f"""
        <div style="font-size: 30px; font-weight: bold;">💡 Prediction: {class_labels[predicted_class]}</div>
        <div style="font-size: 24px; font-weight: normal;">📊 Confidence: {np.max(predictions) * 100:.2f}%</div>
    """, unsafe_allow_html=True)
            
            # Show specific traffic rule and recommended action based on predicted class
            if class_labels[predicted_class] == "Full Stop":
                st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚨 Traffic Rule Explanation:</div>", unsafe_allow_html=True)
                st.write("<div style='font-size: 24px;'>This sign indicates that you must come to a complete stop at the intersection, regardless of whether there are other vehicles or pedestrians. You cannot proceed until it is safe to do so.</div>", unsafe_allow_html=True)
                st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚗 Recommended Action:</div>", unsafe_allow_html=True)
                st.write("<div style='font-size: 24px;'>Stop completely at the intersection, look for oncoming traffic or pedestrians, and only proceed when the way is clear. Make sure to check in all directions before moving forward.</div>", unsafe_allow_html=True)

            elif class_labels[predicted_class] == "Give Way":
                st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚨 Traffic Rule Explanation:</div>", unsafe_allow_html=True)
                st.write("<div style='font-size: 24px;'>This sign indicates that you must yield to all other traffic before proceeding. You do not have the right of way and must slow down or stop if necessary to let other vehicles or pedestrians pass.</div>", unsafe_allow_html=True)
                st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚗 Recommended Action:</div>", unsafe_allow_html=True)
                st.write("<div style='font-size: 24px;'>Approach the intersection carefully, check for oncoming traffic, and only proceed when it is safe to do so. Be aware of other traffic regulations in effect on the road as you drive.</div>", unsafe_allow_html=True)

            elif class_labels[predicted_class] == "Give Way for oncoming traffic":
                st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚨 Traffic Rule Explanation:</div>", unsafe_allow_html=True)
                st.write("<div style='font-size: 24px;'>This sign means that you must yield to oncoming traffic. You have to slow down or stop to let vehicles approaching from the opposite direction pass safely before proceeding.</div>", unsafe_allow_html=True)
                st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚗 Recommended Action:</div>", unsafe_allow_html=True)
                st.write("<div style='font-size: 24px;'>Wait for a clear gap in the oncoming traffic before proceeding. Ensure it is safe to continue and be cautious when making turns or entering intersections.</div>", unsafe_allow_html=True)

            elif class_labels[predicted_class] == "Priority Over Oncoming Traffic":
                st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚨 Traffic Rule Explanation:</div>", unsafe_allow_html=True)
                st.write("<div style='font-size: 24px;'>This sign indicates that vehicles on your side of the road have priority over oncoming traffic. Opposing vehicles must yield until your lane is clear.</div>", unsafe_allow_html=True)
                st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚗 Recommended Action:</div>", unsafe_allow_html=True)
                st.write("<div style='font-size: 24px;'>Proceed with caution, but maintain your right of way. Be prepared for oncoming vehicles that may not yield. Be aware of other traffic regulations in effect on the road as you drive.</div>", unsafe_allow_html=True)

            elif class_labels[predicted_class] == "Priority Road":
                st.write(f"<div style='font-size: 26px; font-weight: bold;margin-top:30px'>🚨 Traffic Rule Explanation:</div>", unsafe_allow_html=True)
                st.write("<div style='font-size: 24px;'>This sign indicates that you are on a priority road, meaning that vehicles on intersecting roads must yield to you. You have the right of way when entering intersections.</div>", unsafe_allow_html=True)
                st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚗 Recommended Action:</div>", unsafe_allow_html=True)
                st.write("<div style='font-size: 24px;'>Continue driving without slowing down for intersecting traffic, but always be alert for potential hazards or unexpected vehicles. Be aware of other traffic regulations in effect on the road as you drive.</div>", unsafe_allow_html=True)

# 8️⃣ If the user uploads a file (new page navigation)
if uploaded_file is not None:
    # Store the uploaded image in session state for later use
    st.session_state["uploaded_image"] = uploaded_file
    # Set flag to show predictions page
    st.session_state["show_predictions"] = True
    st.session_state["camera_image_taken"] = False  # Ensure this is reset when switching to upload

# 9️⃣ Display uploaded image and predictions on a new page
if st.session_state["show_predictions"]:
    uploaded_image = st.session_state["uploaded_image"]

    # Open and convert the image to RGB
    original_image = Image.open(uploaded_image).convert("RGB")
    st.image(original_image, use_container_width=True)

    # Ensure the input shape matches EfficientNetB0 requirements
    expected_input_shape = (128, 128)  # EfficientNetB0 input size
    resized_image = original_image.resize(expected_input_shape)
    
    # Convert resized image to NumPy array and normalize
    img_array = np.array(resized_image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Load EfficientNetB0 for feature extraction
    base_model_efficientnet = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(128, 128, 3))
    
    # Preprocess image using EfficientNetB0 preprocessing
    preprocessed_image = keras.applications.efficientnet.preprocess_input(img_array)
    
    # Extract features using the EfficientNet base model
    extracted_features = base_model_efficientnet.predict(preprocessed_image)
    
    # Load the trained model
    model_path = "model_dl.h5"
    model = tf.keras.models.load_model(model_path)
    
    # Make prediction using the loaded model
    predictions = model.predict(extracted_features)
    predicted_class = np.argmax(predictions)

    # Display prediction result with specific directions
    st.markdown(f"""
        <div style="font-size: 30px; font-weight: bold;">💡 Prediction: {class_labels[predicted_class]}</div>
        <div style="font-size: 24px; font-weight: normal;">📊 Confidence: {np.max(predictions) * 100:.2f}%</div>
    """, unsafe_allow_html=True)
    # Show specific traffic rule and recommended action based on predicted class
    if class_labels[predicted_class] == "Full Stop":
        st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚨 Traffic Rule Explanation:</div>", unsafe_allow_html=True)
        st.write("<div style='font-size: 24px;'>This sign indicates that you must come to a complete stop at the intersection, regardless of whether there are other vehicles or pedestrians. You cannot proceed until it is safe to do so.</div>", unsafe_allow_html=True)
        st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚗 Recommended Action:</div>", unsafe_allow_html=True)
        st.write("<div style='font-size: 24px;'>Stop completely at the intersection, look for oncoming traffic or pedestrians, and only proceed when the way is clear. Make sure to check in all directions before moving forward.</div>", unsafe_allow_html=True)

    elif class_labels[predicted_class] == "Give Way":
        st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚨 Traffic Rule Explanation:</div>", unsafe_allow_html=True)
        st.write("<div style='font-size: 24px;'>This sign indicates that you must yield to all other traffic before proceeding. You do not have the right of way and must slow down or stop if necessary to let other vehicles or pedestrians pass.</div>", unsafe_allow_html=True)
        st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚗 Recommended Action:</div>", unsafe_allow_html=True)
        st.write("<div style='font-size: 24px;'>Approach the intersection carefully, check for oncoming traffic, and only proceed when it is safe to do so. Be aware of other traffic regulations in effect on the road as you drive.</div>", unsafe_allow_html=True)

    elif class_labels[predicted_class] == "Give Way for oncoming traffic":
        st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚨 Traffic Rule Explanation:</div>", unsafe_allow_html=True)
        st.write("<div style='font-size: 24px;'>This sign means that you must yield to oncoming traffic. You have to slow down or stop to let vehicles approaching from the opposite direction pass safely before proceeding.</div>", unsafe_allow_html=True)
        st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚗 Recommended Action:</div>", unsafe_allow_html=True)
        st.write("<div style='font-size: 24px;'>Wait for a clear gap in the oncoming traffic before proceeding. Ensure it is safe to continue and be cautious when making turns or entering intersections.</div>", unsafe_allow_html=True)

    elif class_labels[predicted_class] == "Priority Over Oncoming Traffic":
        st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚨 Traffic Rule Explanation:</div>", unsafe_allow_html=True)
        st.write("<div style='font-size: 24px;'>This sign indicates that vehicles on your side of the road have priority over oncoming traffic. Opposing vehicles must yield until your lane is clear.</div>", unsafe_allow_html=True)
        st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚗 Recommended Action:</div>", unsafe_allow_html=True)
        st.write("<div style='font-size: 24px;'>Proceed with caution, but maintain your right of way. Be prepared for oncoming vehicles that may not yield. Be aware of other traffic regulations in effect on the road as you drive.</div>", unsafe_allow_html=True)

    elif class_labels[predicted_class] == "Priority Road":
        st.write(f"<div style='font-size: 26px; font-weight: bold;margin-top:30px'>🚨 Traffic Rule Explanation:</div>", unsafe_allow_html=True)
        st.write("<div style='font-size: 24px;'>This sign indicates that you are on a priority road, meaning that vehicles on intersecting roads must yield to you. You have the right of way when entering intersections.</div>", unsafe_allow_html=True)
        st.write(f"<div style='font-size: 26px; font-weight: bold; margin-top:30px'>🚗 Recommended Action:</div>", unsafe_allow_html=True)
        st.write("<div style='font-size: 24px;'>Continue driving without slowing down for intersecting traffic, but always be alert for potential hazards or unexpected vehicles. Be aware of other traffic regulations in effect on the road as you drive.</div>", unsafe_allow_html=True)

 
if st.button("Back to main page"):
    st.session_state.clear()  # Clears all stored session state variables
    st.rerun()  # Fully refreshes the app





