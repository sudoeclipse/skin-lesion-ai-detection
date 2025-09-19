import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]

# --- Page Configuration ---
st.set_page_config(
    page_title="Lesion Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Hardcoded Dark Mode CSS ---
# Base CSS for layout and elements
BASE_CSS = """
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .logo { font-size: 1.5rem; font-weight: bold; padding-top: 0.5rem; }
    .stButton>button { width: 100%; border-radius: 0.5rem; padding: 0.75rem 1rem; transition: background-color 0.3s; border: none; }
    .stFileUploader { border: 2px dashed; border-radius: 0.5rem; padding: 1rem; }
    .center-text { text-align: center; }
    .footer { padding: 2rem 5rem; margin-top: 3rem; }
"""

# CSS for the dark (navy blue) theme
DARK_THEME_CSS = """
    body { color: #FFFFFF; background-color: #001f3f; }
    .logo { color: #FFFFFF; }
    .stButton>button { color: #ffffff; background-color: #007BFF; }
    .stButton>button:hover { background-color: #0056b3; }
    .stFileUploader { border-color: #007BFF; background-color: rgba(0, 123, 255, 0.05); }
    .footer { border-top: 1px solid #333; }
    .footer h4 { color: #FFFFFF; }
    .footer, .footer a { color: #CCCCCC; }
    .footer a:hover { color: #FFFFFF; }
"""

# Inject the combined CSS
st.markdown(f"<style>{BASE_CSS}{DARK_THEME_CSS}</style>", unsafe_allow_html=True)


# --- Configuration & Model Loading ---
MODEL_PATH = 'skin_lesion_classifier.h5'
IMG_HEIGHT = 75
IMG_WIDTH = 100

LESION_CLASSES_INFO = {
    0: ('Actinic Keratoses (akiec)', 'Often benign, but can be a precursor to skin cancer. A dermatologist visit is recommended.'),
    1: ('Basal Cell Carcinoma (bcc)', 'A common type of skin cancer. Usually not life-threatening but requires medical treatment.'),
    2: ('Benign Keratosis-like Lesions (bkl)', 'Non-cancerous skin growths, like "age spots" or seborrheic keratoses.'),
    3: ('Dermatofibroma (df)', 'A common benign skin nodule. Typically harmless.'),
    4: ('Melanoma (mel)', 'The most serious type of skin cancer. Early detection is crucial. See a doctor immediately.'),
    5: ('Melanocytic Nevi (nv)', 'Common moles. Mostly benign, but changes should be monitored.'),
    6: ('Vascular Lesions (vasc)', 'Benign lesions like cherry angiomas or spider veins. Usually not a cause for concern.')
}

@st.cache_resource
def load_sk_model(path):
    """Loads the trained Keras model."""
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_sk_model(MODEL_PATH)

# --- Prediction Logic ---
def predict(image: Image.Image):
    """ Preprocesses the image and returns the prediction. """
    img_array = np.array(image)
    img_resized = tf.image.resize(img_array, [IMG_HEIGHT, IMG_WIDTH])
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    prediction_array = model.predict(img_expanded)
    confidence_score = np.max(prediction_array)
    class_index = np.argmax(prediction_array)
    class_name, description = LESION_CLASSES_INFO[class_index]
    return class_name, description, confidence_score

# --- UI Layout ---

# REMOVED: Header no longer needs columns or the toggle
st.markdown('<div class="logo">Lesion Detection</div>', unsafe_allow_html=True)

# Hero Section
st.markdown('<h1 class="center-text">Skin Lesion Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="center-text">Upload an image of a skin lesion to get AI-based classification with expert-backed information.</p>', unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if st.button("Predict", type="primary"):
        if uploaded_file is not None and model is not None:
            with st.spinner('Analyzing...'):
                image = Image.open(uploaded_file)
                class_name, description, score = predict(image)
                st.session_state.prediction_result = (class_name, description, score)
                st.session_state.uploaded_image = image
        elif model is None:
            st.error(f"Model not found at {MODEL_PATH}. Please ensure the model file is in the correct directory.")
        else:
            st.warning("Please upload an image first.")

with col2:
    st.subheader("Analysis Results")
    if 'prediction_result' in st.session_state:
        class_name, description, score = st.session_state.prediction_result
        
        st.success(f"**Predicted Class:** {class_name}")
        st.metric(label="Confidence Score", value=f"{score:.2%}")
        st.info(f"**Information:** {description}")

        with st.expander("Show Uploaded Image"):
            st.image(st.session_state.uploaded_image, use_container_width=True)
        
        st.warning("Disclaimer: This is a student project and not for medical diagnosis. Consult a professional for any health concerns.")
    else:
        st.info("Upload an image and click 'Predict' to see results.")

# Footer
st.markdown('<div style="margin-top: 5rem;"></div>', unsafe_allow_html=True)
st.divider()
footer_cols = st.columns(2)
with footer_cols[0]:
    st.markdown("<h4>Lesion Detection</h4>", unsafe_allow_html=True)
    st.write("AI-powered skin lesion classification system for early detection and improved healthcare outcomes.")
with footer_cols[1]:
    st.markdown("<h4>Our Team</h4>", unsafe_allow_html=True)
    st.write("Samruddhi Amol Shah")
    st.write("*Second Year at VIT Chennai*")
