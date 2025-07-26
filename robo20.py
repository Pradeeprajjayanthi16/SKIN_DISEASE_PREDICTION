# streamlit_app.py
import streamlit as st
from roboflow import Roboflow
from PIL import Image
import io
import tempfile
import time
import pandas as pd

# Initialize Roboflow
ROBOFLOW_API_KEY = "IfjlGXuiNMDTNkysEDeV"
GENAI_API_KEY = "AIzaSyBWdaGV6O-nB3PMFA589E75ranScd9WulU"

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("skin-disease-prediction-1ej1a")
model = project.version(4).model

# Load doctor details from CSV
doctor_df = pd.read_csv('Doctor file.csv')

# Streamlit interface
st.title("Skin Disease Prediction with Roboflow Model")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read and display the uploaded file
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        
        with st.spinner("Classifying..."):
            # Save the image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_image_path = temp_file.name
                image.save(temp_image_path)

            # Make predictions
            predictions = model.predict(temp_image_path, confidence=40).json()
            st.write(f"Predictions: {predictions}")

            # Display prediction results
            if 'predictions' in predictions:
                for prediction in predictions['predictions']:
                    class_name = prediction['class']
                    confidence = prediction['confidence']
                    st.write(f"Class: {class_name}, Confidence: {confidence:.2f}")

                    # Recommend doctors based on the predicted disease
                    recommended_doctors = doctor_df[doctor_df['Speciality'].str.contains(class_name, case=False, na=False)]
                    if not recommended_doctors.empty:
                        st.write("### Recommended Doctors:")
                        for _, row in recommended_doctors.iterrows():
                            st.write(f"**Name:** {row['Name']}")
                            st.write(f"**Clinic Address:** {row['Clinic Address']}")
                            st.write(f"**Speciality:** {row['Speciality']}")
                            st.write("---")

            # Visualize predictions
            model.predict(temp_image_path, confidence=40).save("prediction.jpg")
            st.image("prediction.jpg", caption='Prediction Image.', use_column_width=True)

        # After visualization, provide explanation using Generative AI
        st.write("")
        st.write("Generating additional explanation...")
        try:
            import google.generativeai as genai
            genai.configure(api_key=GENAI_API_KEY)
            gen_model = genai.GenerativeModel("gemini-1.5-flash")

            explained_diseases = set()
            if 'predictions' in predictions:
                for prediction in predictions['predictions']:
                    disease_name = prediction['class']
                    if disease_name not in explained_diseases:
                        response = gen_model.generate_content(
                            f"Explain about {disease_name} in a paragraph and suggest a remedy."
                        )
                        explanation = response.text
                        st.write("### Explanation:")
                        explanation_placeholder = st.empty()

                        # Animate the text display
                        displayed_text = ""
                        for char in explanation:
                            displayed_text += char
                            explanation_placeholder.markdown(
                                f"<div style='word-wrap: break-word;'>{displayed_text}</div>",
                                unsafe_allow_html=True,
                            )
                            time.sleep(0.01)
                        explained_diseases.add(disease_name)
        except Exception as gen_error:
            st.error(f"Generative AI Error: {gen_error}")

    except Exception as e:
        st.error(f"Error: {e}")
        st.warning("Unable to classify the image.")