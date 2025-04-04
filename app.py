import streamlit as st
import os
import io
import json
import re
import requests
import sqlite3
import base64
from datetime import datetime
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import openai

# Set page configuration
st.set_page_config(
    page_title="YouTube Thumbnail Analyzer & Generator",
    page_icon="🎬",
    layout="wide"
)

# ---------- Custom CSS for Dark Mode & YouTube Styling ----------
st.markdown("""
<style>
    .main {
        background-color: #0f0f0f;
        color: #f1f1f1;
    }
    .stApp {
        background-color: #0f0f0f;
    }
    h1, h2, h3 {
        color: #f1f1f1;
        font-family: 'Roboto', sans-serif;
    }
    p, li, div {
        color: #aaaaaa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #272727;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
        color: #f1f1f1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff0000;
        color: white;
    }
    .stButton>button {
        background-color: #ff0000;
        color: white;
        border: none;
        border-radius: 2px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 20px;
        background-color: #121212;
        color: #f1f1f1;
        border: 1px solid #303030;
    }
    .thumbnail-container, .generated-image-container, .db-thumbnail-container {
        border: 1px solid #303030;
        border-radius: 8px;
        padding: 10px;
        background-color: #181818;
        margin-bottom: 10px;
    }
    pre {
        background-color: #121212 !important;
    }
    code {
        color: #a9dc76 !important;
    }
    .label-button {
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- SQLite Database Functions ----------

DB_NAME = "thumbnails.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS thumbnails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB,
            analysis TEXT,
            label TEXT,
            prompt_template TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def store_thumbnail_record(image_bytes, analysis, label, prompt_template):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO thumbnails (image, analysis, label, prompt_template)
        VALUES (?, ?, ?, ?)
    """, (image_bytes, analysis, label, prompt_template))
    conn.commit()
    conn.close()

def get_labels():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT DISTINCT label FROM thumbnails")
    labels = [row[0] for row in c.fetchall()]
    conn.close()
    return labels

def get_records_by_label(label):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, image, analysis, prompt_template, created_at FROM thumbnails WHERE label=?", (label,))
    records = c.fetchall()
    conn.close()
    return records

# ---------- API Credential Setup Functions ----------

def setup_credentials():
    vision_client = None
    openai_client = None

    # Google Vision API Credentials
    try:
        if 'GOOGLE_CREDENTIALS' in st.secrets:
            credentials_dict = st.secrets["GOOGLE_CREDENTIALS"]
            if isinstance(credentials_dict, str):
                credentials_dict = json.loads(credentials_dict)
            credentials = service_account.Credentials.from_service_account_info(credentials_dict)
            vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            if os.path.exists("service-account.json"):
                credentials = service_account.Credentials.from_service_account_file("service-account.json")
                vision_client = vision.ImageAnnotatorClient(credentials=credentials)
            else:
                st.info("Google Vision API credentials not found. Analysis will use only OpenAI.")
    except Exception as e:
        st.info(f"Google Vision API not available: {e}")

    # OpenAI API Credentials
    try:
        api_key = None
        if 'OPENAI_API_KEY' in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        else:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                api_key = st.text_input("Enter your OpenAI API key:", type="password")
                if not api_key:
                    st.warning("Please enter an OpenAI API key to continue.")
        if api_key:
            openai.api_key = api_key
            openai_client = openai
    except Exception as e:
        st.error(f"Error setting up OpenAI API: {e}")
        st.info("Troubleshooting: Try updating the OpenAI library with 'pip install --upgrade openai'")

    return vision_client, openai_client

# ---------- Image Analysis Functions ----------

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def analyze_with_vision(image_bytes, vision_client):
    try:
        image = vision.Image(content=image_bytes)
        label_detection = vision_client.label_detection(image=image)
        text_detection = vision_client.text_detection(image=image)
        face_detection = vision_client.face_detection(image=image)
        logo_detection = vision_client.logo_detection(image=image)
        image_properties = vision_client.image_properties(image=image)
        
        results = {
            "labels": [{"description": label.description, "score": float(label.score)}
                       for label in label_detection.label_annotations],
            "text": [{"description": text.description} for text in text_detection.text_annotations[:1]],
            "faces": [{"joy": face.joy_likelihood.name,
                       "sorrow": face.sorrow_likelihood.name,
                       "anger": face.anger_likelihood.name,
                       "surprise": face.surprise_likelihood.name}
                      for face in face_detection.face_annotations],
            "logos": [{"description": logo.description} for logo in logo_detection.logo_annotations],
            "colors": [{"color": {"red": color.color.red,
                                  "green": color.color.green,
                                  "blue": color.color.blue},
                        "score": float(color.score),
                        "pixel_fraction": float(color.pixel_fraction)}
                       for color in image_properties.image_properties_annotation.dominant_colors.colors[:5]]
        }
        return results
    except Exception as e:
        st.error(f"Error analyzing image with Google Vision API: {e}")
        return None

def analyze_with_openai(openai_client, base64_image):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this YouTube thumbnail. Describe what you see in detail."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        try:
            response = openai_client.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Analyze this YouTube thumbnail. Describe what you see in detail."}],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception:
            # Fallback simplified text prompt if image analysis is not available
            response = openai_client.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt="Analyze a YouTube thumbnail. Describe what you see in detail.",
                max_tokens=500
            )
            return response['choices'][0]['text']
    except Exception as e:
        st.error(f"Error analyzing image with OpenAI: {e}")
        return "Unable to analyze the image."

# ---------- Thumbnail Classification & Generic Prompt Generation ----------

def classify_thumbnail(openai_client, vision_results, openai_desc):
    """
    Combine the vision results and OpenAI description, then ask OpenAI
    to label the thumbnail into one of the popular categories.
    """
    try:
        analysis_data = {
            "vision_analysis": vision_results if vision_results else "No Vision Data",
            "openai_description": openai_desc
        }
        prompt = f"""
        Based on the following thumbnail analysis data:
        {json.dumps(analysis_data, indent=2)}
        
        Classify the thumbnail into one of these popular categories:
        Face Close-up, Text-heavy, Split-screen, Reaction shot, Chart/Graph, Before-After, Cinematic Frame.
        Provide only the category name.
        """
        response = openai_client.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        label = response.choices[0].message.content.strip()
        return label
    except Exception as e:
        st.error(f"Error classifying thumbnail: {e}")
        return "Uncategorized"

def generate_prompt_template(label):
    """
    Generate a generic prompt template (at the label level) for creating similar thumbnails.
    You can modify these templates as needed.
    """
    templates = {
        "Face Close-up": (
            "Generate a YouTube thumbnail that emphasizes a close-up of a person's face. "
            "Focus on clear facial expressions, vibrant colors, and an engaging background suitable for high-impact content."
        ),
        "Text-heavy": (
            "Generate a YouTube thumbnail dominated by bold and clear text elements. "
            "Incorporate contrasting colors and a dynamic layout to ensure the text stands out."
        ),
        "Split-screen": (
            "Generate a YouTube thumbnail using a split-screen layout that showcases two contrasting scenes or ideas. "
            "Balance the visuals to create an eye-catching and informative design."
        ),
        "Reaction shot": (
            "Generate a YouTube thumbnail featuring a reaction shot with expressive emotions. "
            "Highlight facial expressions and use dramatic lighting to capture the moment."
        ),
        "Chart/Graph": (
            "Generate a YouTube thumbnail that incorporates a chart or graph element. "
            "Design the thumbnail to be clean, modern, and informative, perfect for educational or analytical content."
        ),
        "Before-After": (
            "Generate a YouTube thumbnail that visually contrasts a before-and-after scenario. "
            "Utilize side-by-side comparisons and clear visual cues to emphasize transformation."
        ),
        "Cinematic Frame": (
            "Generate a YouTube thumbnail with a cinematic look, using dramatic lighting, composition, and color grading "
            "to evoke emotion and tell a compelling story."
        )
    }
    default_prompt = (
        "Generate a YouTube thumbnail with a 16:9 aspect ratio that is engaging, vibrant, and aligned with current design trends."
    )
    return templates.get(label, default_prompt)

# ---------- Library Explorer UI Functions ----------

def library_explorer():
    st.header("Thumbnail Library Explorer")
    labels = get_labels()
    if not labels:
        st.info("No thumbnails have been processed yet.")
        return

    # Check if a label has been selected in session state.
    if "selected_label" not in st.session_state:
        st.session_state.selected_label = None

    # If no label is selected, show label buttons.
    if st.session_state.selected_label is None:
        st.markdown("### Select a Category to Explore")
        cols = st.columns(4)
        for idx, label in enumerate(labels):
            with cols[idx % 4]:
                if st.button(label, key=f"btn_{label}"):
                    st.session_state.selected_label = label
    else:
        st.markdown(f"### Thumbnails for category: **{st.session_state.selected_label}**")
        records = get_records_by_label(st.session_state.selected_label)
        if records:
            for rec in records:
                rec_id, image_blob, analysis, prompt_template, created_at = rec
                image = Image.open(io.BytesIO(image_blob))
                with st.expander(f"Thumbnail ID: {rec_id} (Uploaded on: {created_at})"):
                    st.image(image, caption=f"Category: {st.session_state.selected_label}", use_column_width=True)
                    st.markdown("**Analysis Data:**")
                    st.code(analysis, language="json", key=f"analysis_{rec_id}")
                    st.markdown("**Generic Prompt Template:**")
                    st.text_area("", value=prompt_template, height=80, key=f"prompt_{rec_id}")
        else:
            st.info("No thumbnails found for this category.")
        if st.button("Back to Categories", key="back_button"):
            st.session_state.selected_label = None

# ---------- Upload and Process Functions ----------

def upload_and_process(vision_client, openai_client):
    st.header("Upload and Analyze Thumbnails")
    st.info("Upload up to 10 thumbnail images at once.")

    uploaded_files = st.file_uploader("Choose thumbnail images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("Please upload a maximum of 10 images at once.")
            return
        
        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file)
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                image_bytes = img_byte_arr.getvalue()

                st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    base64_image = encode_image(image_bytes)
                    openai_desc = analyze_with_openai(openai_client, base64_image)
                    vision_results = None
                    if vision_client:
                        vision_results = analyze_with_vision(image_bytes, vision_client)
                    
                    # Classify the thumbnail using the combined analysis data
                    label = classify_thumbnail(openai_client, vision_results, openai_desc)
                    prompt_template = generate_prompt_template(label)
                    
                    st.markdown(f"**Category:** {label}")
                    st.markdown("**Generic Prompt Template:**")
                    st.text_area("", value=prompt_template, height=80, key=f"upload_prompt_{uploaded_file.name}")
                    
                    analysis_details = {
                        "openai_description": openai_desc,
                        "vision_analysis": vision_results if vision_results else "No Vision Data"
                    }
                    store_thumbnail_record(image_bytes, json.dumps(analysis_details, indent=2), label, prompt_template)
                    st.success(f"Processed and stored {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

# ---------- Main App ----------

def main():
    init_db()
    
    st.markdown(
        '<div style="display: flex; align-items: center; padding: 10px 0;">'
        '<span style="color: #FF0000; font-size: 28px; font-weight: bold; margin-right: 5px;">▶️</span>'
        '<h1 style="margin: 0; color: #f1f1f1;">YouTube Thumbnail Analyzer & Generator</h1></div>',
        unsafe_allow_html=True
    )
    st.markdown('<p style="color: #aaaaaa; margin-top: 0;">Analyze, label, and explore your thumbnail library</p>', unsafe_allow_html=True)
    
    vision_client, openai_client = setup_credentials()
    if not openai_client:
        st.error("OpenAI client not initialized. Please check your API key.")
        return

    menu = st.sidebar.radio("Navigation", ["Upload Thumbnails", "Library Explorer"])
    
    if menu == "Upload Thumbnails":
        upload_and_process(vision_client, openai_client)
    elif menu == "Library Explorer":
        library_explorer()

if __name__ == "__main__":
    main()
