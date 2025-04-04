import streamlit as st
import os
import io
import json
import sqlite3
import base64
from datetime import datetime
from PIL import Image
import openai

# Set page configuration
st.set_page_config(
    page_title="Thumbnail Analyzer & Generator",
    page_icon="🎬",
    layout="wide"
)

# ---------- Custom CSS for Dark Mode & Styling ----------
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
    .thumbnail-container, .db-thumbnail-container {
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

# ---------- OpenAI API Credential Setup ----------

def setup_openai():
    openai_client = None
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
    return openai_client

# ---------- Utility Function ----------
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# ---------- OpenAI Analysis & Classification Function ----------
def analyze_and_classify_thumbnail(openai_client, image_bytes):
    base64_image = encode_image(image_bytes)
    prompt = f"""
You are a professional YouTube thumbnail analyst. Analyze the following thumbnail image (provided as a base64 string) and provide a detailed description of its visual style, layout, and key elements. Then, based on your analysis, classify the thumbnail into exactly one of the following categories (choose only one):

1. Text-Dominant: Large, bold typography takes up most of the space with minimal imagery.
2. Minimalist / Clean: Simple background, limited color palette, clean font, and one key focal point.
3. Face-Focused: A close-up of a person's face showing strong emotion.
4. Before & After: A split view clearly showing two different states (e.g., transformation).
5. Collage / Multi-Image: Multiple images combined to show variety or list content.
6. Branded: Consistent use of channel colors, fonts, and logos to build brand recognition.
7. Curiosity Gap / Intrigue: Uses elements like blurring or arrows to spark curiosity.

Provide your analysis in a structured format. In your final output, on a new line output exactly: 
Category: <Your Category>

Here is the image:
data:image/jpeg;base64,{base64_image}
    """
    try:
        response = openai_client.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error from OpenAI: {e}")
        result = "Analysis not available.\nCategory: Uncategorized"
    # Try to parse the category from the final line
    lines = result.splitlines()
    category = None
    for line in reversed(lines):
        if line.startswith("Category:"):
            category = line.replace("Category:", "").strip()
            break
    if not category:
        category = "Uncategorized"
    # Return the full analysis text and the parsed category.
    return result, category

# ---------- Generic Prompt Template Generation ----------
def generate_prompt_template(label):
    templates = {
        "Text-Dominant": (
            "Generate a YouTube thumbnail that emphasizes large, bold typography and a compelling text hook. "
            "Keep imagery minimal so that the text stands out and grabs attention."
        ),
        "Minimalist / Clean": (
            "Generate a YouTube thumbnail with a simple background and limited color palette. "
            "Focus on one clear focal point and clean typography for a modern, professional look."
        ),
        "Face-Focused": (
            "Generate a YouTube thumbnail featuring a close-up of a person’s face showing strong emotion. "
            "Ensure the facial expression is engaging and the overall design creates a human connection."
        ),
        "Before & After": (
            "Generate a YouTube thumbnail that clearly shows a transformation through a split-view design. "
            "Highlight the contrast between the before and after states to emphasize change."
        ),
        "Collage / Multi-Image": (
            "Generate a YouTube thumbnail that combines multiple images to showcase variety or a list of items. "
            "Balance the images well to avoid clutter while hinting at diverse content."
        ),
        "Branded": (
            "Generate a YouTube thumbnail that uses consistent channel colors, fonts, and logo placement to build brand recognition. "
            "The design should be cohesive and easily identifiable as part of a brand."
        ),
        "Curiosity Gap / Intrigue": (
            "Generate a YouTube thumbnail that uses visual cues like partial blurring, arrows, or intriguing elements to spark curiosity. "
            "The design should invite viewers to click to find out more."
        )
    }
    default_prompt = (
        "Generate a YouTube thumbnail with a 16:9 aspect ratio that is engaging, vibrant, and aligned with current design trends."
    )
    return templates.get(label, default_prompt)

# ---------- Upload and Process Function ----------
def upload_and_process(openai_client):
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
                    analysis_text, category = analyze_and_classify_thumbnail(openai_client, image_bytes)
                    prompt_template = generate_prompt_template(category)
                    
                    st.markdown(f"**Category:** {category}")
                    st.markdown("**Generic Prompt Template:**")
                    st.text_area("", value=prompt_template, height=80, key=f"upload_prompt_{uploaded_file.name}")
                    
                    # Store the record in the SQLite database
                    store_thumbnail_record(image_bytes, analysis_text, category, prompt_template)
                    st.success(f"Processed and stored {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

# ---------- Library Explorer ----------
def library_explorer():
    st.header("Thumbnail Library Explorer")
    labels = get_labels()
    if not labels:
        st.info("No thumbnails have been processed yet.")
        return

    if "selected_label" not in st.session_state:
        st.session_state.selected_label = None

    # Show label buttons if none is selected
    if st.session_state.selected_label is None:
        st.markdown("### Select a Category to Explore")
        cols = st.columns(4)
        for idx, label in enumerate(labels):
            with cols[idx % 4]:
                if st.button(label, key=f"btn_{label}"):
                    st.session_state.selected_label = label
    else:
        st.markdown(f"### Thumbnails for Category: **{st.session_state.selected_label}**")
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

# ---------- Main App ----------
def main():
    init_db()
    
    st.markdown(
        '<div style="display: flex; align-items: center; padding: 10px 0;">'
        '<span style="color: #FF0000; font-size: 28px; font-weight: bold; margin-right: 5px;">▶️</span>'
        '<h1 style="margin: 0; color: #f1f1f1;">Thumbnail Analyzer & Generator</h1></div>',
        unsafe_allow_html=True
    )
    st.markdown('<p style="color: #aaaaaa; margin-top: 0;">Analyze and label your thumbnail library using OpenAI</p>', unsafe_allow_html=True)
    
    openai_client = setup_openai()
    if not openai_client:
        st.error("OpenAI client not initialized. Please check your API key.")
        return

    menu = st.sidebar.radio("Navigation", ["Upload Thumbnails", "Library Explorer"])
    
    if menu == "Upload Thumbnails":
        upload_and_process(openai_client)
    elif menu == "Library Explorer":
        library_explorer()

if __name__ == "__main__":
    main()
