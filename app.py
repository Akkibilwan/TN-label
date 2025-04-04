import streamlit as st
import os
import io
import sqlite3
import base64
from datetime import datetime
from PIL import Image
import openai

# Set page configuration
st.set_page_config(
    page_title="Thumbnail Analyzer",
    page_icon="🎬",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main { background-color: #0f0f0f; color: #f1f1f1; }
    .stApp { background-color: #0f0f0f; }
    h1, h2, h3 { color: #f1f1f1; font-family: 'Roboto', sans-serif; }
    p, li, div { color: #aaaaaa; }
    .stButton>button { background-color: #ff0000; color: white; border: none; border-radius: 2px; padding: 8px 16px; font-weight: 500; }
    .thumbnail-container, .db-thumbnail-container { border: 1px solid #303030; border-radius: 8px; padding: 10px; background-color: #181818; margin-bottom: 10px; }
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
            label TEXT,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def store_thumbnail_record(image_bytes, label, reason):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO thumbnails (image, label, reason)
        VALUES (?, ?, ?)
    """, (image_bytes, label, reason))
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
    c.execute("SELECT id, image, label, reason, created_at FROM thumbnails WHERE label=?", (label,))
    records = c.fetchall()
    conn.close()
    return records

# ---------- OpenAI API Setup ----------
def setup_openai():
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
            return openai  # Return the module
    except Exception as e:
        st.error(f"Error setting up OpenAI API: {e}")
    return None

# ---------- Utility Function ----------
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# ---------- OpenAI Analysis & Classification Function ----------
def analyze_and_classify_thumbnail(image_bytes):
    base64_image = encode_image(image_bytes)
    prompt = f"""
You are a professional YouTube thumbnail analyst.
Analyze the following thumbnail image (provided as a base64 string) and decide which of the following categories best describes its visual style:
1. Text-Dominant
2. Minimalist / Clean
3. Face-Focused
4. Before & After
5. Collage / Multi-Image
6. Branded
7. Curiosity Gap / Intrigue

Return your answer in exactly this format (do not include any extra text):
Label: <Category>
Reason: <A brief reason in one or two sentences>

Image data:
data:image/jpeg;base64,{base64_image}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error from OpenAI: {e}")
        result = "Label: Uncategorized\nReason: Analysis not available."
    # Parse the output
    label = "Uncategorized"
    reason = "No reason provided."
    for line in result.splitlines():
        if line.startswith("Label:"):
            label = line.replace("Label:", "").strip()
        elif line.startswith("Reason:"):
            reason = line.replace("Reason:", "").strip()
    valid_categories = {
        "Text-Dominant",
        "Minimalist / Clean",
        "Face-Focused",
        "Before & After",
        "Collage / Multi-Image",
        "Branded",
        "Curiosity Gap / Intrigue"
    }
    if label not in valid_categories:
        label = "Uncategorized"
    return label, reason

# ---------- Upload and Process Function ----------
def upload_and_process():
    st.header("Upload and Analyze Thumbnails")
    st.info("Upload up to 10 thumbnail images.")
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
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    label, reason = analyze_and_classify_thumbnail(image_bytes)
                    st.markdown(f"**Category:** {label}")
                    st.markdown(f"**Reason:** {reason}")
                    store_thumbnail_record(image_bytes, label, reason)
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

    if st.session_state.selected_label is None:
        st.markdown("### Select a Category")
        cols = st.columns(4)
        for idx, label in enumerate(labels):
            with cols[idx % 4]:
                if st.button(label, key=f"btn_{label}"):
                    st.session_state.selected_label = label
    else:
        st.markdown(f"### Thumbnails in Category: **{st.session_state.selected_label}**")
        records = get_records_by_label(st.session_state.selected_label)
        if records:
            for rec in records:
                rec_id, image_blob, label, reason, created_at = rec
                image = Image.open(io.BytesIO(image_blob))
                with st.expander(f"Thumbnail ID: {rec_id} (Uploaded on: {created_at})"):
                    st.image(image, caption=f"Category: {label}", use_container_width=True)
                    st.markdown(f"**Reason:** {reason}")
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
        '<h1 style="margin: 0; color: #f1f1f1;">Thumbnail Analyzer</h1></div>',
        unsafe_allow_html=True
    )
    st.markdown('<p style="color: #aaaaaa; margin-top: 0;">Upload thumbnails to have them analyzed and categorized, and explore by category.</p>', unsafe_allow_html=True)
    
    openai_api = setup_openai()
    if not openai_api:
        st.error("OpenAI API not initialized. Please check your API key.")
        return

    menu = st.sidebar.radio("Navigation", ["Upload Thumbnails", "Library Explorer"])
    if menu == "Upload Thumbnails":
        upload_and_process()
    elif menu == "Library Explorer":
        library_explorer()

if __name__ == "__main__":
    main()
