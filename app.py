import streamlit as st
import os
import io
import sqlite3
import base64
import zipfile # Added for zip functionality
from datetime import datetime
from PIL import Image
from openai import OpenAI

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
    .stButton>button { background-color: #4CAF50; color: white; border: none; border-radius: 4px; padding: 8px 16px; font-weight: 500; margin-top: 5px; margin-right: 5px;} /* Green add button */
    .stDownloadButton>button { background-color: #007bff; color: white; border: none; border-radius: 4px; padding: 8px 16px; font-weight: 500; margin-top: 5px;} /* Blue download button */
    .stButton>button:hover { opacity: 0.8; }
    .stDownloadButton>button:hover { opacity: 0.8; }
    .thumbnail-container, .db-thumbnail-container { border: 1px solid #303030; border-radius: 8px; padding: 15px; background-color: #181818; margin-bottom: 15px; }
    .analysis-box { border: 1px dashed #444; padding: 10px; margin-top: 10px; border-radius: 4px; background-color: #202020;}
    .stExpander > div:first-child > button { color: #f1f1f1 !important; }
</style>
""", unsafe_allow_html=True)

# ---------- SQLite Database Functions ----------
DB_NAME = "thumbnails.db"

def init_db():
    # (Database functions remain the same)
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

# Modified to be called explicitly by button click
def store_thumbnail_record(image_bytes, label, reason):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO thumbnails (image, label, reason)
            VALUES (?, ?, ?)
        """, (sqlite3.Binary(image_bytes), label, reason))
        conn.commit()
        return True # Indicate success
    except sqlite3.Error as e:
        st.error(f"Database Error: Failed to store record. {e}")
        return False # Indicate failure
    finally:
        conn.close()


def get_labels():
    # (Database functions remain the same)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT DISTINCT label FROM thumbnails WHERE label IS NOT NULL AND label != 'Uncategorized'")
    labels = sorted([row[0] for row in c.fetchall() if row[0]])
    conn.close()
    return labels

def get_records_by_label(label):
    # (Database functions remain the same)
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # Return rows as dict-like objects
    c = conn.cursor()
    c.execute("SELECT id, image, label, reason, created_at FROM thumbnails WHERE label=?", (label,))
    records = c.fetchall()
    conn.close()
    return records

# ---------- OpenAI API Setup ----------
# (setup_openai_client function remains the same)
def setup_openai_client():
    api_key = None
    if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password", key="api_key_input")
        if not api_key:
            # Removed warning here, will be handled in main() if client is None
            return None

    try:
        client = OpenAI(api_key=api_key)
        # Optionally add a check here to see if the key is valid, e.g., list models
        # client.models.list()
        return client
    except Exception as e:
        st.sidebar.error(f"Error initializing OpenAI client: {e}")
        return None

# ---------- Utility Function ----------
# (encode_image function remains the same)
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# ---------- OpenAI Analysis & Classification Function ----------
# (analyze_and_classify_thumbnail function remains mostly the same)
def analyze_and_classify_thumbnail(client: OpenAI, image_bytes: bytes):
    if not client:
        return "Uncategorized", "OpenAI client not initialized."

    base64_image = encode_image(image_bytes)
    image_data_uri = f"data:image/jpeg;base64,{base64_image}" # Assumes JPEG

    category_definitions_text = """
    1.  **Text-Dominant:** Large, bold typography is the main focus...
    2.  **Minimalist / Clean:** Uncluttered, simple background...
    3.  **Face-Focused:** A close-up, expressive human face is the largest or most central element...
    4.  **Before & After:** Clearly divided layout showing two distinct states...
    5.  **Comparison / Versus:** Layout structured (often split-screen) comparing items/ideas...
    6.  **Collage / Multi-Image:** Composed of multiple distinct images arranged together...
    7.  **Image-Focused:** A single, high-quality photo/illustration is dominant...
    8.  **Branded:** Most defining trait is prominent, consistent channel branding...
    9.  **Curiosity Gap / Intrigue:** Deliberately obscures info...
    10. **Other / Unclear:** Doesn't fit well or mixes styles heavily...
    """
    valid_categories = {
        "Text-Dominant", "Minimalist / Clean", "Face-Focused",
        "Before & After", "Comparison / Versus", "Collage / Multi-Image",
        "Image-Focused", "Branded", "Curiosity Gap / Intrigue", "Other / Unclear"
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                 {
                    "role": "system",
                    "content": "You are a professional YouTube thumbnail analyst..." # Same system prompt
                 },
                 {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Classify this thumbnail using these definitions:\n{category_definitions_text}\n\nProvide your answer strictly in the format:\nLabel: <Category Name>\nReason: <Your brief reason>"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri, "detail": "low"}
                        }
                    ]
                 }
            ],
            max_tokens=100
        )
        result = response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error during OpenAI analysis: {e}")
        result = "Label: Uncategorized\nReason: Analysis failed due to an API error."

    # (Parsing logic remains the same)
    label = "Uncategorized"
    reason = "Analysis could not determine a category or reason."
    try:
        lines = result.splitlines()
        if len(lines) >= 1 and lines[0].startswith("Label:"):
            label_candidate = lines[0].replace("Label:", "").strip()
            # Validate against known categories
            found = False
            for cat in valid_categories:
                 if cat.lower() == label_candidate.lower(): # Exact match preferred now
                     label = cat
                     found = True
                     break
            if not found: # Simple fuzzy check if exact match fails
                 for cat in valid_categories:
                      if cat.lower() in label_candidate.lower():
                          label = cat
                          found = True
                          break
            if not found:
                 label = "Other / Unclear"

        if len(lines) >= 2 and lines[1].startswith("Reason:"):
            reason = lines[1].replace("Reason:", "").strip()
        elif len(lines) == 1 and label not in ["Uncategorized", "Other / Unclear"]:
             reason = "No specific reason provided by AI."

    except Exception as parse_error:
        st.warning(f"Could not parse AI response: '{result}'. Error: {parse_error}")
        label = "Uncategorized"
        reason = "Failed to parse the analysis result format."

    return label, reason

# ---------- Callback function for Add to Library button ----------
def add_to_library_callback(file_id, image_bytes, label, reason):
    """Stores the record and updates session state."""
    if store_thumbnail_record(image_bytes, label, reason):
        st.session_state[f'added_{file_id}'] = True # Mark as added
        st.toast(f"Thumbnail added to '{label}' category!", icon="✅") # Use toast for less intrusive feedback
    else:
        # Error message is shown by store_thumbnail_record
        pass

# ---------- Upload and Process Function (Modified) ----------
def upload_and_process(client: OpenAI):
    st.header("Upload and Analyze Thumbnails")
    st.info("Upload up to 10 thumbnail images (JPG, JPEG, PNG). Analysis results appear below each image.")
    uploaded_files = st.file_uploader(
        "Choose thumbnail images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("Please upload a maximum of 10 images at once.")
            return

        cols = st.columns(3)
        col_index = 0

        for uploaded_file in uploaded_files:
            # Use file_id as a stable key if available, otherwise use name (less reliable if names repeat)
            file_id = uploaded_file.id if hasattr(uploaded_file, 'id') else uploaded_file.name

            with cols[col_index % 3]:
                st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
                try:
                    image_bytes = uploaded_file.getvalue() # Get original bytes first
                    display_image = Image.open(io.BytesIO(image_bytes))

                    # Display original uploaded image first
                    st.image(display_image, caption=f"{uploaded_file.name}", use_container_width=True)

                    # Prepare bytes for analysis/storage (convert to JPEG)
                    img_byte_arr = io.BytesIO()
                    if display_image.mode in ('RGBA', 'LA') or (display_image.mode == 'P' and 'transparency' in display_image.info):
                         display_image = display_image.convert('RGB')
                    display_image.save(img_byte_arr, format='JPEG', quality=85)
                    processed_image_bytes = img_byte_arr.getvalue()

                    # --- Analysis Section ---
                    label = "Analysis Pending"
                    reason = "Analysis Pending"
                    analysis_placeholder = st.empty() # Placeholder for analysis results

                    with st.spinner(f"Analyzing {uploaded_file.name}..."):
                        label, reason = analyze_and_classify_thumbnail(client, processed_image_bytes)

                    # Display analysis results in a distinct box
                    with analysis_placeholder.container():
                         st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                         st.markdown(f"**Suggested Category:** `{label}`")
                         st.markdown(f"**Reason:** _{reason}_")

                         # --- Add to Library Button ---
                         # Check session state to see if it's already added
                         is_added = st.session_state.get(f'added_{file_id}', False)

                         st.button(
                             "✅ Add to Library" if not is_added else "✔️ Added",
                             key=f'btn_add_{file_id}',
                             on_click=add_to_library_callback,
                             args=(file_id, processed_image_bytes, label, reason), # Pass necessary data
                             disabled=is_added or label == "Uncategorized" # Disable if added or uncategorized
                         )
                         st.markdown('</div>', unsafe_allow_html=True)


                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                finally:
                    st.markdown('</div>', unsafe_allow_html=True) # Close container

            col_index += 1


# ---------- Function to create Zip File in Memory ----------
def create_zip_in_memory(records):
    """Creates a zip file containing images from the records."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for record in records:
            rec_id = record['id']
            image_blob = record['image']
            label = record['label'].replace('/', '_').replace(' ', '_') # Sanitize label for filename
            timestamp_str = record['created_at'].split('.')[0].replace(':', '-').replace(' ','_') # Sanitize timestamp

            # Create a meaningful filename
            filename_in_zip = f"{label}_{rec_id}_{timestamp_str}.jpg" # Assuming JPEG was stored
            try:
                zip_file.writestr(filename_in_zip, image_blob)
            except Exception as zip_err:
                st.warning(f"Could not add image ID {rec_id} to zip: {zip_err}")

    zip_buffer.seek(0)
    return zip_buffer


# ---------- Library Explorer (Modified) ----------
def library_explorer():
    st.header("Thumbnail Library Explorer")
    st.markdown("Browse thumbnails categorized by the AI, and download categories as Zip files.")
    labels = get_labels()
    if not labels:
        st.info("The library is empty. Upload and analyze some thumbnails first!")
        return

    if "selected_label" not in st.session_state:
        st.session_state.selected_label = None

    if st.session_state.selected_label is None:
        st.markdown("### Select a Category")
        cols_per_row = 4
        cols = st.columns(cols_per_row)
        for idx, label in enumerate(labels):
            with cols[idx % cols_per_row]:
                if st.button(label, key=f"btn_{label.replace(' ', '_')}", use_container_width=True):
                    st.session_state.selected_label = label
                    st.rerun()
    else:
        selected_category = st.session_state.selected_label
        st.markdown(f"### Category: **{selected_category}**")

        col1, col2 = st.columns([3, 1]) # Columns for back button and download button
        with col1:
            if st.button("⬅️ Back to Categories", key="back_button"):
                st.session_state.selected_label = None
                st.rerun()

        records = get_records_by_label(selected_category)

        if records:
            # --- Add Download Button ---
            with col2:
                zip_buffer = create_zip_in_memory(records)
                st.download_button(
                    label="⬇️ Download Zip",
                    data=zip_buffer,
                    file_name=f"{selected_category.replace('/', '_').replace(' ', '_')}_thumbnails.zip",
                    mime="application/zip",
                    key=f"download_{selected_category.replace(' ', '_')}"
                )

            # Display thumbnails
            num_records = len(records)
            cols_per_row = 3
            record_cols = st.columns(cols_per_row)
            col_idx = 0

            for record in records: # Use the dict-like record from row_factory
                with record_cols[col_idx % cols_per_row]:
                    rec_id = record['id']
                    image_blob = record['image']
                    label = record['label']
                    reason = record['reason']
                    created_at_str = record['created_at']
                    try:
                       created_at = datetime.strptime(created_at_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
                       display_time = created_at.strftime('%Y-%m-%d %H:%M')
                    except (ValueError, TypeError):
                        display_time = created_at_str

                    st.markdown('<div class="db-thumbnail-container">', unsafe_allow_html=True)
                    try:
                        image = Image.open(io.BytesIO(image_blob))
                        st.image(image, caption=f"ID: {rec_id} | {display_time}", use_container_width=True)
                        st.markdown(f"**Reason:** {reason}", help="Reason provided by AI during analysis.")
                    except Exception as img_err:
                         st.warning(f"Could not load image for ID {rec_id}: {img_err}")
                    st.markdown('</div>', unsafe_allow_html=True)
                col_idx += 1
        else:
            st.info(f"No thumbnails found for the category: '{selected_category}'.")
            # Still show back button even if no records
            # if st.button("⬅️ Back to Categories", key="back_button_no_rec"):
            #     st.session_state.selected_label = None
            #     st.rerun()


# ---------- Main App ----------
def main():
    init_db()

    st.sidebar.markdown(
        '<div style="display: flex; align-items: center; padding: 10px 0;">'
        '<span style="color: #FF0000; font-size: 28px; font-weight: bold; margin-right: 5px;">▶️</span>'
        '<h1 style="margin: 0; color: #f1f1f1; font-size: 24px;">Thumbnail Analyzer</h1></div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown('<p style="color: #aaaaaa; margin-top: 0;">Analyze & Explore Thumbnails</p>', unsafe_allow_html=True)
    st.sidebar.markdown("---")

    client = setup_openai_client() # Initialize client

    menu = st.sidebar.radio(
        "Navigation",
        ["Upload & Analyze", "Library Explorer"],
        key="nav_menu"
    )
    st.sidebar.markdown("---") # Separator
    st.sidebar.info("Thumbnails added to the library are stored locally in a 'thumbnails.db' file.")


    # Main content area logic
    if menu == "Upload & Analyze":
        if not client:
            st.error("❌ OpenAI client not initialized. Please provide a valid API key in the sidebar.")
        else:
            upload_and_process(client) # Pass client object
    elif menu == "Library Explorer":
        library_explorer()

if __name__ == "__main__":
    # Initialize session state keys if they don't exist
    if 'selected_label' not in st.session_state:
        st.session_state.selected_label = None
    # Add other session state initializations if needed, e.g. for the 'added_' flags,
    # but they are dynamically created, so explicit init might not be strictly necessary unless checking them before creation.
    main()
