import streamlit as st
import os
import io
import sqlite3
import base64
from datetime import datetime
from PIL import Image
from openai import OpenAI # Import the main OpenAI class

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
    .stButton>button:hover { background-color: #cc0000; }
    .thumbnail-container, .db-thumbnail-container { border: 1px solid #303030; border-radius: 8px; padding: 10px; background-color: #181818; margin-bottom: 10px; }
    .stExpander > div:first-child > button { color: #f1f1f1 !important; } /* Expander title color */
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

def store_thumbnail_record(image_bytes, label, reason):
    # (Database functions remain the same)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO thumbnails (image, label, reason)
        VALUES (?, ?, ?)
    """, (sqlite3.Binary(image_bytes), label, reason)) # Use sqlite3.Binary for BLOB
    conn.commit()
    conn.close()

def get_labels():
    # (Database functions remain the same)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT DISTINCT label FROM thumbnails WHERE label IS NOT NULL AND label != 'Uncategorized'")
    labels = sorted([row[0] for row in c.fetchall() if row[0]]) # Ensure not None and sort
    conn.close()
    return labels

def get_records_by_label(label):
    # (Database functions remain the same)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, image, label, reason, created_at FROM thumbnails WHERE label=?", (label,))
    records = c.fetchall()
    conn.close()
    return records

# ---------- OpenAI API Setup ----------
def setup_openai_client():
    """Initializes and returns the OpenAI client."""
    api_key = None
    # Prioritize Streamlit secrets if available
    if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        # Fallback to environment variable
        api_key = os.environ.get('OPENAI_API_KEY')

    # If still no key, prompt the user (less secure for deployed apps)
    if not api_key:
        api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password", key="api_key_input")
        if not api_key:
            st.sidebar.warning("Please enter an OpenAI API key in the sidebar to enable analysis.")
            return None # Return None if no key is provided

    try:
        # Initialize the client with the API key
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None

# ---------- Utility Function ----------
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# ---------- OpenAI Analysis & Classification Function ----------
def analyze_and_classify_thumbnail(client: OpenAI, image_bytes: bytes):
    """
    Analyzes thumbnail using OpenAI multimodal model (v1.x syntax).
    Requires a model like 'gpt-4o' or 'gpt-4-vision-preview'.
    """
    if not client:
        return "Uncategorized", "OpenAI client not initialized."

    base64_image = encode_image(image_bytes)
    image_data_uri = f"data:image/jpeg;base64,{base64_image}" # Assumes JPEG, adjust if needed

    # Define categories clearly for the prompt
    category_definitions_text = """
    1.  **Text-Dominant:** Large, bold typography is the main focus, covering a significant portion. Minimal imagery.
    2.  **Minimalist / Clean:** Uncluttered, simple background, limited colors, clean font, few elements, lots of negative space.
    3.  **Face-Focused:** A close-up, expressive human face is the largest or most central element.
    4.  **Before & After:** Clearly divided layout showing two distinct states side-by-side.
    5.  **Comparison / Versus:** Layout structured (often split-screen) comparing items/ideas.
    6.  **Collage / Multi-Image:** Composed of multiple distinct images arranged together.
    7.  **Image-Focused:** A single, high-quality photo/illustration is dominant; text is secondary.
    8.  **Branded:** Most defining trait is prominent, consistent channel logos, colors, or fonts.
    9.  **Curiosity Gap / Intrigue:** Deliberately obscures info (blurring, question marks, arrows).
    10. **Other / Unclear:** Doesn't fit well into other categories or mixes styles heavily.
    """

    valid_categories = { # Keep this set updated with the prompt list
        "Text-Dominant", "Minimalist / Clean", "Face-Focused",
        "Before & After", "Comparison / Versus", "Collage / Multi-Image",
        "Image-Focused", "Branded", "Curiosity Gap / Intrigue", "Other / Unclear"
    }

    try:
        # Use the client.chat.completions.create method (v1.x syntax)
        # Ensure you use a model capable of image input like "gpt-4o"
        response = client.chat.completions.create(
            model="gpt-4o", # Use a multimodal model
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional YouTube thumbnail analyst. Analyze the provided thumbnail image and classify it into ONE category based on its most dominant visual style/layout features. Use the provided definitions. Respond ONLY in the format 'Label: <Category Name>\nReason: <Brief one or two sentence reason>'."
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
                            "image_url": {
                               "url": image_data_uri,
                               "detail": "low" # Use low detail for faster processing/lower cost if sufficient
                            }
                        }
                    ]
                }
            ],
            max_tokens=100 # Reduced max_tokens as only label and short reason needed
        )
        result = response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error during OpenAI analysis: {e}")
        result = "Label: Uncategorized\nReason: Analysis failed due to an API error."

    # Parse the output
    label = "Uncategorized"
    reason = "Analysis could not determine a category or reason."
    try:
        lines = result.splitlines()
        if len(lines) >= 1 and lines[0].startswith("Label:"):
            label_candidate = lines[0].replace("Label:", "").strip()
            # Validate against known categories
            if label_candidate in valid_categories:
                 label = label_candidate
            else: # Handle cases where the model might invent a category slightly differently
                 # Simple fuzzy matching (optional improvement)
                 found = False
                 for cat in valid_categories:
                     if cat.lower() in label_candidate.lower():
                         label = cat
                         found = True
                         break
                 if not found:
                     label = "Other / Unclear" # Default if mismatch

        if len(lines) >= 2 and lines[1].startswith("Reason:"):
            reason = lines[1].replace("Reason:", "").strip()
        elif len(lines) == 1 and label != "Uncategorized": # Sometimes reason might be omitted by LLM
             reason = "No specific reason provided by AI."

    except Exception as parse_error:
        st.warning(f"Could not parse AI response: '{result}'. Error: {parse_error}")
        label = "Uncategorized"
        reason = "Failed to parse the analysis result format."


    return label, reason


# ---------- Upload and Process Function ----------
def upload_and_process(client: OpenAI): # Pass the initialized client
    st.header("Upload and Analyze Thumbnails")
    st.info("Upload up to 10 thumbnail images (JPG, JPEG, PNG).")
    uploaded_files = st.file_uploader(
        "Choose thumbnail images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="file_uploader" # Add key for potential state management later
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("Please upload a maximum of 10 images at once.")
            return # Stop processing if too many files

        cols = st.columns(3) # Create columns for layout
        col_index = 0

        for uploaded_file in uploaded_files:
            with cols[col_index % 3]: # Cycle through columns
                try:
                    image = Image.open(uploaded_file)
                    # Convert image to JPEG bytes for consistency before storing/analyzing
                    img_byte_arr = io.BytesIO()
                    # Handle transparency for PNGs by converting to RGB
                    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                         image = image.convert('RGB')
                    image.save(img_byte_arr, format='JPEG', quality=85) # Save as JPEG
                    image_bytes = img_byte_arr.getvalue()

                    # Display in a container
                    st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
                    # Display the potentially converted image for visual consistency
                    display_image = Image.open(io.BytesIO(image_bytes))
                    st.image(display_image, caption=f"{uploaded_file.name}", use_container_width=True)

                    with st.spinner(f"Analyzing {uploaded_file.name}..."):
                        label, reason = analyze_and_classify_thumbnail(client, image_bytes)

                    st.markdown(f"**Category:** `{label}`")
                    st.markdown(f"**Reason:** {reason}")

                    # Store the record (using JPEG bytes)
                    store_thumbnail_record(image_bytes, label, reason)
                    st.success(f"Processed: {uploaded_file.name}")
                    st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                    st.markdown('</div>', unsafe_allow_html=True) # Close container even on error

            col_index += 1 # Move to the next column

# ---------- Library Explorer ----------
def library_explorer():
    st.header("Thumbnail Library Explorer")
    labels = get_labels()
    if not labels:
        st.info("No categorized thumbnails found in the library yet.")
        return

    # Use session state to remember the selected label
    if "selected_label" not in st.session_state:
        st.session_state.selected_label = None

    # Display category selection buttons if none is selected
    if st.session_state.selected_label is None:
        st.markdown("### Select a Category to View")
        # Dynamically create columns based on number of labels
        num_labels = len(labels)
        cols_per_row = 4
        cols = st.columns(cols_per_row)
        for idx, label in enumerate(labels):
            with cols[idx % cols_per_row]:
                if st.button(label, key=f"btn_{label.replace(' ', '_')}", use_container_width=True):
                    st.session_state.selected_label = label
                    st.rerun() # Rerun to update the view immediately
    else:
        # Display thumbnails for the selected category
        st.markdown(f"### Category: **{st.session_state.selected_label}**")
        if st.button("⬅️ Back to Categories", key="back_button"):
            st.session_state.selected_label = None
            st.rerun() # Rerun to go back

        records = get_records_by_label(st.session_state.selected_label)
        if records:
            # Create columns for displaying records
            num_records = len(records)
            cols_per_row = 3 # Adjust as needed
            record_cols = st.columns(cols_per_row)
            col_idx = 0

            for rec in records:
                with record_cols[col_idx % cols_per_row]:
                    rec_id, image_blob, label, reason, created_at_str = rec
                    # Parse timestamp
                    try:
                       # Adjust format string if needed based on how SQLite stores it
                       created_at = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')
                       display_time = created_at.strftime('%Y-%m-%d %H:%M')
                    except (ValueError, TypeError):
                        display_time = created_at_str # Fallback if parsing fails


                    st.markdown('<div class="db-thumbnail-container">', unsafe_allow_html=True)
                    try:
                        image = Image.open(io.BytesIO(image_blob))
                        st.image(image, caption=f"ID: {rec_id} | {display_time}", use_container_width=True)
                        st.markdown(f"**Reason:** {reason}", help="Reason provided by AI during analysis.") # Use help for tooltip
                    except Exception as img_err:
                         st.warning(f"Could not load image for ID {rec_id}: {img_err}")
                    st.markdown('</div>', unsafe_allow_html=True)
                col_idx += 1
        else:
            st.info(f"No thumbnails found for the category: '{st.session_state.selected_label}'.")


# ---------- Main App ----------
def main():
    init_db() # Initialize the database schema if it doesn't exist

    # Sidebar Navigation and Title
    st.sidebar.markdown(
        '<div style="display: flex; align-items: center; padding: 10px 0;">'
        '<span style="color: #FF0000; font-size: 28px; font-weight: bold; margin-right: 5px;">▶️</span>'
        '<h1 style="margin: 0; color: #f1f1f1; font-size: 24px;">Thumbnail Analyzer</h1></div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown('<p style="color: #aaaaaa; margin-top: 0;">Analyze & Explore Thumbnails</p>', unsafe_allow_html=True)
    st.sidebar.markdown("---") # Separator

    # Setup OpenAI Client - place key input logically in sidebar
    client = setup_openai_client()

    # Main navigation moved to sidebar
    menu = st.sidebar.radio(
        "Navigation",
        ["Upload & Analyze", "Library Explorer"],
        key="nav_menu"
    )
    st.sidebar.markdown("---") # Separator

    # Main content area
    if menu == "Upload & Analyze":
        if not client:
            st.warning("OpenAI client not initialized. Please provide your API key in the sidebar.")
        else:
            upload_and_process(client)
    elif menu == "Library Explorer":
        library_explorer()

if __name__ == "__main__":
    main()
