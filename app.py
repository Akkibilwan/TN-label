import streamlit as st
import os
import io
import sqlite3 # Still needed just for the Binary type hint maybe, or remove completely if not used
import base64
import zipfile
from datetime import datetime
from PIL import Image
from openai import OpenAI
import re
import pathlib # For path manipulation
import shutil # For creating zip archives from folders

# --- Configuration ---
LIBRARY_DIR = "thumbnail_library" # Main directory for storing category folders

# Set page configuration
st.set_page_config(
    page_title="Thumbnail Analyzer (Folder Mode)",
    page_icon="📁",
    layout="wide"
)

# ---------- Custom CSS ----------
# (Keep the existing CSS, maybe update button styling slightly if needed)
st.markdown("""
<style>
    /* Existing CSS */
    .main { background-color: #0f0f0f; color: #f1f1f1; }
    .stApp { background-color: #0f0f0f; }
    h1, h2, h3 { color: #f1f1f1; font-family: 'Roboto', sans-serif; }
    p, li, div[data-testid="stMarkdownContainer"] { color: #aaaaaa; }
    .stButton>button { background-color: #444; color: #f1f1f1; border: 1px solid #666; border-radius: 4px; padding: 8px 16px; font-weight: 500; margin-top: 5px; margin-right: 5px; }
    .stButton>button:hover { opacity: 0.8; border-color: #888; }
    button:has(span:contains("✅ Add to Library")) { background-color: #4CAF50; border: none;}
    button:disabled:has(span:contains("✔️ Added")) { background-color: #3a7d3d; border: none; opacity: 0.7;}
    button:has(span:contains("⬆️ Add Uploaded Image")) { background-color: #ffc107; border: none; color: #333;}
    .stDownloadButton>button { background-color: #007bff; border: none; }
    button:has(span:contains("⬅️ Back")) { background-color: #6c757d; border: none; }
    button:has(span:contains("Clear Uploads")) { background-color: #dc3545; border: none; }
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] { background-color: #333; color: #f1f1f1; border: 1px solid #555; }
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover { border-color: #aaa; color: #fff; background-color: #444; }
    .thumbnail-container, .db-thumbnail-container { border: 1px solid #303030; border-radius: 8px; padding: 15px; background-color: #181818; margin-bottom: 15px; height: 100%; display: flex; flex-direction: column;}
    .analysis-box { border: 1px dashed #444; padding: 10px; margin-top: 10px; border-radius: 4px; background-color: #202020;}
    .stExpander > div:first-child > button { color: #f1f1f1 !important; }
    img { border-radius: 4px; }
    div[data-testid="stImage"]{ display: flex; justify-content: center; margin-bottom: 10px; }
    .db-thumbnail-container .stImage { flex-grow: 1; }
    .multi-label-tag { background-color: #555; color: #eee; padding: 2px 6px; margin: 2px; border-radius: 3px; font-size: 0.8em; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# ---------- Filesystem Library Functions ----------

def sanitize_foldername(name):
    """Removes or replaces characters invalid for folder names."""
    # Remove leading/trailing whitespace
    name = name.strip()
    # Replace common invalid characters with underscore
    name = re.sub(r'[<>:"/\\|?*.,;]+', '_', name)
    # Replace multiple underscores with single underscore
    name = re.sub(r'_+', '_', name)
    # Ensure name isn't empty after sanitizing
    return name if name else "uncategorized"

def ensure_library_dir():
    """Creates the main library directory if it doesn't exist."""
    pathlib.Path(LIBRARY_DIR).mkdir(parents=True, exist_ok=True)

def save_image_to_category(image_bytes, labels_list, original_filename="thumbnail"):
    """Saves the image bytes into folders for each label."""
    ensure_library_dir()
    saved_paths = []
    base_filename, _ = os.path.splitext(original_filename)
    # Sanitize base filename as well
    base_filename_sanitized = re.sub(r'[^\w\-]+', '_', base_filename).strip('_')[:50] # Limit length
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19] # Use microseconds for uniqueness

    for label in labels_list:
        if not label or label == "Uncategorized":
            continue # Don't save under 'Uncategorized' explicitly unless it's the only label? Decide policy. Let's skip for now.

        sanitized_label = sanitize_foldername(label)
        category_path = pathlib.Path(LIBRARY_DIR) / sanitized_label
        category_path.mkdir(parents=True, exist_ok=True) # Create folder if needed

        # Create a unique filename
        filename = f"{base_filename_sanitized}_{timestamp}.jpg" # Save as JPEG
        filepath = category_path / filename
        counter = 1
        while filepath.exists(): # Ensure unique filename within folder
             filename = f"{base_filename_sanitized}_{timestamp}_{counter}.jpg"
             filepath = category_path / filename
             counter += 1

        try:
            # Save the image bytes (assumed to be JPEG already from processing step)
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            saved_paths.append(str(filepath))
        except Exception as e:
            st.error(f"Error saving image to '{filepath}': {e}")
            return False, [] # Indicate failure
    return True, saved_paths # Indicate success

def get_categories_from_folders():
    """Lists subdirectories in the library dir as categories."""
    ensure_library_dir()
    try:
        return sorted([d.name for d in pathlib.Path(LIBRARY_DIR).iterdir() if d.is_dir()])
    except FileNotFoundError:
        return []

def get_images_in_category(category_name):
    """Lists image files in a specific category folder."""
    sanitized_category = sanitize_foldername(category_name)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_category
    image_files = []
    if category_path.is_dir():
        for item in category_path.iterdir():
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_files.append(item) # Store Path objects
    return sorted(image_files, key=os.path.getmtime, reverse=True) # Sort by modification time

# ---------- OpenAI API Setup ----------
# (setup_openai_client remains the same)
def setup_openai_client():
    api_key = None
    if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password", key="api_key_input")
        if not api_key:
            return None
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.sidebar.error(f"Error initializing OpenAI client: {e}. Check API key and permissions.")
        return None

# ---------- Utility Function ----------
# (encode_image remains the same)
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# ---------- OpenAI Analysis & Classification Function (Modified: No Reason) ----------
def analyze_and_classify_thumbnail(client: OpenAI, image_bytes: bytes):
    """
    Analyzes thumbnail for labels only (up to 3).
    """
    if not client:
        return [], "OpenAI client not initialized."

    base64_image = encode_image(image_bytes)
    image_data_uri = f"data:image/jpeg;base64,{base64_image}"

    category_definitions_text = """
    1.  Text-Dominant: ...
    2.  Minimalist / Clean: ...
    3.  Face-Focused: ...
    4.  Before & After: ...
    5.  Comparison / Versus: ...
    6.  Collage / Multi-Image: ...
    7.  Image-Focused: ...
    8.  Branded: ...
    9.  Curiosity Gap / Intrigue: ...
    10. Other / Unclear: ...
    """ # Keep definitions for context
    valid_categories = {
        "Text-Dominant", "Minimalist / Clean", "Face-Focused",
        "Before & After", "Comparison / Versus", "Collage / Multi-Image",
        "Image-Focused", "Branded", "Curiosity Gap / Intrigue", "Other / Unclear"
    }
    label_delimiter_for_ai = ";;" # Tell AI to use this delimiter

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                 {
                    "role": "system",
                    "content": f"You are an expert analyst of YouTube thumbnail visual styles. Analyze the provided image and identify the **top 1 to 3 most relevant** visual style categories using the provided definitions. If only one category clearly applies, list only that one. Respond ONLY with the category names, separated by '{label_delimiter_for_ai}'. Do NOT include 'Labels:' prefix or any reasoning."
                 },
                 {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Classify this thumbnail using ONLY these definitions, providing 1 to 3 relevant labels separated by '{label_delimiter_for_ai}':\n{category_definitions_text}\n\nOutput ONLY the label(s)."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri, "detail": "low"}
                        }
                    ]
                 }
            ],
            temperature=0.2,
            max_tokens=50 # Reduced max_tokens as only labels needed
        )
        result = response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error during OpenAI analysis: {e}")
        return [], "Analysis failed due to an API error."

    # Parse the multi-label output (only labels expected now)
    labels = []
    reason = "Reason not stored in filesystem mode." # Static reason
    try:
        if result:
            # Split labels by delimiter and validate
            raw_labels = [lbl.strip() for lbl in result.split(label_delimiter_for_ai) if lbl.strip()]
            for raw_label in raw_labels:
                found = False
                for valid_cat in valid_categories:
                    if valid_cat.lower() == raw_label.lower():
                        labels.append(valid_cat)
                        found = True
                        break
                if not found:
                    st.warning(f"AI returned unrecognized label: '{raw_label}'. Discarding.")

            labels = labels[:3] # Ensure max 3 labels

    except Exception as parse_error:
        st.warning(f"Could not parse AI labels response: '{result}'. Error: {parse_error}")
        labels = []

    if not labels:
        labels = ["Uncategorized"]

    # Return list of labels and a static reason string
    return labels, reason

# ---------- Callback function for Add to Library button ----------
def add_to_library_callback(file_id, image_bytes, labels_list, filename):
    """Saves image to folders and updates session state."""
    success, _ = save_image_to_category(image_bytes, labels_list, filename)
    if success:
        st.session_state[f'upload_cache_{file_id}']['status'] = 'added'
        st.toast(f"Thumbnail saved to categories: {', '.join(labels_list)}!", icon="✅")
    else:
        st.toast(f"Failed to save thumbnail to library folders.", icon="❌")

# ---------- Callback for Direct Add in Library Explorer ----------
def add_direct_to_library_callback(file_id, image_bytes, selected_category, filename):
    """Saves a manually uploaded image to the selected category folder."""
    # Save only to the selected category
    success, _ = save_image_to_category(image_bytes, [selected_category], filename)
    if success:
        st.session_state[f'direct_added_{file_id}'] = True
        st.toast(f"Image added directly to '{selected_category}' folder!", icon="⬆️")
        # We might need to manually trigger a rerun if we want immediate refresh here
        # Or rely on user interaction
    else:
        st.toast(f"Failed to add image directly to library folder.", icon="❌")

# ---------- Upload and Process Function (Using Session State) ----------
def upload_and_process(client: OpenAI):
    st.header("Upload & Analyze Thumbnails")
    st.info("Upload images, get AI category suggestions, and click '✅ Add to Library' to save images to category folders.")

    if 'upload_cache' not in st.session_state:
        st.session_state.upload_cache = {}

    uploaded_files = st.file_uploader(
        "Choose thumbnail images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    # Process newly uploaded files and add to cache
    if uploaded_files:
        new_files_processed = False
        for uploaded_file in uploaded_files:
            # Use a combination of name and size for a more stable ID if file ID not available
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_id not in st.session_state.upload_cache:
                new_files_processed = True
                try:
                    image_bytes = uploaded_file.getvalue()
                    display_image = Image.open(io.BytesIO(image_bytes)) # Check if it's a valid image
                    img_byte_arr = io.BytesIO()
                    processed_image = display_image.convert('RGB') # Ensure RGB for JPEG save
                    processed_image.save(img_byte_arr, format='JPEG', quality=85)
                    processed_image_bytes = img_byte_arr.getvalue()

                    st.session_state.upload_cache[file_id] = {
                        'name': uploaded_file.name,
                        'original_bytes': image_bytes,
                        'processed_bytes': processed_image_bytes, # Use JPEG bytes
                        'labels': None,
                        'reason': "Reason not stored.", # Static reason
                        'status': 'uploaded'
                    }
                except Exception as e:
                     st.error(f"Error reading file {uploaded_file.name}: {e}")
                     st.session_state.upload_cache[file_id] = {'status': 'error', 'error_msg': str(e), 'name': uploaded_file.name}
        # If new files were added, maybe trigger a rerun to ensure they display immediately?
        # This can sometimes cause loops, use carefully. Let's omit for now.
        # if new_files_processed: st.rerun()


    # --- Display and Process items from Cache ---
    if st.session_state.upload_cache:
        st.markdown("---")
        if st.button("Clear Uploads and Analyses", key="clear_uploads"):
            st.session_state.upload_cache = {}
            st.rerun()

        num_columns = 3
        cols = st.columns(num_columns)
        col_index = 0

        # Display items from cache
        for file_id, item_data in list(st.session_state.upload_cache.items()):
             with cols[col_index % num_columns]:
                 with st.container():
                     st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
                     try:
                         display_image = Image.open(io.BytesIO(item_data['original_bytes']))
                         st.image(display_image, caption=f"{item_data['name']}", use_container_width=True)

                         analysis_placeholder = st.empty()

                         if item_data['status'] == 'uploaded':
                             # Show Analyze button immediately if client is ready
                             if client:
                                 if st.button("Analyze", key=f"analyze_{file_id}"):
                                     st.session_state.upload_cache[file_id]['status'] = 'analyzing'
                                     st.rerun()
                             else:
                                 analysis_placeholder.warning("Waiting for API Key...")

                         elif item_data['status'] == 'analyzing':
                             with analysis_placeholder.container():
                                 with st.spinner(f"Analyzing {item_data['name']}..."):
                                     # Pass JPEG bytes for analysis
                                     labels, reason = analyze_and_classify_thumbnail(client, item_data['processed_bytes'])
                                     st.session_state.upload_cache[file_id]['labels'] = labels
                                     st.session_state.upload_cache[file_id]['reason'] = reason # Store static reason
                                     st.session_state.upload_cache[file_id]['status'] = 'analyzed'
                                     st.rerun()

                         elif item_data['status'] in ['analyzed', 'added']:
                             with analysis_placeholder.container():
                                 st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                                 labels = item_data.get('labels', ['Uncategorized'])
                                 label_tags_html = "".join([f'<span class="multi-label-tag">{lbl}</span>' for lbl in labels])
                                 st.markdown(f"**Suggested:** {label_tags_html}", unsafe_allow_html=True)
                                 # Reason is not displayed as it's not saved meaningfully

                                 is_added = (item_data['status'] == 'added')

                                 st.button(
                                     "✅ Add to Library" if not is_added else "✔️ Added",
                                     key=f'btn_add_{file_id}',
                                     on_click=add_to_library_callback,
                                     # Pass JPEG bytes and original name for saving
                                     args=(file_id, item_data['processed_bytes'], labels, item_data['name']),
                                     disabled=is_added or "Uncategorized" in labels
                                 )
                                 st.markdown('</div>', unsafe_allow_html=True)

                         elif item_data['status'] == 'error':
                             analysis_placeholder.error(f"Error: {item_data.get('error_msg', 'Unknown processing error')}")

                     except Exception as e:
                          st.error(f"Error displaying item {item_data.get('name', file_id)}: {e}")
                     finally:
                          st.markdown('</div>', unsafe_allow_html=True)

             col_index += 1

    elif not uploaded_files: # Show only if no files uploaded AND cache is empty
         st.markdown("<p style='text-align: center; font-style: italic;'>Upload some thumbnails to get started!</p>", unsafe_allow_html=True)

# ---------- Function to create Zip File from Folder ----------
def create_zip_from_folder(category_name):
    """Creates a zip file containing images from a category folder."""
    sanitized_category = sanitize_foldername(category_name)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_category
    zip_buffer = io.BytesIO()

    if not category_path.is_dir():
        st.error(f"Category folder '{category_name}' not found.")
        return None

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in category_path.glob('*'): # Use glob to find files
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    # Arcname determines the name inside the zip (use original filename)
                    zipf.write(item, arcname=item.name)
                except Exception as zip_err:
                    st.warning(f"Could not add file '{item.name}' to zip: {zip_err}")

    zip_buffer.seek(0)
    return zip_buffer

# ---------- Library Explorer (Using Filesystem) ----------
def library_explorer():
    st.header("Thumbnail Library Explorer")
    st.markdown("Browse thumbnails saved in category folders. You can also directly upload to a selected category.")
    categories = get_categories_from_folders() # Get folders as categories

    if "selected_category_folder" not in st.session_state:
        st.session_state.selected_category_folder = None

    # Category Selection Grid
    if st.session_state.selected_category_folder is None:
        st.markdown("### Select a Category Folder to View")
        if not categories:
            st.info("The library is empty. Add some analyzed thumbnails first!")
            return
        cols_per_row = 4
        num_categories = len(categories)
        num_rows = (num_categories + cols_per_row - 1) // cols_per_row
        for i in range(num_rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < num_categories:
                    cat_name = categories[idx]
                    if cols[j].button(cat_name, key=f"btn_lib_{cat_name}", use_container_width=True):
                        st.session_state.selected_category_folder = cat_name
                        st.rerun()
    else:
        # Display Selected Category Content
        selected_category = st.session_state.selected_category_folder
        st.markdown(f"### Category Folder: **{selected_category}**")

        # --- Top Bar: Back Button, Direct Upload Expander, Download Button ---
        top_cols = st.columns([0.25, 0.45, 0.3])
        with top_cols[0]:
            if st.button("⬅️ Back to Categories", key="back_button", use_container_width=True):
                st.session_state.selected_category_folder = None
                st.rerun()

        # --- Direct Upload Section ---
        with top_cols[1]:
            with st.expander(f"⬆️ Add Image Directly to '{selected_category}' Folder"):
                direct_uploaded_file = st.file_uploader(
                    f"Upload image for '{selected_category}'",
                    type=["jpg", "jpeg", "png"],
                    key=f"direct_upload_{selected_category}" # One uploader per category view
                )
                if direct_uploaded_file:
                    file_id = f"direct_{selected_category}_{direct_uploaded_file.name}_{direct_uploaded_file.size}"
                    if f'direct_added_{file_id}' not in st.session_state:
                         st.session_state[f'direct_added_{file_id}'] = False
                    is_added = st.session_state[f'direct_added_{file_id}']

                    st.image(direct_uploaded_file, width=150)
                    img_bytes_direct = direct_uploaded_file.getvalue()
                    img_direct = Image.open(io.BytesIO(img_bytes_direct))
                    img_byte_arr_direct = io.BytesIO()
                    img_direct = img_direct.convert("RGB") # Ensure RGB
                    img_direct.save(img_byte_arr_direct, format='JPEG', quality=85)
                    processed_bytes_direct = img_byte_arr_direct.getvalue()

                    st.button(
                        f"⬆️ Add Uploaded Image" if not is_added else "✔️ Added",
                        key=f"btn_direct_add_{file_id}",
                        on_click=add_direct_to_library_callback,
                        args=(file_id, processed_bytes_direct, selected_category, direct_uploaded_file.name),
                        disabled=is_added
                    )

        image_files = get_images_in_category(selected_category) # Get list of Path objects

        if image_files:
             # --- Download Button ---
             with top_cols[2]:
                  # Create zip directly from folder path
                  zip_buffer = create_zip_from_folder(selected_category)
                  if zip_buffer:
                      st.download_button(
                          label=f"⬇️ Download ({len(image_files)}) Zip",
                          data=zip_buffer,
                          file_name=f"{sanitize_foldername(selected_category)}_thumbnails.zip",
                          mime="application/zip",
                          key=f"download_{selected_category}",
                          use_container_width=True
                      )

             # --- Thumbnail Display Grid ---
             st.markdown("---")
             cols_per_row_thumbs = 4
             thumb_cols = st.columns(cols_per_row_thumbs)
             col_idx = 0

             for image_path in image_files:
                 with thumb_cols[col_idx % cols_per_row_thumbs]:
                     with st.container():
                         st.markdown('<div class="db-thumbnail-container">', unsafe_allow_html=True)
                         try:
                             # Display image directly from path
                             st.image(str(image_path), caption=f"{image_path.name}", use_container_width=True)
                             # No Reason or Labels stored with file system approach easily
                             # Optionally show file size or modified date
                             # file_stat = image_path.stat()
                             # st.caption(f"Size: {file_stat.st_size // 1024} KB")
                         except Exception as img_err:
                              st.warning(f"Could not load image: {image_path.name}: {img_err}")
                         st.markdown('</div>', unsafe_allow_html=True)
                 col_idx += 1
        else:
            st.info(f"No thumbnails found in the folder: '{selected_category}'. Add some via Upload & Analyze or Directly.")


# ---------- Main App ----------
def main():
    # Ensure library base directory exists on startup
    ensure_library_dir()

    # --- Initialize Session State Keys ---
    if 'selected_category_folder' not in st.session_state:
        st.session_state.selected_category_folder = None
    if 'upload_cache' not in st.session_state:
        st.session_state.upload_cache = {}
    # Other keys ('added_', 'direct_added_') are created dynamically

    # --- Sidebar Setup ---
    with st.sidebar:
        st.markdown(
            '<div style="display: flex; align-items: center; padding: 10px 0;">'
            '<span style="color: #FF0000; font-size: 28px; font-weight: bold; margin-right: 5px;">▶️</span>'
            '<h1 style="margin: 0; color: #f1f1f1; font-size: 24px;">Thumbnail Analyzer</h1></div>',
            unsafe_allow_html=True
        )
        st.markdown('<p style="color: #aaaaaa; margin-top: 0;">Analyze & Explore Thumbnails (Folder Storage)</p>', unsafe_allow_html=True)
        st.markdown("---")

        client = setup_openai_client()

        menu = st.radio(
            "Navigation",
            ["Upload & Analyze", "Library Explorer"],
            key="nav_menu",
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.info(f"Thumbnails added to the library are stored in subfolders within the '{LIBRARY_DIR}' directory.")
        st.caption(f"Using OpenAI model for analysis.")

    # --- Main Content Area ---
    if menu == "Upload & Analyze":
        if not client:
            st.error("❌ OpenAI client not initialized. Please provide a valid API key in the sidebar.")
        else:
            upload_and_process(client)
    elif menu == "Library Explorer":
        library_explorer()

if __name__ == "__main__":
    main()
