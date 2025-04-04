# Updated Code - Includes MORE categories

import streamlit as st
import os
import io
# import sqlite3 # No longer needed
import base64
import zipfile
from datetime import datetime
from PIL import Image
from openai import OpenAI
import re
import pathlib # For path manipulation
import shutil # For creating zip archives from folders (though zipfile is used)
import time

# --- Configuration ---
LIBRARY_DIR = "thumbnail_library" # Main directory for storing category folders

# --- Updated Standard Category Definitions ---
STANDARD_CATEGORIES = [
    # Original + Contextual
    "Text-Dominant",
    "Minimalist / Clean",
    "Face-Focused",
    "Before & After",
    "Comparison / Versus",
    "Collage / Multi-Image",
    "Image-Focused",
    "Branded",
    "Curiosity Gap / Intrigue",
    # Newly Added
    "High Contrast",
    "Gradient Background",
    "Bordered / Framed",
    "Inset / PiP", # Picture-in-Picture
    "Arrow/Circle Emphasis",
    "Icon-Driven",
    "Retro / Vintage",
    "Hand-Drawn / Sketch",
    "Textured Background",
    "Extreme Close-Up (Object)",
    # Fallback
    "Other / Unclear"
]


# Set page configuration
st.set_page_config(
    page_title="Thumbnail Analyzer (Ext Cat)", # Updated Title slightly
    page_icon="📁",
    layout="wide"
)

# ---------- Custom CSS ----------
# (CSS remains the same)
st.markdown("""
<style>
    /* Existing CSS */
    /* ... */
</style>
""", unsafe_allow_html=True)


# ---------- Filesystem Library Functions ----------

def sanitize_foldername(name):
    # (Sanitize function remains the same)
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*.,;]+', '_', name)
    name = re.sub(r'_+', '_', name)
    if name.upper() in ["CON", "PRN", "AUX", "NUL"] or re.match(r"^(COM|LPT)[1-9]$", name.upper()):
        name = f"_{name}_"
    return name if name else "uncategorized"

def ensure_library_dir():
    # (Ensure dir function remains the same)
    pathlib.Path(LIBRARY_DIR).mkdir(parents=True, exist_ok=True)

# Function to Pre-create Standard Category Folders
def create_predefined_category_folders(category_list):
    """Creates folders for standard categories if they don't exist."""
    ensure_library_dir()
    st.sidebar.write("Checking standard category folders...")
    created_count = 0
    for category_name in category_list:
        sanitized_name = sanitize_foldername(category_name)
        # Avoid creating folders for generic/empty names unless explicitly desired
        if not sanitized_name or sanitized_name in ["uncategorized", "other_unclear"]:
            continue

        folder_path = pathlib.Path(LIBRARY_DIR) / sanitized_name
        if not folder_path.exists():
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                created_count += 1
            except Exception as e:
                st.sidebar.warning(f"Could not create folder for '{category_name}': {e}")
    # Only show counts if something actually changed or needed checking
    # if created_count > 0:
    #      st.sidebar.caption(f"Created {created_count} new category folders.")
    # else:
    #      st.sidebar.caption(f"Standard category folders checked.")


# Modified for single label
def save_image_to_category(image_bytes, label, original_filename="thumbnail"):
    # (Function logic remains the same - saves to single label folder)
    ensure_library_dir()
    if not label or label in ["Uncategorized", "Other / Unclear"]:
        st.warning(f"Cannot save image '{original_filename}' with label '{label}'.")
        return False, None

    base_filename, _ = os.path.splitext(original_filename)
    base_filename_sanitized = re.sub(r'[^\w\-]+', '_', base_filename).strip('_')[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]

    sanitized_label = sanitize_foldername(label)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_label
    category_path.mkdir(parents=True, exist_ok=True) # Create just in case

    filename = f"{base_filename_sanitized}_{timestamp}.jpg"
    filepath = category_path / filename
    counter = 1
    while filepath.exists():
         filename = f"{base_filename_sanitized}_{timestamp}_{counter}.jpg"; counter += 1
         filepath = category_path / filename

    try:
        with open(filepath, "wb") as f: f.write(image_bytes)
        return True, str(filepath)
    except Exception as e: st.error(f"Error saving image to '{filepath}': {e}"); return False, None


def get_categories_from_folders():
    # (Function remains the same)
    ensure_library_dir()
    try: return sorted([d.name for d in pathlib.Path(LIBRARY_DIR).iterdir() if d.is_dir() and not d.name.startswith('.')])
    except FileNotFoundError: return []

def get_images_in_category(category_name):
    # (Function remains the same)
    sanitized_category = sanitize_foldername(category_name)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_category
    image_files = []
    if category_path.is_dir():
        for item in category_path.iterdir():
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'] and not item.name.startswith('.'):
                image_files.append(item)
    return sorted(image_files, key=os.path.getmtime, reverse=True)


def delete_image_file(image_path_str):
    # (Function remains the same)
    try:
        file_path = pathlib.Path(image_path_str)
        if file_path.is_file(): file_path.unlink(); st.toast(f"Deleted: {file_path.name}", icon="🗑️"); return True
        else: st.error(f"File not found for deletion: {file_path.name}"); return False
    except Exception as e: st.error(f"Error deleting file {image_path_str}: {e}"); return False


# ---------- OpenAI API Setup ----------
# (setup_openai_client remains the same)
def setup_openai_client():
    api_key = None
    # ... (rest of the function is the same) ...
    if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets: api_key = st.secrets["OPENAI_API_KEY"]
    else: api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key: api_key = st.sidebar.text_input("Enter OpenAI API key:", type="password", key="api_key_input_sidebar")
    if not api_key: return None
    try: client = OpenAI(api_key=api_key); return client
    except Exception as e: st.sidebar.error(f"Error initializing OpenAI client: {e}. Check API key."); return None


# ---------- Utility Function ----------
# (encode_image remains the same)
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# ---------- OpenAI Analysis & Classification Function (Updated Categories) ----------
def analyze_and_classify_thumbnail(client: OpenAI, image_bytes: bytes):
    """ Analyzes thumbnail for the single most relevant label from the expanded list. """
    if not client: return "Uncategorized", "OpenAI client not initialized."

    base64_image = encode_image(image_bytes)
    image_data_uri = f"data:image/jpeg;base64,{base64_image}"

    # --- Updated Category Definitions for Prompt ---
    category_definitions_list = [
        "Text-Dominant: Large, bold typography is the primary focus.",
        "Minimalist / Clean: Uncluttered, simple background, few elements.",
        "Face-Focused: Close-up, expressive human face is central.",
        "Before & After: Divided layout showing two distinct states.",
        "Comparison / Versus: Layout structured comparing items/ideas.",
        "Collage / Multi-Image: Composed of multiple distinct images arranged together.",
        "Image-Focused: A single, high-quality photo/illustration is dominant.",
        "Branded: Prominent, consistent channel branding is the key feature.",
        "Curiosity Gap / Intrigue: Deliberately obscures info (blurring, arrows, etc.).",
        "High Contrast: Stark differences in color values (e.g., brights on black).",
        "Gradient Background: Prominent color gradient as background/overlay.",
        "Bordered / Framed: Distinct border around the thumbnail or key elements.",
        "Inset / PiP: Smaller image inset within a larger one (e.g., reaction, tutorial).",
        "Arrow/Circle Emphasis: Prominent graphical arrows/circles drawing attention.",
        "Icon-Driven: Relies mainly on icons or simple vector graphics.",
        "Retro / Vintage: Evokes a specific past era stylistically.",
        "Hand-Drawn / Sketch: Uses elements styled to look drawn or sketched.",
        "Textured Background: Background is a distinct visual texture (paper, wood, etc.).",
        "Extreme Close-Up (Object): Intense focus on a non-face object/detail.",
        "Other / Unclear: Doesn't fit well or mixes styles heavily."
    ]
    category_definitions_text = "\n".join([f"- {cat}" for cat in category_definitions_list]) # Simple list format

    # --- Updated Validation Set ---
    # Use the STANDARD_CATEGORIES list defined globally
    valid_categories = set(STANDARD_CATEGORIES)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                 {
                    "role": "system",
                    "content": f"You are an expert analyst of YouTube thumbnail visual styles. Analyze the provided image and identify the **single most relevant** visual style category using ONLY the following definitions. Respond ONLY with the single category name from the list. Do NOT include numbers, prefixes like 'Label:', reasoning, or explanation."
                 },
                 { "role": "user", "content": [ { "type": "text", "text": f"Classify this thumbnail using ONLY these definitions, providing the single most relevant category name:\n{category_definitions_text}\n\nOutput ONLY the single category name." }, { "type": "image_url", "image_url": {"url": image_data_uri, "detail": "low"} } ] }
            ],
            temperature=0.1, max_tokens=40 # Increased slightly for longer category names
        )
        result = response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error during OpenAI analysis: {e}")
        return "Uncategorized", "Analysis failed due to an API error."

    # Validate the single label output
    label = "Uncategorized"
    reason = "Reason not stored."
    try:
        if result:
            found = False
            # Check against STANDARD_CATEGORIES (case-insensitive)
            for valid_cat in valid_categories:
                if valid_cat.strip().lower() == result.strip().lower():
                    label = valid_cat # Use the official casing
                    found = True
                    break
            if not found:
                st.warning(f"AI returned unrecognized category: '{result}'. Classifying as 'Other / Unclear'.")
                # Attempt fallback or default
                label = "Other / Unclear"
        else: st.warning("AI returned an empty category response.")

    except Exception as parse_error:
        st.warning(f"Could not parse AI label response: '{result}'. Error: {parse_error}")
        label = "Uncategorized"

    return label, reason


# ---------- Callbacks ----------
# (add_to_library_callback remains the same)
def add_to_library_callback(file_id, image_bytes, label, filename):
    success, _ = save_image_to_category(image_bytes, label, filename)
    if success:
        if 'upload_cache' in st.session_state and file_id in st.session_state.upload_cache:
            st.session_state.upload_cache[file_id]['status'] = 'added'
            st.toast(f"Thumbnail saved to '{label}' folder!", icon="✅")
        else: st.warning(f"Cache status update failed for {filename}. File likely saved."); st.toast(f"Thumbnail saved.", icon="✅")
    else: st.toast(f"Failed to save thumbnail.", icon="❌")

# (add_direct_to_library_callback remains the same)
def add_direct_to_library_callback(file_id, image_bytes, selected_category, filename):
    success, _ = save_image_to_category(image_bytes, selected_category, filename)
    if success: st.session_state[f'direct_added_{file_id}'] = True; st.toast(f"Image added to '{selected_category}' folder!", icon="⬆️")
    else: st.toast(f"Failed to add image directly.", icon="❌")

# (analyze_all_callback remains the same)
def analyze_all_callback():
    # ... (same logic) ...
    if 'upload_cache' in st.session_state:
        triggered_count = 0
        for file_id, item_data in st.session_state.upload_cache.items():
            if item_data.get('status') == 'uploaded': st.session_state.upload_cache[file_id]['status'] = 'analyzing'; triggered_count += 1
        if triggered_count > 0: st.toast(f"Triggered analysis for {triggered_count} thumbnail(s).", icon="🧠")
        else: st.toast("No thumbnails awaiting analysis.", icon="🤷")


# ---------- Upload and Process Function ----------
# (upload_and_process remains the same - displays single label)
def upload_and_process(client: OpenAI):
    # ... (Function logic is largely the same, ensure it uses the updated analyze function) ...
    st.header("Upload & Analyze Thumbnails")
    st.info("Upload images, click '🧠 Analyze All Pending', then '✅ Add to Library' to save.")

    if 'upload_cache' not in st.session_state: st.session_state.upload_cache = {}

    uploaded_files = st.file_uploader( "Choose thumbnail images...", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key="file_uploader")

    # Process newly uploaded files
    if uploaded_files:
        # ... (File reading/processing logic is the same) ...
        for uploaded_file in uploaded_files:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_id not in st.session_state.upload_cache:
                try:
                    # ... (same image processing) ...
                    image_bytes = uploaded_file.getvalue()
                    display_image = Image.open(io.BytesIO(image_bytes)); display_image.verify(); display_image = Image.open(io.BytesIO(image_bytes))
                    img_byte_arr = io.BytesIO(); processed_image = display_image.convert('RGB'); processed_image.save(img_byte_arr, format='JPEG', quality=85)
                    processed_image_bytes = img_byte_arr.getvalue()
                    st.session_state.upload_cache[file_id] = { 'name': uploaded_file.name, 'original_bytes': image_bytes, 'processed_bytes': processed_image_bytes, 'label': None, 'reason': "Reason not stored.", 'status': 'uploaded' }
                except Exception as e: st.error(f"Error reading {uploaded_file.name}: {e}"); st.session_state.upload_cache[file_id] = {'status': 'error', 'error_msg': str(e), 'name': uploaded_file.name}

    # Display and Process items from Cache
    if st.session_state.upload_cache:
        st.markdown("---")
        # Control Buttons
        col1, col2 = st.columns(2)
        with col1:
             items_to_analyze = any(item.get('status') == 'uploaded' for item in st.session_state.upload_cache.values())
             analyze_all_disabled = not items_to_analyze or not client
             if st.button("🧠 Analyze All Pending", key="analyze_all", on_click=analyze_all_callback, disabled=analyze_all_disabled, use_container_width=True): st.rerun()
        with col2:
             if st.button("Clear Uploads and Analyses", key="clear_uploads", use_container_width=True): st.session_state.upload_cache = {}; st.rerun()
        st.markdown("---")

        # Thumbnail Grid
        num_columns = 3; cols = st.columns(num_columns); col_index = 0
        for file_id, item_data in list(st.session_state.upload_cache.items()):
            if not isinstance(item_data, dict) or 'status' not in item_data: continue
            with cols[col_index % num_columns]:
                with st.container():
                    st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
                    try:
                        # (Image display logic is the same)
                        display_image = Image.open(io.BytesIO(item_data['original_bytes'])); st.image(display_image, caption=f"{item_data['name']}", use_container_width=True)
                        analysis_placeholder = st.empty()
                        # Status handling logic (Analyze All triggers 'analyzing' status)
                        if item_data['status'] == 'uploaded': analysis_placeholder.info("Ready for analysis (Click 'Analyze All').")
                        elif item_data['status'] == 'analyzing':
                             # (Analysis logic is the same, calls analyze_and_classify_thumbnail)
                             with analysis_placeholder.container():
                                 with st.spinner(f"Analyzing {item_data['name']}..."):
                                     label, reason = analyze_and_classify_thumbnail(client, item_data['processed_bytes'])
                                     st.session_state.upload_cache[file_id]['label'] = label; st.session_state.upload_cache[file_id]['reason'] = reason; st.session_state.upload_cache[file_id]['status'] = 'analyzed'
                                     st.rerun()
                        elif item_data['status'] in ['analyzed', 'added']:
                            # (Display logic is the same, shows single label)
                            with analysis_placeholder.container():
                                st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                                label = item_data.get('label', 'Uncategorized')
                                st.markdown(f"**Suggested:** `{label}`")
                                is_added = (item_data['status'] == 'added')
                                st.button("✅ Add to Library" if not is_added else "✔️ Added", key=f'btn_add_{file_id}', on_click=add_to_library_callback, args=(file_id, item_data['processed_bytes'], label, item_data['name']), disabled=is_added or label == "Uncategorized" or not label)
                                st.markdown('</div>', unsafe_allow_html=True)
                        elif item_data['status'] == 'error': analysis_placeholder.error(f"Error: {item_data.get('error_msg', 'Unknown error')}")
                    except Exception as e: st.error(f"Display error for {item_data.get('name', file_id)}: {e}")
                    finally: st.markdown('</div>', unsafe_allow_html=True)
            col_index += 1
    elif not uploaded_files: st.markdown("<p style='text-align: center; font-style: italic;'>Upload thumbnails to start analysis!</p>", unsafe_allow_html=True)


# ---------- Function to create Zip File from Folder ----------
# (create_zip_from_folder remains the same)
def create_zip_from_folder(category_name):
    # ... (same logic) ...
    sanitized_category = sanitize_foldername(category_name); category_path = pathlib.Path(LIBRARY_DIR) / sanitized_category
    zip_buffer = io.BytesIO(); added_files = 0
    if not category_path.is_dir(): return None
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in category_path.glob('*'):
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'] and not item.name.startswith('.'):
                try: zipf.write(item, arcname=item.name); added_files += 1
                except Exception as zip_err: st.warning(f"Zip error for {item.name}: {zip_err}")
    if added_files == 0: return None
    zip_buffer.seek(0); return zip_buffer


# ---------- Library Explorer ----------
# (library_explorer remains the same - displays folders, includes delete)
def library_explorer():
    # ... (Function logic remains the same) ...
    st.header("Thumbnail Library Explorer")
    st.markdown("Browse saved thumbnails by category folder. Delete images or download categories.")
    categories = get_categories_from_folders()
    if 'confirm_delete_path' not in st.session_state: st.session_state.confirm_delete_path = None
    if "selected_category_folder" not in st.session_state: st.session_state.selected_category_folder = None

    if st.session_state.selected_category_folder is None:
        # Category Selection Grid
        st.markdown("### Select a Category Folder to View")
        if not categories: st.info("Library is empty."); return
        cols_per_row = 4; num_categories = len(categories); num_rows = (num_categories + cols_per_row - 1) // cols_per_row
        for i in range(num_rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < num_categories:
                    cat_name = categories[idx]
                    if cols[j].button(cat_name, key=f"btn_lib_{cat_name}", use_container_width=True): st.session_state.selected_category_folder = cat_name; st.rerun()
    else:
        # Display Selected Category Content
        selected_category = st.session_state.selected_category_folder
        st.markdown(f"### Category Folder: **{selected_category}**")
        # Top Bar: Back, Direct Upload, Download
        top_cols = st.columns([0.25, 0.45, 0.3])
        with top_cols[0]:
            if st.button("⬅️ Back to Categories", key="back_button", use_container_width=True): st.session_state.selected_category_folder = None; st.rerun()
        # Direct Upload Expander
        with top_cols[1]:
             with st.expander(f"⬆️ Add Image Directly to '{selected_category}' Folder"):
                 # ... (Direct Upload logic is the same) ...
                 direct_uploaded_file = st.file_uploader(f"Upload image for '{selected_category}'", type=["jpg", "jpeg", "png", "webp"], key=f"direct_upload_{selected_category}")
                 if direct_uploaded_file:
                     file_id = f"direct_{selected_category}_{direct_uploaded_file.name}_{direct_uploaded_file.size}"
                     if f'direct_added_{file_id}' not in st.session_state: st.session_state[f'direct_added_{file_id}'] = False
                     is_added = st.session_state[f'direct_added_{file_id}']
                     st.image(direct_uploaded_file, width=150)
                     try:
                         img_bytes_direct = direct_uploaded_file.getvalue(); img_direct = Image.open(io.BytesIO(img_bytes_direct)); img_direct.verify(); img_direct = Image.open(io.BytesIO(img_bytes_direct)); img_direct = img_direct.convert("RGB")
                         img_byte_arr_direct = io.BytesIO(); img_direct.save(img_byte_arr_direct, format='JPEG', quality=85); processed_bytes_direct = img_byte_arr_direct.getvalue()
                         st.button(f"⬆️ Add Uploaded Image" if not is_added else "✔️ Added", key=f"btn_direct_add_{file_id}", on_click=add_direct_to_library_callback, args=(file_id, processed_bytes_direct, selected_category, direct_uploaded_file.name), disabled=is_added)
                     except Exception as e: st.error(f"Failed to process direct upload: {e}")

        image_files = get_images_in_category(selected_category)
        # Download Button
        if image_files:
             with top_cols[2]:
                  zip_buffer = create_zip_from_folder(selected_category)
                  if zip_buffer: st.download_button(label=f"⬇️ Download ({len(image_files)}) Zip", data=zip_buffer, file_name=f"{sanitize_foldername(selected_category)}_thumbnails.zip", mime="application/zip", key=f"download_{selected_category}", use_container_width=True)

             # Thumbnail Display Grid (with Delete Button)
             st.markdown("---")
             cols_per_row_thumbs = 4; thumb_cols = st.columns(cols_per_row_thumbs); col_idx = 0
             for image_path in image_files:
                 with thumb_cols[col_idx % cols_per_row_thumbs]:
                     with st.container():
                         st.markdown('<div class="db-thumbnail-container">', unsafe_allow_html=True)
                         try:
                             image_path_str = str(image_path)
                             st.image(image_path_str, caption=f"{image_path.name}", use_container_width=True)
                             st.markdown('<div class="delete-button-container">', unsafe_allow_html=True)
                             mtime = image_path.stat().st_mtime; del_key = f"del_{image_path.name}_{mtime}"
                             if st.button("🗑️ Delete", key=del_key, help="Delete this image"):
                                 st.session_state.confirm_delete_path = image_path_str; st.rerun()
                             st.markdown('</div>', unsafe_allow_html=True)
                         except Exception as img_err: st.warning(f"Could not load image: {image_path.name} ({img_err})")
                         finally: st.markdown('</div>', unsafe_allow_html=True)
                 col_idx += 1
        elif not direct_uploaded_file:
            st.info(f"No thumbnails found in the folder: '{selected_category}'.")


# ---------- Delete Confirmation Dialog Function ----------
# (display_delete_confirmation remains the same)
def display_delete_confirmation():
     if 'confirm_delete_path' in st.session_state and st.session_state.confirm_delete_path:
        st.warning(f"**Confirm Deletion:** Are you sure you want to permanently delete `{os.path.basename(st.session_state.confirm_delete_path)}`?")
        col1, col2, col3 = st.columns([1.5, 1, 5])
        with col1:
            if st.button("🔥 Confirm Delete", key="confirm_delete_yes"):
                if delete_image_file(st.session_state.confirm_delete_path): st.session_state.confirm_delete_path = None; st.rerun()
                else: st.session_state.confirm_delete_path = None; st.rerun()
        with col2:
            if st.button("🚫 Cancel", key="confirm_delete_cancel"): st.session_state.confirm_delete_path = None; st.rerun()


# ---------- Main App ----------
def main():
    ensure_library_dir()
    # Create predefined folders on startup
    create_predefined_category_folders(STANDARD_CATEGORIES)

    # Initialize Session State Keys
    if 'selected_category_folder' not in st.session_state: st.session_state.selected_category_folder = None
    if 'upload_cache' not in st.session_state: st.session_state.upload_cache = {}
    if 'confirm_delete_path' not in st.session_state: st.session_state.confirm_delete_path = None

    # --- Sidebar Setup ---
    with st.sidebar:
        # (Sidebar setup remains the same)
        st.markdown('<div>... Sidebar HTML ...</div>', unsafe_allow_html=True) # Placeholder
        st.markdown('<p>Analyze & Explore Thumbnails (Folder Storage)</p>', unsafe_allow_html=True)
        st.markdown("---")
        client = setup_openai_client()
        menu = st.radio("Navigation", ["Upload & Analyze", "Library Explorer"], key="nav_menu", label_visibility="collapsed")
        st.markdown("---")
        st.info(f"Library stored in './{LIBRARY_DIR}'")
        st.caption(f"Using OpenAI model for analysis.")
        # Display standard categories list in sidebar
        with st.expander("Standard Categories"):
             st.markdown("\n".join([f"- {cat}" for cat in STANDARD_CATEGORIES if cat != "Other / Unclear"]))


    # --- Main Content Area ---
    if st.session_state.confirm_delete_path:
         display_delete_confirmation()
    else:
         if menu == "Upload & Analyze":
             if not client: st.error("❌ OpenAI client not initialized. Provide API key.")
             else: upload_and_process(client)
         elif menu == "Library Explorer":
             library_explorer()

if __name__ == "__main__":
    main()
