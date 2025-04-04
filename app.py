import streamlit as st
import os
import io
import base64
import zipfile
from datetime import datetime
from PIL import Image
from openai import OpenAI
import re
import pathlib # For path manipulation
# import shutil # Not strictly needed as zipfile handles folder zipping
import time
import requests # Needed for downloading generated image if URL is returned (though using b64)

# --- Configuration ---
LIBRARY_DIR = "thumbnail_library" # Main directory for storing category folders

# --- Updated Standard Category Definitions (Now with descriptions) ---
STANDARD_CATEGORIES_WITH_DESC = [
    {'name': "Text-Dominant", 'description': "Large, bold typography is the primary focus."},
    {'name': "Minimalist / Clean", 'description': "Uncluttered, simple background, few elements."},
    {'name': "Face-Focused", 'description': "Close-up, expressive human face is central."},
    {'name': "Before & After", 'description': "Divided layout showing two distinct states."},
    {'name': "Comparison / Versus", 'description': "Layout structured comparing items/ideas."},
    {'name': "Collage / Multi-Image", 'description': "Composed of multiple distinct images arranged together."},
    {'name': "Image-Focused", 'description': "A single, high-quality photo/illustration is dominant."},
    {'name': "Branded", 'description': "Prominent, consistent channel branding is the key feature."},
    {'name': "Curiosity Gap / Intrigue", 'description': "Deliberately obscures info (blurring, arrows, etc.)."},
    {'name': "High Contrast", 'description': "Stark differences in color values (e.g., brights on black)."},
    {'name': "Gradient Background", 'description': "Prominent color gradient as background/overlay."},
    {'name': "Bordered / Framed", 'description': "Distinct border around the thumbnail or key elements."},
    {'name': "Inset / PiP", 'description': "Smaller image inset within a larger one (e.g., reaction, tutorial)."},
    {'name': "Arrow/Circle Emphasis", 'description': "Prominent graphical arrows/circles drawing attention."},
    {'name': "Icon-Driven", 'description': "Relies mainly on icons or simple vector graphics."},
    {'name': "Retro / Vintage", 'description': "Evokes a specific past era stylistically."},
    {'name': "Hand-Drawn / Sketch", 'description': "Uses elements styled to look drawn or sketched."},
    {'name': "Textured Background", 'description': "Background is a distinct visual texture (paper, wood, etc.)."},
    {'name': "Extreme Close-Up (Object)", 'description': "Intense focus on a non-face object/detail."},
    {'name': "Other / Unclear", 'description': "Doesn't fit well or mixes styles heavily."}
]

# Extract just the names for convenience in some functions
STANDARD_CATEGORIES = [cat['name'] for cat in STANDARD_CATEGORIES_WITH_DESC]

# Set page configuration
st.set_page_config(
    page_title="Thumbnail Toolkit (Analyze & Generate)", # Updated Title
    page_icon="🖼️", # Changed icon
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Existing CSS */
    .thumbnail-container, .db-thumbnail-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
        position: relative; /* Needed for absolute positioning of delete button */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 100%; /* Make containers equal height in a row */
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Push button to bottom */
    }
    .analysis-box {
        margin-top: 10px;
        padding: 8px;
        background-color: #e9ecef;
        border-radius: 4px;
        border: 1px solid #ced4da;
    }
    .analysis-box p { /* Style paragraphs inside analysis box */
       margin-bottom: 5px;
       font-size: 0.9em;
    }
    .delete-button-container {
        /* Removed absolute positioning, button flows naturally */
        margin-top: 10px; /* Add space above the button */
        text-align: center; /* Center the button */
    }
    .delete-button-container button {
       width: 80%; /* Make delete button slightly smaller */
       background-color: #dc3545; /* Red color */
       color: white;
       border: none;
       padding: 5px 10px;
       border-radius: 4px;
       cursor: pointer;
       transition: background-color 0.2s ease;
    }
    .delete-button-container button:hover {
        background-color: #c82333; /* Darker red on hover */
    }
    .stButton>button { /* Style general Streamlit buttons */
        border-radius: 5px;
        padding: 8px 15px;
    }
    .stDownloadButton>button { /* Style download buttons */
        width: 100%;
    }
     /* Ensure generated image fits well */
    .generated-image-container img {
        max-width: 100%;
        height: auto;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    .generated-image-container {
        margin-top: 15px;
        padding: 15px;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Filesystem Library Functions ----------

def sanitize_foldername(name):
    # (Sanitize function remains the same)
    name = name.strip()
    # Remove or replace characters invalid in Windows/Linux/macOS folder names
    # Including .,; which might cause issues in some contexts
    name = re.sub(r'[<>:"/\\|?*.,;]+', '_', name)
    # Replace multiple consecutive underscores with a single one
    name = re.sub(r'_+', '_', name)
    # Handle reserved names in Windows
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
    # st.sidebar.write("Checking standard category folders...") # Less verbose
    created_count = 0
    for category_name in category_list: # Now iterates through names only
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
    # Show message only if folders were created
    if created_count > 0:
        st.sidebar.caption(f"Created {created_count} new category folders.")


# Modified for single label, handles bytes input
def save_image_to_category(image_bytes, label, original_filename="thumbnail"):
    """Saves image bytes to the specified category folder."""
    ensure_library_dir()
    if not label or label in ["Uncategorized", "Other / Unclear"]:
        st.warning(f"Cannot save image '{original_filename}' with label '{label}'. Please select a valid category.")
        return False, None

    # Sanitize base filename more aggressively
    base_filename, _ = os.path.splitext(original_filename)
    base_filename_sanitized = re.sub(r'[^\w\-]+', '_', base_filename).strip('_')[:50] # Max 50 chars
    if not base_filename_sanitized: base_filename_sanitized = "image" # Fallback if name becomes empty

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19] # microseconds added

    sanitized_label = sanitize_foldername(label)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_label
    category_path.mkdir(parents=True, exist_ok=True) # Create just in case

    # Determine file extension (try to keep original if known, default to jpg)
    # For generated images (b64 from DALL-E is usually PNG) or processed uploads (we convert to JPG)
    file_extension = ".jpg" # Default for processed uploads
    try:
        # Check common magic numbers
        if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            file_extension = ".png"
        elif image_bytes.startswith(b'\xff\xd8\xff'):
            file_extension = ".jpg"
        elif image_bytes.startswith(b'RIFF') and image_bytes[8:12] == b'WEBP':
             file_extension = ".webp"
        # Add more checks if needed (GIF, etc.)
    except Exception:
        pass # Keep default if check fails or bytes too short

    # If original filename had a known extension, maybe prioritize it?
    # _, orig_ext = os.path.splitext(original_filename)
    # if orig_ext.lower() in ['.png', '.webp', '.jpeg', '.jpg']:
    #      file_extension = orig_ext.lower() # Consider using original extension

    filename = f"{base_filename_sanitized}_{timestamp}{file_extension}"
    filepath = category_path / filename
    counter = 1
    while filepath.exists():
        # Append counter if filename collision
        filename = f"{base_filename_sanitized}_{timestamp}_{counter}{file_extension}"
        filepath = category_path / filename
        counter += 1

    try:
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        return True, str(filepath)
    except Exception as e:
        st.error(f"Error saving image to '{filepath}': {e}")
        return False, None


def get_categories_from_folders():
    # (Function remains the same)
    ensure_library_dir()
    try:
        # List directories, filter out hidden ones (like .DS_Store)
        return sorted([d.name for d in pathlib.Path(LIBRARY_DIR).iterdir() if d.is_dir() and not d.name.startswith('.')])
    except FileNotFoundError:
        return []

def get_images_in_category(category_name):
    # (Function remains the same)
    sanitized_category = sanitize_foldername(category_name)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_category
    image_files = []
    if category_path.is_dir():
        for item in category_path.iterdir():
            # Check for common image extensions, ignore hidden files
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'] and not item.name.startswith('.'):
                image_files.append(item)
    # Sort by modification time, newest first
    return sorted(image_files, key=os.path.getmtime, reverse=True)


def delete_image_file(image_path_str):
    # (Function remains the same)
    try:
        file_path = pathlib.Path(image_path_str)
        if file_path.is_file():
            file_path.unlink() # Delete the file
            st.toast(f"Deleted: {file_path.name}", icon="🗑️")
            return True
        else:
            st.error(f"File not found for deletion: {file_path.name}")
            return False
    except Exception as e:
        st.error(f"Error deleting file {image_path_str}: {e}")
        return False

# ---------- NEW: Function to create Zip File of Entire Library ----------
def create_zip_of_library():
    """Creates a zip file containing all category folders and their images."""
    ensure_library_dir()
    zip_buffer = io.BytesIO()
    added_files_count = 0
    library_path = pathlib.Path(LIBRARY_DIR)

    if not any(library_path.iterdir()): # Check if library directory is empty
        return None, 0

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Iterate through each category folder in the library directory
        for category_folder in library_path.iterdir():
            if category_folder.is_dir() and not category_folder.name.startswith('.'):
                # Iterate through files within the category folder
                for item in category_folder.iterdir(): # Use iterdir instead of glob('*')
                     if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'] and not item.name.startswith('.'):
                        try:
                            # Add file to zip, preserving directory structure
                            arcname = f"{category_folder.name}/{item.name}"
                            zipf.write(item, arcname=arcname)
                            added_files_count += 1
                        except Exception as zip_err:
                            st.warning(f"Could not add {item.name} to zip: {zip_err}")

    if added_files_count == 0:
        return None, 0 # No files were added

    zip_buffer.seek(0)
    return zip_buffer, added_files_count


# ---------- OpenAI API Setup ----------
def setup_openai_client():
    # (setup_openai_client remains the same)
    api_key = None
    # Try Streamlit secrets
    if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        # Try environment variables
        api_key = os.environ.get('OPENAI_API_KEY')

    # If not found, ask user in sidebar
    if not api_key:
        api_key = st.sidebar.text_input(
            "Enter OpenAI API key:",
            type="password",
            key="api_key_input_sidebar",
            help="Required for analyzing and generating thumbnails."
        )

    if not api_key:
        # st.sidebar.warning("OpenAI API key is missing.") # Show warning persistently if needed
        return None # Return None if no key

    try:
        client = OpenAI(api_key=api_key)
        # Optional: Add a simple test call here if needed to verify key
        # client.models.list()
        return client
    except Exception as e:
        st.sidebar.error(f"Error initializing OpenAI client: {e}. Check API key.")
        return None


# ---------- Utility Function ----------
def encode_image(image_bytes):
    # (encode_image remains the same)
    return base64.b64encode(image_bytes).decode('utf-8')

# ---------- OpenAI Analysis & Classification Function (Updated Categories) ----------
def analyze_and_classify_thumbnail(client: OpenAI, image_bytes: bytes):
    """ Analyzes thumbnail for the single most relevant label from the expanded list. """
    if not client:
        return "Uncategorized", "OpenAI client not initialized."

    base64_image = encode_image(image_bytes)
    image_data_uri = f"data:image/jpeg;base64,{base64_image}" # Assuming JPEG bytes after processing

    # --- Use Category Definitions from STANDARD_CATEGORIES_WITH_DESC ---
    category_definitions_list = [f"{cat['name']}: {cat['description']}" for cat in STANDARD_CATEGORIES_WITH_DESC]
    category_definitions_text = "\n".join([f"- {cat_def}" for cat_def in category_definitions_list])

    # --- Use STANDARD_CATEGORIES list for validation ---
    valid_categories = set(STANDARD_CATEGORIES)

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Use the specified model
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert analyst of YouTube thumbnail visual styles. Analyze the provided image and identify the **single most relevant** visual style category using ONLY the following definitions. Respond ONLY with the single category name from the list. Do NOT include numbers, prefixes like 'Label:', reasoning, or explanation."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Classify this thumbnail using ONLY these definitions, providing the single most relevant category name:\n{category_definitions_text}\n\nOutput ONLY the single category name."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri, "detail": "low"} # Use low detail for faster analysis
                        }
                    ]
                }
            ],
            temperature=0.1, # Low temperature for consistency
            max_tokens=50 # Slightly increased buffer for longer names like 'Arrow/Circle Emphasis'
        )
        result = response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error during OpenAI analysis: {e}")
        return "Uncategorized", "Analysis failed due to an API error."

    # Validate the single label output
    label = "Uncategorized" # Default
    reason = "Analysis complete." # Simple default reason

    try:
        if result:
            found = False
            # Check against STANDARD_CATEGORIES (case-insensitive comparison)
            for valid_cat in valid_categories:
                if valid_cat.strip().lower() == result.strip().lower():
                    label = valid_cat # Use the official casing from STANDARD_CATEGORIES
                    found = True
                    break
            if not found:
                st.warning(f"AI returned unrecognized category: '{result}'. Classifying as 'Other / Unclear'.")
                # Fallback to 'Other / Unclear' if defined, else 'Uncategorized'
                label = "Other / Unclear" if "Other / Unclear" in valid_categories else "Uncategorized"
        else:
            st.warning("AI returned an empty category response. Classifying as 'Uncategorized'.")
            label = "Uncategorized"

    except Exception as parse_error:
        st.warning(f"Could not process AI label response: '{result}'. Error: {parse_error}. Classifying as 'Uncategorized'.")
        label = "Uncategorized"

    # The 'reason' part isn't really used or stored meaningfully anymore with the single label focus.
    return label, reason


# ---------- Callbacks ----------
def add_to_library_callback(file_id, image_bytes, label, filename):
    """Callback to save an image (uploaded or generated) to the library."""
    # Note: file_id helps manage state for uploaded items, less critical for generated ones unless tracking saved status
    success, saved_path = save_image_to_category(image_bytes, label, filename)
    if success:
        # Update status for uploaded items if applicable
        if 'upload_cache' in st.session_state and file_id in st.session_state.upload_cache:
            st.session_state.upload_cache[file_id]['status'] = 'added'

        # Set flag for generated images if applicable (check file_id prefix)
        if file_id.startswith("gen_"):
             st.session_state.generated_image_saved = True # Mark as saved

        st.toast(f"Image saved to '{label}' folder!", icon="✅")
    else:
        st.toast(f"Failed to save image to '{label}'.", icon="❌")

    # Rerun to update button states (e.g., "Added", "Saved") and potentially clear generated image section if desired
    # For generated images, setting the flag might be enough, rerun might clear the image display. Let's test.
    # A rerun is generally needed to reflect the 'added' status in the upload list correctly.
    # Let's keep the rerun for now.
    st.rerun()


def add_direct_to_library_callback(file_id, image_bytes, selected_category, filename):
    # (Callback remains largely the same)
    success, _ = save_image_to_category(image_bytes, selected_category, filename)
    if success:
        # Use a unique key to track added status for direct uploads
        st.session_state[f'direct_added_{file_id}'] = True
        st.toast(f"Image added to '{selected_category}' folder!", icon="⬆️")
        st.rerun() # Rerun to update the button state in the expander
    else:
        st.toast(f"Failed to add image directly to '{selected_category}'.", icon="❌")

def analyze_all_callback():
    # (Callback remains the same)
    if 'upload_cache' in st.session_state:
        triggered_count = 0
        for file_id, item_data in st.session_state.upload_cache.items():
            # Only trigger analysis if status is 'uploaded'
            if isinstance(item_data, dict) and item_data.get('status') == 'uploaded':
                st.session_state.upload_cache[file_id]['status'] = 'analyzing'
                triggered_count += 1
        if triggered_count > 0:
            st.toast(f"Triggered analysis for {triggered_count} thumbnail(s).", icon="🧠")
            # No rerun here needed, the main loop will see 'analyzing' status and handle it
        else:
            st.toast("No thumbnails awaiting analysis.", icon="🤷")


# ---------- Upload and Process Function ----------
def upload_and_process(client: OpenAI):
    # (Function logic is largely the same, uses updated analyze function)
    st.header("Upload & Analyze Thumbnails")
    st.info("Upload images, click '🧠 Analyze All Pending', then '✅ Add to Library' to save to the suggested category folder.")

    if 'upload_cache' not in st.session_state:
        st.session_state.upload_cache = {}

    uploaded_files = st.file_uploader(
        "Choose thumbnail images...",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    # Process newly uploaded files
    if uploaded_files:
        # new_files_added = False # Removed rerun based on this
        for uploaded_file in uploaded_files:
            # Create a unique ID based on name and size
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_id not in st.session_state.upload_cache:
                # new_files_added = True
                try:
                    image_bytes = uploaded_file.getvalue()
                    # Basic validation and conversion to JPEG bytes for consistency
                    display_image = Image.open(io.BytesIO(image_bytes))
                    display_image.verify() # Verify image data
                    # Re-open after verify
                    display_image = Image.open(io.BytesIO(image_bytes))

                    # Convert to RGB (for JPEG saving) and save to bytes buffer
                    img_byte_arr = io.BytesIO()
                    # Ensure image is converted to RGB before saving as JPEG
                    if display_image.mode == 'RGBA' or display_image.mode == 'P':
                         processed_image = display_image.convert('RGB')
                    else:
                         processed_image = display_image

                    processed_image.save(img_byte_arr, format='JPEG', quality=85) # Use quality 85
                    processed_image_bytes = img_byte_arr.getvalue()

                    st.session_state.upload_cache[file_id] = {
                        'name': uploaded_file.name,
                        'original_bytes': image_bytes,      # Keep original bytes for display
                        'processed_bytes': processed_image_bytes, # Use processed for analysis/saving
                        'label': None,
                        'reason': "Awaiting analysis",
                        'status': 'uploaded' # Initial status
                    }
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}. File skipped.")
                    # Add error status to cache to prevent re-processing attempts
                    st.session_state.upload_cache[file_id] = {
                        'status': 'error',
                        'error_msg': str(e),
                        'name': uploaded_file.name
                    }
        # Optional: Rerun immediately after processing uploads to show them instantly
        # if new_files_added: st.rerun()


    # Display and Process items from Cache
    if st.session_state.upload_cache:
        st.markdown("---")
        # Control Buttons
        col1, col2 = st.columns(2)
        with col1:
            # Check if any items have 'uploaded' status
            items_to_analyze = any(
                isinstance(item, dict) and item.get('status') == 'uploaded'
                for item in st.session_state.upload_cache.values()
            )
            analyze_all_disabled = not items_to_analyze or not client
            # Use the callback for 'Analyze All'
            st.button(
                "🧠 Analyze All Pending",
                key="analyze_all",
                on_click=analyze_all_callback,
                disabled=analyze_all_disabled,
                use_container_width=True,
                help="Requires OpenAI API Key" if not client else "Analyze all thumbnails not yet processed"
            )

        with col2:
            # Button to clear the cache
            if st.button("Clear Uploads and Analyses", key="clear_uploads", use_container_width=True):
                st.session_state.upload_cache = {}
                st.rerun() # Rerun to clear the display

        st.markdown("---")

        # Thumbnail Grid
        num_columns = 4 # Adjust number of columns
        cols = st.columns(num_columns)
        col_index = 0

        # Iterate over a copy of keys to allow modification during iteration (if needed, though callbacks handle it now)
        for file_id in list(st.session_state.upload_cache.keys()):
            item_data = st.session_state.upload_cache.get(file_id) # Use .get for safety

            # Skip if item_data is somehow invalid or removed
            if not isinstance(item_data, dict) or 'status' not in item_data:
                continue

            with cols[col_index % num_columns]:
                # Use st.container for grouping elements for each thumbnail
                with st.container(border=False): # Use border=False if using CSS border
                    st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
                    try:
                        if item_data['status'] == 'error':
                            st.error(f"Error with {item_data.get('name', 'Unknown File')}: {item_data.get('error_msg', 'Unknown error')}")
                        else:
                            # Display image using original bytes
                            st.image(
                                item_data['original_bytes'],
                                caption=f"{item_data.get('name', 'Unnamed Thumbnail')}",
                                use_container_width=True
                            )

                            analysis_placeholder = st.empty() # Placeholder for status/buttons

                            # Handle different statuses
                            if item_data['status'] == 'uploaded':
                                analysis_placeholder.info("Ready for analysis.")
                            elif item_data['status'] == 'analyzing':
                                # This block runs when status is set by the analyze_all_callback
                                with analysis_placeholder.container():
                                     with st.spinner(f"Analyzing {item_data['name']}..."):
                                        # Perform analysis - Use processed bytes
                                        label, reason = analyze_and_classify_thumbnail(client, item_data['processed_bytes'])
                                        # Update cache entry - MUTATING session state here
                                        if file_id in st.session_state.upload_cache: # Check if still exists
                                            st.session_state.upload_cache[file_id]['label'] = label
                                            st.session_state.upload_cache[file_id]['reason'] = reason # Store reason/status msg
                                            st.session_state.upload_cache[file_id]['status'] = 'analyzed'
                                            # Rerun needed to update display from 'analyzing' to 'analyzed' state
                                            st.rerun()
                            elif item_data['status'] in ['analyzed', 'added']:
                                # Display results and 'Add to Library' button
                                with analysis_placeholder.container():
                                    st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                                    label = item_data.get('label', 'Uncategorized')
                                    st.markdown(f"**Suggested:** `{label}`")

                                    is_added = (item_data['status'] == 'added')
                                    add_button_disabled = (
                                        is_added or
                                        label == "Uncategorized" or
                                        not label or
                                        label == "Other / Unclear" # Also disable for Other/Unclear
                                    )

                                    st.button(
                                        "✅ Add to Library" if not is_added else "✔️ Added",
                                        key=f'btn_add_{file_id}',
                                        on_click=add_to_library_callback,
                                        args=(file_id, item_data['processed_bytes'], label, item_data['name']),
                                        disabled=add_button_disabled,
                                        use_container_width=True, # Make button fill width
                                        help="Save this image to the suggested category folder." if not add_button_disabled else ("Image already added" if is_added else "Cannot add Uncategorized or Other/Unclear images")
                                    )
                                    st.markdown('</div>', unsafe_allow_html=True)

                    except KeyError as e:
                        st.error(f"Missing data for an item: {e}. Try re-uploading.")
                        if file_id in st.session_state.upload_cache: del st.session_state.upload_cache[file_id] # Clean up bad entry
                    except Exception as e:
                        st.error(f"Display error for {item_data.get('name', file_id)}: {e}")
                    finally:
                         st.markdown('</div>', unsafe_allow_html=True) # Close container div

            col_index += 1

    # Message if cache is empty and no files were uploaded in this run
    elif not uploaded_files:
        st.markdown("<p style='text-align: center; font-style: italic;'>Upload some thumbnails to get started!</p>", unsafe_allow_html=True)


# ---------- Function to create Zip File from Single Folder ----------
def create_zip_from_folder(category_name):
    # (Function remains the same)
    sanitized_category = sanitize_foldername(category_name)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_category
    zip_buffer = io.BytesIO()
    added_files = 0

    if not category_path.is_dir():
        return None # Category folder doesn't exist

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in category_path.iterdir():
            # Ensure it's a file, has an image extension, and not hidden
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'] and not item.name.startswith('.'):
                try:
                    # Add file to the root of the zip archive
                    zipf.write(item, arcname=item.name)
                    added_files += 1
                except Exception as zip_err:
                    st.warning(f"Zip error for {item.name} in {category_name}: {zip_err}")

    if added_files == 0:
        return None # No valid image files found in the folder

    zip_buffer.seek(0) # Rewind buffer to the beginning
    return zip_buffer


# ---------- Library Explorer ----------
def library_explorer():
    # (Function remains largely the same, displays folders, includes delete)
    st.header("Thumbnail Library Explorer")
    st.markdown("Browse saved thumbnails by category folder. Delete images or download category Zips.")

    # Initialize state variables if they don't exist
    if 'confirm_delete_path' not in st.session_state:
        st.session_state.confirm_delete_path = None
    if "selected_category_folder" not in st.session_state:
        st.session_state.selected_category_folder = None

    # Get current categories from filesystem
    categories = get_categories_from_folders()

    # --- Display Confirmation Dialog if an image is marked for deletion ---
    # This check MUST happen before the main explorer view renders
    if st.session_state.confirm_delete_path:
        display_delete_confirmation() # Call the confirmation dialog function
        # Stop further execution in this explorer view while confirming
        return # IMPORTANT: Prevent rest of the function from running

    # --- Main Explorer View ---
    if st.session_state.selected_category_folder is None:
        # Category Selection Grid
        st.markdown("### Select a Category Folder to View")
        if not categories:
            st.info("Your thumbnail library is currently empty. Add some images via the 'Upload & Analyze' tab or 'Generate Thumbnail' tab.")
            return

        cols_per_row = 5 # Adjust number of columns for categories
        num_categories = len(categories)
        num_rows = (num_categories + cols_per_row - 1) // cols_per_row

        for i in range(num_rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < num_categories:
                    cat_name = categories[idx]
                    # Display category name on the button
                    if cols[j].button(cat_name, key=f"btn_lib_{cat_name}", use_container_width=True):
                        st.session_state.selected_category_folder = cat_name
                        st.rerun() # Rerun to show the selected category's content

    else:
        # Display Selected Category Content
        selected_category = st.session_state.selected_category_folder
        st.markdown(f"### Category Folder: **{selected_category}**")

        # Top Bar: Back, Direct Upload, Download
        # Adjust column widths: Make back button smaller, expander larger
        top_cols = st.columns([0.2, 0.5, 0.3])

        with top_cols[0]: # Back Button
            if st.button("⬅️ Back", key="back_button", use_container_width=True, help="Go back to category list"):
                st.session_state.selected_category_folder = None
                st.session_state.confirm_delete_path = None # Ensure confirmation state is cleared when going back
                st.rerun()

        # Direct Upload Expander
        with top_cols[1]:
            with st.expander(f"⬆️ Add Image Directly to '{selected_category}'"):
                direct_uploaded_file = st.file_uploader(
                    f"Upload image for '{selected_category}'",
                    type=["jpg", "jpeg", "png", "webp"],
                    key=f"direct_upload_{selected_category}",
                    label_visibility="collapsed" # Hide label to save space
                )
                if direct_uploaded_file:
                    # Generate a unique ID for this upload instance
                    file_id = f"direct_{selected_category}_{direct_uploaded_file.name}_{direct_uploaded_file.size}"

                    # Initialize session state for tracking if added
                    if f'direct_added_{file_id}' not in st.session_state:
                        st.session_state[f'direct_added_{file_id}'] = False

                    is_added = st.session_state[f'direct_added_{file_id}']

                    # Display uploaded image preview
                    st.image(direct_uploaded_file, width=150)

                    try:
                        # Process the uploaded image (convert to JPEG bytes for consistency)
                        img_bytes_direct = direct_uploaded_file.getvalue()
                        img_direct = Image.open(io.BytesIO(img_bytes_direct))
                        img_direct.verify()
                        img_direct = Image.open(io.BytesIO(img_bytes_direct))

                        # Convert to RGB before saving as JPEG if needed
                        if img_direct.mode == 'RGBA' or img_direct.mode == 'P':
                            img_direct = img_direct.convert("RGB")

                        img_byte_arr_direct = io.BytesIO()
                        img_direct.save(img_byte_arr_direct, format='JPEG', quality=85)
                        processed_bytes_direct = img_byte_arr_direct.getvalue()

                        # Add button (disabled if already added)
                        st.button(
                            f"⬆️ Add This Image" if not is_added else "✔️ Added",
                            key=f"btn_direct_add_{file_id}",
                            on_click=add_direct_to_library_callback,
                            args=(file_id, processed_bytes_direct, selected_category, direct_uploaded_file.name),
                            disabled=is_added,
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Failed to process direct upload: {e}")

        # Get images for the selected category
        image_files = get_images_in_category(selected_category)

        # Download Button (only if there are images)
        if image_files:
            with top_cols[2]:
                zip_buffer = create_zip_from_folder(selected_category)
                if zip_buffer:
                    st.download_button(
                        label=f"⬇️ Download ({len(image_files)})",
                        data=zip_buffer,
                        file_name=f"{sanitize_foldername(selected_category)}_thumbnails.zip",
                        mime="application/zip",
                        key=f"download_{selected_category}",
                        use_container_width=True,
                        help=f"Download all images in '{selected_category}' as a zip file."
                    )
                else:
                    # Show disabled placeholder if zipping failed or folder empty after filtering
                     st.button(f"⬇️ Download ({len(image_files)})", disabled=True, use_container_width=True, help="No valid images to download.")


            # Thumbnail Display Grid (with Delete Button)
            st.markdown("---")
            cols_per_row_thumbs = 5 # More columns for thumbnails
            thumb_cols = st.columns(cols_per_row_thumbs)
            col_idx = 0
            for image_path in image_files:
                with thumb_cols[col_idx % cols_per_row_thumbs]:
                    with st.container(border=False):
                        # Use the db-thumbnail-container class from CSS
                        st.markdown('<div class="db-thumbnail-container">', unsafe_allow_html=True)
                        try:
                            image_path_str = str(image_path)
                            # Display the image
                            st.image(image_path_str, caption=f"{image_path.name}", use_container_width=True)

                            # Container for the delete button
                            st.markdown('<div class="delete-button-container">', unsafe_allow_html=True)
                            # Unique key for delete button using path and mtime
                            mtime = image_path.stat().st_mtime
                            del_key = f"del_{image_path.name}_{mtime}"

                            # Delete button - sets state to trigger confirmation
                            if st.button("🗑️", key=del_key, help="Delete this image"):
                                st.session_state.confirm_delete_path = image_path_str
                                st.rerun() # Rerun to show the confirmation dialog

                            st.markdown('</div>', unsafe_allow_html=True) # Close delete button container

                        except Exception as img_err:
                            st.warning(f"Could not load: {image_path.name} ({img_err})")
                        finally:
                            st.markdown('</div>', unsafe_allow_html=True) # Close db-thumbnail-container
                col_idx += 1

        # Message if category folder is empty (and no direct upload happened)
        # Only show if not currently interacting with direct upload file picker
        elif not direct_uploaded_file or (direct_uploaded_file and not st.session_state.get(f'direct_added_direct_{selected_category}_{direct_uploaded_file.name}_{direct_uploaded_file.size}', False)):
             st.info(f"No thumbnails found in the folder: '{selected_category}'. You can add images directly using the expander above.")


# ---------- Delete Confirmation Dialog Function ----------
def display_delete_confirmation():
    """ Displays the confirmation dialog for deleting an image. """
    # This function assumes it's only called when st.session_state.confirm_delete_path is set
    file_to_delete = st.session_state.confirm_delete_path
    if not file_to_delete: return # Should not happen based on calling logic, but safe check

    # Use a modal dialog or a prominent warning box
    # Using st.warning for simplicity
    with st.warning(f"**Confirm Deletion:** Are you sure you want to permanently delete `{os.path.basename(file_to_delete)}`?"):
        # Arrange buttons horizontally
        col1, col2, col3 = st.columns([1.5, 1, 5]) # Adjust spacing as needed
        with col1:
            if st.button("🔥 Confirm Delete", key="confirm_delete_yes", type="primary"):
                delete_success = delete_image_file(file_to_delete)
                st.session_state.confirm_delete_path = None # Clear the state regardless of success
                st.rerun() # Refresh the library view
        with col2:
             if st.button("🚫 Cancel", key="confirm_delete_cancel"):
                st.session_state.confirm_delete_path = None # Clear the state
                st.rerun() # Refresh the view to hide dialog


# ---------- NEW: Thumbnail Generation UI and Logic ----------
def thumbnail_generator_ui(client: OpenAI):
    """ Renders the UI for generating thumbnails and handles the process. """
    st.header("Generate Thumbnail with AI")
    st.markdown("Describe the thumbnail you want, select reference style categories (optional), and generate!")

    if not client:
        st.error("❌ OpenAI client not initialized. Please provide your API key in the sidebar.")
        return

    # --- Input Fields ---
    prompt_text = st.text_area(
        "**Thumbnail Prompt:**",
        height=150,
        key="generator_prompt_text", # Add key for potential state access
        placeholder="e.g., 'A futuristic cityscape at sunset with a glowing neon sign saying SUBSCRIBE', 'Close up of a golden retriever puppy looking curious', 'Split screen showing messy vs tidy desk'"
    )

    # Use descriptions from STANDARD_CATEGORIES_WITH_DESC for style hints
    category_map = {cat['name']: cat['description'] for cat in STANDARD_CATEGORIES_WITH_DESC}

    selected_style_categories = st.multiselect(
        "**Reference Style Categories (Optional):**",
        options=STANDARD_CATEGORIES, # Use the standard list as potential styles
        key="generator_style_categories",
        help="Select one or more categories to guide the visual style (e.g., 'Face-Focused', 'High Contrast'). The AI will try to incorporate these styles."
    )

    # --- Generation Button ---
    if st.button("✨ Generate Thumbnail", key="generate_thumb", disabled=not prompt_text):
        # Clear previous generation state before starting anew
        st.session_state.generated_image_b64 = None
        st.session_state.generation_prompt_used = None
        st.session_state.generated_image_saved = False

        with st.spinner("Generating thumbnail with DALL-E 3... This may take a moment."):
            try:
                # --- Construct the Full Prompt ---
                # Base prompt requesting a YouTube thumbnail format
                full_prompt = f"Create a hyper-realistic YouTube thumbnail image (aspect ratio 16:9) depicting: {prompt_text}."

                # Add style guidance if categories are selected
                if selected_style_categories:
                    style_hints = []
                    for cat_name in selected_style_categories:
                        if cat_name in category_map:
                             # Add specific descriptions as style guidance
                             style_hints.append(f"- Style emphasis: {category_map[cat_name]}")
                        else:
                             style_hints.append(f"- Style emphasis: {cat_name}") # Fallback to name if no desc found

                    if style_hints:
                         full_prompt += "\n\nIncorporate the following stylistic elements if relevant to the main subject:\n" + "\n".join(style_hints)
                else:
                    # Generic guidance if no styles selected
                    full_prompt += "\n\nUse a visually striking and engaging style suitable for a high-clickthrough YouTube thumbnail."

                # Add standard negative prompts or quality boosters if desired
                full_prompt += "\n\nEnsure the image is high quality, clear, and follows the main prompt accurately."


                st.caption(f"Sending prompt to DALL-E 3...") # Simplified caption
                print(f"DALL-E Prompt: {full_prompt}") # Print to console for debugging complex prompts

                # --- Call OpenAI Image Generation API (DALL-E 3) ---
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=full_prompt,
                    size="1792x1024", # Landscape, close to 16:9
                    quality="hd",     # Use "hd" for better quality/detail
                    n=1,              # Generate one image
                    response_format="b64_json" # Get base64 encoded image data directly
                )

                # --- Process Response ---
                if response.data and response.data[0].b64_json:
                    b64_data = response.data[0].b64_json
                    # Store generated data in session state to persist across reruns (for saving)
                    st.session_state.generated_image_b64 = b64_data
                    # Store the user's *original* prompt text for filename generation
                    st.session_state.generation_prompt_used = prompt_text
                    st.session_state.generated_image_saved = False # Reset saved flag for the new image

                    st.rerun() # Rerun to display the generated image and save options

                else:
                     st.error("❌ Image generation succeded but no image data received.")


            except Exception as e:
                st.error(f"❌ Thumbnail generation failed: {e}")
                if hasattr(e, 'response') and e.response:
                     st.error(f"API Response: {e.response.text}") # Show more details if available
                if "safety system" in str(e).lower():
                     st.warning("The prompt may have been blocked by the safety system. Please revise your prompt.")
                # Clear any potentially stale generated image data on failure
                st.session_state.generated_image_b64 = None
                st.session_state.generation_prompt_used = None


    # --- Display Generated Image and Save Options (if image exists in state) ---
    if 'generated_image_b64' in st.session_state and st.session_state.generated_image_b64:
        st.markdown("---")
        st.subheader("Generated Thumbnail")
        # Check save status *before* displaying controls that might trigger a rerun
        is_saved = st.session_state.get('generated_image_saved', False)

        st.markdown('<div class="generated-image-container">', unsafe_allow_html=True) # Add class for styling

        try:
            # Decode base64 image data
            generated_image_bytes = base64.b64decode(st.session_state.generated_image_b64)
            st.image(generated_image_bytes, caption="Generated Image Preview", use_container_width=True)

            st.markdown("---")
            st.markdown("**Save Generated Thumbnail to Library:**")

            # Allow user to choose which category to save to
            # Get current list of folders each time to ensure it's up-to-date
            current_folders = get_categories_from_folders()
            # Filter out generic/unwanted categories if needed
            savable_categories = [cat for cat in current_folders if cat not in ["uncategorized", "other_unclear"]]

            save_category = st.selectbox(
                "Choose destination category:",
                options=savable_categories, # Use existing folders as destinations
                key="save_gen_category_select",
                index=0 if savable_categories else None # Default to first if available
            )

            # Create a filename based on the prompt (sanitized)
            prompt_part_for_filename = re.sub(r'\W+', '_', st.session_state.get('generation_prompt_used', 'generated_image'))[:40].strip('_')
            if not prompt_part_for_filename: prompt_part_for_filename = "generated"
            # DALL-E b64 is typically PNG
            generated_filename = f"{prompt_part_for_filename}.png"

            # Unique file ID for the generated image instance (helps callback distinguish)
            gen_file_id = f"gen_{st.session_state.generation_prompt_used[:10]}_{int(time.time())}" # Simple timestamp + prompt hint ID


            save_button_disabled = not save_category or is_saved

            st.button(
                "✅ Save to Selected Category" if not is_saved else "✔️ Saved",
                key=f"btn_save_gen_{gen_file_id}",
                on_click=add_to_library_callback, # Reuse the same callback
                args=(
                    gen_file_id, # Pass the unique ID
                    generated_image_bytes, # Pass the raw bytes
                    save_category,
                    generated_filename
                ),
                disabled=save_button_disabled,
                use_container_width=True, # Make button fill width
                help="Save this generated image to the chosen category folder." if not save_button_disabled else ("Image already saved" if is_saved else "Please select a valid category")
            )

        except Exception as display_err:
            st.error(f"Error displaying or preparing generated image for saving: {display_err}")
            # Clean up state if display fails badly
            st.session_state.generated_image_b64 = None
            st.session_state.generation_prompt_used = None
            st.session_state.generated_image_saved = False
        finally:
             st.markdown('</div>', unsafe_allow_html=True) # Close generated-image-container


# ---------- Main App ----------
def main():
    ensure_library_dir()
    # Create predefined folders on startup using the names list
    create_predefined_category_folders(STANDARD_CATEGORIES)

    # Initialize Session State Keys robustly if they don't exist
    # This prevents errors if the app restarts or keys are cleared unexpectedly
    keys_to_init = {
        'selected_category_folder': None,
        'upload_cache': {},
        'confirm_delete_path': None,
        'generated_image_b64': None,
        'generation_prompt_used': None,
        'generated_image_saved': False
    }
    for key, default_value in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


    # --- Sidebar Setup ---
    with st.sidebar:
        # Simple title/logo placeholder
        st.markdown('<div style="text-align: center; padding: 10px;">🖼️</div>', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center;">Thumbnail Toolkit</h2>', unsafe_allow_html=True)
        st.markdown("Analyze, organize, and generate YouTube thumbnails.", unsafe_allow_html=True)
        st.markdown("---")

        # OpenAI Client Setup (Crucial for Analysis & Generation)
        client = setup_openai_client()
        if not client:
             st.warning("OpenAI API key needed for Analysis and Generation features.", icon="🔑")

        # Navigation Menu
        menu_options = ["Upload & Analyze", "Generate Thumbnail", "Library Explorer"]
        # Check if 'nav_menu' exists and is valid, otherwise default
        current_selection = st.session_state.get("nav_menu", menu_options[0])
        if current_selection not in menu_options:
            current_selection = menu_options[0] # Default to first option if invalid state

        menu = st.radio(
            "Navigation",
            menu_options,
            key="nav_menu",
            index=menu_options.index(current_selection), # Set index based on current state
            label_visibility="collapsed" # Hide the label "Navigation"
            )
        st.markdown("---")

        # Library Info & Download All
        st.info(f"Library stored in:\n`./{LIBRARY_DIR}`")
        try:
            zip_buffer, file_count = create_zip_of_library()
            if zip_buffer and file_count > 0:
                st.download_button(
                    label=f"⬇️ Download All ({file_count} images)",
                    data=zip_buffer,
                    file_name="thumbnail_library_archive.zip",
                    mime="application/zip",
                    key="download_all_library",
                    use_container_width=True,
                    help="Download all categories and images as a single zip file."
                )
            else:
                st.button("⬇️ Download All (Empty)", disabled=True, use_container_width=True, help="Library is empty or contains no images.")
        except Exception as zip_all_err:
             st.error(f"Error creating library zip: {zip_all_err}")
             st.button("⬇️ Download All (Error)", disabled=True, use_container_width=True)


        st.markdown("---")

        # Display standard categories list in sidebar
        with st.expander("View Standard Categories & Descriptions"):
            for cat in STANDARD_CATEGORIES_WITH_DESC:
                 # Don't list the fallback category explicitly here unless needed
                 if cat['name'] != "Other / Unclear":
                    st.markdown(f"**{cat['name']}**: _{cat['description']}_")


    # --- Main Content Area ---

    # Handle delete confirmation dialog first if active.
    # This prevents other UI elements from rendering and interfering.
    if st.session_state.confirm_delete_path:
        display_delete_confirmation()
    else:
        # Render the selected page based on the menu
        if menu == "Upload & Analyze":
            upload_and_process(client) # Pass the client object
        elif menu == "Generate Thumbnail":
            thumbnail_generator_ui(client) # Pass the client object
        elif menu == "Library Explorer":
            library_explorer()


if __name__ == "__main__":
    main()
