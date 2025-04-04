import streamlit as st
import os
import io
import base64
import zipfile
from datetime import datetime, date # Added date
from PIL import Image
from openai import OpenAI
import requests # Added for downloading thumbnails
import yt_dlp # Added for fetching youtube data
import re
import pathlib
import time

# --- Configuration ---
LIBRARY_DIR = "thumbnail_library"
LABEL_DELIMITER = ";;" # Keep for potential future multi-label, though currently single

# --- Standard Category Definitions ---
# (Using the expanded list)
STANDARD_CATEGORIES = [
    "Text-Dominant","Minimalist / Clean","Face-Focused","Before & After",
    "Comparison / Versus","Collage / Multi-Image","Image-Focused","Branded",
    "Curiosity Gap / Intrigue","High Contrast","Gradient Background",
    "Bordered / Framed","Inset / PiP","Arrow/Circle Emphasis","Icon-Driven",
    "Retro / Vintage","Hand-Drawn / Sketch","Textured Background",
    "Extreme Close-Up (Object)","Other / Unclear"
]

# Set page configuration
st.set_page_config(
    page_title="Thumbnail Analyzer Pro",
    page_icon="✨",
    layout="wide"
)

# ---------- Custom CSS ----------
# (Keep existing CSS - maybe adjust button colors slightly if desired)
st.markdown("""
<style>
    /* Existing CSS */
    .main { background-color: #0f0f0f; color: #f1f1f1; }
    /* ... other styles ... */
    button:has(span:contains("Fetch Thumbnails")) { background-color: #fd7e14; border: none;} /* Orange */
    button:has(span:contains("Analyze Selected")) { background-color: #20c997; border: none;} /* Teal */
    .thumbnail-grid-item { /* Style for fetched thumbnails */
        border: 1px solid #333; border-radius: 4px; padding: 8px; margin-bottom: 8px;
        background-color: #1e1e1e; text-align: center;
    }
    .thumbnail-grid-item img { max-width: 100%; height: auto; border-radius: 4px; margin-bottom: 5px;}
    .thumbnail-grid-item .stCheckbox { justify-content: center; } /* Center checkbox */
</style>
""", unsafe_allow_html=True)


# ---------- Filesystem Library Functions ----------
# (sanitize_foldername, ensure_library_dir, save_image_to_category - single label,
#  get_categories_from_folders, get_images_in_category, delete_image_file,
#  create_zip_from_folder remain largely the same as the previous version)

def sanitize_foldername(name):
    name = name.strip(); name = re.sub(r'[<>:"/\\|?*.,;]+', '_', name); name = re.sub(r'_+', '_', name)
    if name.upper() in ["CON", "PRN", "AUX", "NUL"] or re.match(r"^(COM|LPT)[1-9]$", name.upper()): name = f"_{name}_"
    return name if name else "uncategorized"

def ensure_library_dir():
    pathlib.Path(LIBRARY_DIR).mkdir(parents=True, exist_ok=True)

def create_predefined_category_folders(category_list):
    ensure_library_dir() # Ensure base exists first
    # st.sidebar.write("Ensuring standard category folders...") # Reduce verbosity
    for category_name in category_list:
        sanitized_name = sanitize_foldername(category_name)
        if not sanitized_name or sanitized_name in ["uncategorized", "other_unclear"]: continue
        folder_path = pathlib.Path(LIBRARY_DIR) / sanitized_name
        try: folder_path.mkdir(parents=True, exist_ok=True)
        except Exception: pass # Ignore errors if folder creation fails silently

def save_image_to_category(image_bytes, label, original_filename="thumbnail"):
    ensure_library_dir()
    if not label or label in ["Uncategorized", "Other / Unclear"]: return False, None
    base_filename, _ = os.path.splitext(original_filename)
    base_filename_sanitized = re.sub(r'[^\w\-]+', '_', base_filename).strip('_')[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    sanitized_label = sanitize_foldername(label)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_label
    category_path.mkdir(parents=True, exist_ok=True)
    filename = f"{base_filename_sanitized}_{timestamp}.jpg"; filepath = category_path / filename
    counter = 1
    while filepath.exists(): filename = f"{base_filename_sanitized}_{timestamp}_{counter}.jpg"; filepath = category_path / filename; counter += 1
    try:
        with open(filepath, "wb") as f: f.write(image_bytes)
        return True, str(filepath)
    except Exception as e: st.error(f"Error saving to '{filepath}': {e}"); return False, None

def get_categories_from_folders():
    ensure_library_dir()
    try: return sorted([d.name for d in pathlib.Path(LIBRARY_DIR).iterdir() if d.is_dir() and not d.name.startswith('.')])
    except FileNotFoundError: return []

def get_images_in_category(category_name):
    sanitized_category = sanitize_foldername(category_name)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_category
    image_files = []
    if category_path.is_dir():
        for item in category_path.iterdir():
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'] and not item.name.startswith('.'):
                image_files.append(item)
    return sorted(image_files, key=lambda p: p.stat().st_mtime, reverse=True) # Sort by mod time

def delete_image_file(image_path_str):
    try:
        file_path = pathlib.Path(image_path_str)
        if file_path.is_file(): file_path.unlink(); st.toast(f"Deleted: {file_path.name}", icon="🗑️"); return True
        else: st.error(f"File not found for deletion: {file_path.name}"); return False
    except Exception as e: st.error(f"Error deleting {image_path_str}: {e}"); return False

def create_zip_from_folder(category_name):
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

# ---------- OpenAI API Setup (Using Secrets) ----------
def setup_openai_client():
    """Initializes and returns the OpenAI client using st.secrets."""
    api_key = None
    # Prioritize Streamlit secrets
    if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        # Fallback for local development (if secrets file not present)
        st.sidebar.warning("OpenAI API Key not found in Streamlit Secrets. Using fallback input.", icon="⚠️")
        api_key = st.sidebar.text_input("Enter OpenAI API key:", type="password", key="api_key_input_sidebar")

    if not api_key:
        st.sidebar.error("OpenAI API key is required.")
        return None

    try:
        client = OpenAI(api_key=api_key)
        # Optional: Test connection - client.models.list()
        return client
    except Exception as e:
        st.sidebar.error(f"Error initializing OpenAI client: {e}")
        return None

# ---------- Utility Function ----------
# (encode_image remains the same)
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# ---------- YouTube Fetching Function ----------
def fetch_channel_videos(channel_url, max_to_scan=50):
    """Fetches video data using yt-dlp."""
    st.write(f"Attempting to fetch up to {max_to_scan} latest videos from channel...")
    ydl_opts = {
        'extract_flat': True, # Don't download video, just get info
        'playlistend': max_to_scan, # Limit number of videos scanned
        'quiet': True,
        'ignoreerrors': True, # Skip videos that cause errors
    }
    videos = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # yt-dlp needs URL, works with channel URLs directly
            info_dict = ydl.extract_info(channel_url, download=False)

            if info_dict and 'entries' in info_dict:
                for entry in info_dict['entries']:
                    if entry and entry.get('ie_key') == 'Youtube': # Ensure it's a YT video
                        upload_date_str = entry.get('upload_date') # YYYYMMDD format
                        upload_date_obj = None
                        if upload_date_str:
                            try:
                                upload_date_obj = datetime.strptime(upload_date_str, '%Y%m%d').date()
                            except ValueError:
                                upload_date_obj = None # Handle parsing errors

                        # Try to get the best thumbnail
                        thumbnail_url = entry.get('thumbnail') # Default thumbnail
                        thumbnails = entry.get('thumbnails', [])
                        if thumbnails:
                             # Prefer maxres, then standard, high, medium, default
                            best_thumb = thumbnails[-1] # Start with lowest res as fallback
                            for t in thumbnails:
                                if t.get('id') == 'maxres': best_thumb = t; break
                                if t.get('id') == 'sd': best_thumb = t
                                elif t.get('id') == 'hq' and best_thumb.get('id') != 'sd': best_thumb = t
                                elif t.get('id') == 'mq' and best_thumb.get('id') not in ['sd', 'hq']: best_thumb = t
                            thumbnail_url = best_thumb.get('url', thumbnail_url)


                        videos.append({
                            'video_id': entry.get('id'),
                            'title': entry.get('title', 'N/A'),
                            'thumbnail_url': thumbnail_url,
                            'upload_date': upload_date_obj,
                            'webpage_url': entry.get('url') # Added for context if needed
                        })
            else:
                 st.warning("Could not extract video entries. Is the URL correct and a valid channel/playlist?")

    except yt_dlp.utils.DownloadError as e:
        st.error(f"yt-dlp Error: Could not process URL. Is it a valid YouTube channel URL? Details: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during fetching: {e}")

    st.write(f"Found {len(videos)} videos initially.")
    return videos

# ---------- OpenAI Analysis Function (Single Label) ----------
def analyze_and_classify_thumbnail(client: OpenAI, image_bytes: bytes):
    """ Analyzes thumbnail for the single most relevant label. """
    # (Function remains the same as previous version - requests single label)
    if not client: return "Uncategorized", "OpenAI client not initialized."
    base64_image = encode_image(image_bytes); image_data_uri = f"data:image/jpeg;base64,{base64_image}"
    category_definitions_list = [ "Text-Dominant: ...", "Minimalist / Clean: ...", # Keep descriptions short
                                  "Face-Focused: ...", "Before & After: ...", "Comparison / Versus: ...",
                                  "Collage / Multi-Image: ...", "Image-Focused: ...", "Branded: ...",
                                  "Curiosity Gap / Intrigue: ...", "High Contrast: ...", "Gradient Background: ...",
                                  "Bordered / Framed: ...", "Inset / PiP: ...", "Arrow/Circle Emphasis: ...",
                                  "Icon-Driven: ...", "Retro / Vintage: ...", "Hand-Drawn / Sketch: ...",
                                  "Textured Background: ...", "Extreme Close-Up (Object): ...", "Other / Unclear: ..."]
    category_definitions_text = "\n".join([f"- {cat.split(':')[0]}" for cat in category_definitions_list]) # Just list names for prompt
    valid_categories = set(STANDARD_CATEGORIES)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[ { "role": "system", "content": f"Expert analyst: Identify the SINGLE most relevant category for the thumbnail from the list provided. Output ONLY the category name." },
                       { "role": "user", "content": [ { "type": "text", "text": f"Classify using ONLY these categories: {', '.join(valid_categories)}. Output the single best category name." },
                                                       { "type": "image_url", "image_url": {"url": image_data_uri, "detail": "low"} } ] } ],
            temperature=0.1, max_tokens=40
        )
        result = response.choices[0].message.content.strip()
    except Exception as e: st.error(f"OpenAI Error: {e}"); return "Uncategorized", "API Error."
    # Validation
    label = "Uncategorized"; reason = "Reason not stored."
    if result:
        found = False
        for valid_cat in valid_categories:
            if valid_cat.strip().lower() == result.strip().lower(): label = valid_cat; found = True; break
        if not found: label = "Other / Unclear"; st.warning(f"AI suggested unknown category: '{result}'. Using 'Other / Unclear'.")
    else: label = "Uncategorized"; st.warning("AI returned empty response.")
    return label, reason

# ---------- Callbacks ----------
# (add_to_library_callback remains the same - takes single label)
def add_to_library_callback(item_key, image_bytes, label, filename, source='upload'):
    success, _ = save_image_to_category(image_bytes, label, filename)
    cache_key = 'upload_items' if source == 'upload' else 'fetch_items'
    if success:
        if cache_key in st.session_state and item_key in st.session_state[cache_key]:
            st.session_state[cache_key][item_key]['status'] = 'added'
            st.toast(f"Saved to '{label}' folder!", icon="✅")
        else: st.warning(f"Cache status update failed for {filename}. File likely saved."); st.toast(f"Saved (cache status update failed).", icon="✅")
    else: st.toast(f"Failed to save thumbnail.", icon="❌")

# (add_direct_to_library_callback remains the same)
def add_direct_to_library_callback(file_id, image_bytes, selected_category, filename):
    success, _ = save_image_to_category(image_bytes, selected_category, filename)
    if success: st.session_state[f'direct_added_{file_id}'] = True; st.toast(f"Image added to '{selected_category}'!", icon="⬆️")
    else: st.toast(f"Failed to add image directly.", icon="❌")

# (analyze_all_callback - now specific to upload items)
def analyze_all_uploads_callback():
    """Sets status to 'analyzing' for all 'uploaded' items in upload_items cache."""
    if 'upload_items' in st.session_state:
        triggered_count = 0
        for item_id, item_data in st.session_state.upload_items.items():
            if item_data.get('status') == 'uploaded':
                st.session_state.upload_items[item_id]['status'] = 'analyzing'
                triggered_count += 1
        if triggered_count > 0: st.toast(f"Triggered analysis for {triggered_count} uploaded thumbnail(s).", icon="🧠")
        else: st.toast("No uploaded thumbnails awaiting analysis.", icon="🤷")

# Callback for Analyze Selected fetched items
def analyze_selected_callback():
    """Downloads and prepares selected fetched thumbnails for analysis."""
    if 'selected_thumbnails' in st.session_state and st.session_state.selected_thumbnails:
        if 'fetch_items' not in st.session_state: st.session_state.fetch_items = {}
        if 'fetched_thumbnails' not in st.session_state: st.session_state.fetched_thumbnails = []

        triggered_count = 0
        st.write("Preparing selected thumbnails for analysis...") # User feedback
        progress_bar = st.progress(0)
        total_selected = len(st.session_state.selected_thumbnails)

        for i, video_id in enumerate(list(st.session_state.selected_thumbnails)): # Iterate copy
             # Check if already processed or being processed
             if video_id in st.session_state.fetch_items and st.session_state.fetch_items[video_id]['status'] not in ['selected', 'error_download']:
                 st.write(f"Skipping {video_id} (already processed/analyzing).")
                 continue

             # Find thumbnail URL from fetched data
             video_data = next((item for item in st.session_state.fetched_thumbnails if item['video_id'] == video_id), None)
             if not video_data or not video_data.get('thumbnail_url'):
                 st.warning(f"Could not find data or thumbnail URL for video ID: {video_id}")
                 if video_id in st.session_state.fetch_items: st.session_state.fetch_items[video_id]['status'] = 'error_download'
                 continue

             # Download thumbnail
             try:
                 response = requests.get(video_data['thumbnail_url'], stream=True, timeout=10)
                 response.raise_for_status() # Raise error for bad responses (4xx or 5xx)
                 image_bytes = response.content

                 # Process image (convert to JPEG)
                 img = Image.open(io.BytesIO(image_bytes)); img.verify()
                 img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                 img_byte_arr = io.BytesIO(); img.save(img_byte_arr, format='JPEG', quality=85)
                 processed_bytes = img_byte_arr.getvalue()

                 # Add to fetch_items cache for processing
                 st.session_state.fetch_items[video_id] = {
                     'name': video_data.get('title', video_id),
                     'original_bytes': image_bytes, # Keep original for display? Or just processed? Let's keep processed only for simplicity here.
                     'processed_bytes': processed_bytes,
                     'label': None,
                     'reason': "Reason not stored.",
                     'status': 'analyzing', # Ready for analysis loop
                     'source': 'fetch'
                 }
                 triggered_count += 1
             except requests.exceptions.RequestException as e:
                 st.error(f"Error downloading thumbnail for {video_id}: {e}")
                 if video_id not in st.session_state.fetch_items: st.session_state.fetch_items[video_id] = {}
                 st.session_state.fetch_items[video_id]['status'] = 'error_download'
             except Exception as e:
                 st.error(f"Error processing image for {video_id}: {e}")
                 if video_id not in st.session_state.fetch_items: st.session_state.fetch_items[video_id] = {}
                 st.session_state.fetch_items[video_id]['status'] = 'error_processing'

             progress_bar.progress((i + 1) / total_selected)

        if triggered_count > 0: st.toast(f"Added {triggered_count} selected thumbnail(s) to analysis queue.", icon="👍")
        else: st.toast("No new thumbnails added to queue (already processed or errors).", icon="🤷")
        # Clear selection after adding to queue? Optional.
        # st.session_state.selected_thumbnails = set()

    else:
        st.toast("No thumbnails selected for analysis.", icon="🤔")


# ---------- Display and Process Analysis Items ----------
def display_analysis_items(client: OpenAI):
    """Displays items from upload_items and fetch_items caches."""
    st.subheader("Analysis Queue & Results")
    items_to_display = []
    if 'upload_items' in st.session_state:
        items_to_display.extend([(k, v) for k,v in st.session_state.upload_items.items()])
    if 'fetch_items' in st.session_state:
        items_to_display.extend([(k, v) for k,v in st.session_state.fetch_items.items()])

    if not items_to_display:
        st.caption("Upload files or fetch from a channel and select thumbnails to analyze.")
        return

    num_columns = 3
    cols = st.columns(num_columns)
    col_index = 0

    for item_id, item_data in items_to_display:
        if not isinstance(item_data, dict) or 'status' not in item_data: continue # Skip bad data

        source = item_data.get('source', 'upload') # Default to upload if source not set
        cache_key_name = 'upload_items' if source == 'upload' else 'fetch_items'

        with cols[col_index % num_columns]:
            with st.container():
                st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
                try:
                    # Use processed bytes for display consistency if original isn't stored
                    display_bytes = item_data.get('original_bytes', item_data.get('processed_bytes'))
                    if display_bytes:
                        display_image = Image.open(io.BytesIO(display_bytes))
                        st.image(display_image, caption=f"{item_data['name']}", use_container_width=True)
                    else:
                        st.error("Image data missing.")

                    analysis_placeholder = st.empty()

                    # Status handling logic
                    if item_data['status'] == 'uploaded': # Only for direct uploads
                        analysis_placeholder.info("Ready for analysis (Click 'Analyze All Uploads').")

                    elif item_data['status'] == 'analyzing':
                        with analysis_placeholder.container():
                            with st.spinner(f"Analyzing {item_data['name']}..."):
                                label, reason = analyze_and_classify_thumbnail(client, item_data['processed_bytes'])
                                # Update the correct cache
                                st.session_state[cache_key_name][item_id]['label'] = label
                                st.session_state[cache_key_name][item_id]['reason'] = reason
                                st.session_state[cache_key_name][item_id]['status'] = 'analyzed'
                                st.rerun()

                    elif item_data['status'] in ['analyzed', 'added']:
                         with analysis_placeholder.container():
                             st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                             label = item_data.get('label', 'Uncategorized')
                             st.markdown(f"**Suggested:** `{label}`")
                             is_added = (item_data['status'] == 'added')
                             st.button("✅ Add to Library" if not is_added else "✔️ Added",
                                       key=f'btn_add_{item_id}',
                                       on_click=add_to_library_callback,
                                       args=(item_id, item_data['processed_bytes'], label, item_data['name'], source), # Pass source
                                       disabled=is_added or label == "Uncategorized" or not label)
                             st.markdown('</div>', unsafe_allow_html=True)

                    elif item_data['status'].startswith('error'):
                         analysis_placeholder.error(f"Error: {item_data.get('error_msg', 'Unknown processing error')}")

                except Exception as e: st.error(f"Display error for {item_data.get('name', item_id)}: {e}")
                finally: st.markdown('</div>', unsafe_allow_html=True) # Close thumbnail-container
        col_index += 1


# ---------- Library Explorer ----------
# (library_explorer remains mostly the same, uses display_delete_confirmation)
def library_explorer():
    # ... (Code remains the same as previous version) ...
    st.header("Thumbnail Library Explorer")
    # ... (rest of the function including category selection, direct upload, zip, delete confirmation call) ...


# ---------- Delete Confirmation Dialog Function ----------
# (display_delete_confirmation remains the same)
def display_delete_confirmation():
     if 'confirm_delete_path' in st.session_state and st.session_state.confirm_delete_path:
        # ... (confirmation logic is the same) ...


# ---------- Main App ----------
def main():
    ensure_library_dir()
    create_predefined_category_folders(STANDARD_CATEGORIES)

    # Initialize Session State Keys
    if 'selected_category_folder' not in st.session_state: st.session_state.selected_category_folder = None
    # Use separate caches for uploaded vs fetched items
    if 'upload_items' not in st.session_state: st.session_state.upload_items = {}
    if 'fetch_items' not in st.session_state: st.session_state.fetch_items = {}
    if 'fetched_thumbnails' not in st.session_state: st.session_state.fetched_thumbnails = []
    if 'selected_thumbnails' not in st.session_state: st.session_state.selected_thumbnails = set()
    if 'confirm_delete_path' not in st.session_state: st.session_state.confirm_delete_path = None

    # --- Sidebar Setup ---
    with st.sidebar:
        st.markdown( # Placeholder for title HTML
            '<div><span style="color: #FF0000; font-size: 28px;">▶️</span><h1 style="display:inline; color:#f1f1f1;">Thumbnail Analyzer Pro</h1></div>',
             unsafe_allow_html=True)
        st.markdown('<p style="color:#aaaaaa;">Analyze, Fetch & Organize Thumbnails</p>', unsafe_allow_html=True)
        st.markdown("---")
        client = setup_openai_client()
        menu = st.radio("Navigation", ["Analyze Thumbnails", "Library Explorer"], key="nav_menu", label_visibility="collapsed")
        st.markdown("---")
        st.info(f"Library stored in './{LIBRARY_DIR}'")
        with st.expander("Standard Categories"):
            st.caption("\n".join(f"- {cat}" for cat in STANDARD_CATEGORIES if cat != "Other / Unclear"))
        st.caption("Uses OpenAI & yt-dlp.")


    # --- Main Content Area ---
    if st.session_state.confirm_delete_path:
         display_delete_confirmation() # Show confirmation dialog if active
    else:
         if menu == "Analyze Thumbnails":
             if not client:
                 st.error("❌ OpenAI client not initialized. Please provide API key in sidebar or secrets.")
             else:
                tab1, tab2 = st.tabs(["⬆️ Upload Files", "📡 Fetch from Channel URL"])

                with tab1:
                    # === Upload Files Tab ===
                    st.header("Upload Files")
                    st.info("Upload images, click '🧠 Analyze All Uploads', then '✅ Add to Library' to save.")
                    # File Uploader Logic (populates upload_items cache)
                    uploaded_files = st.file_uploader( "Choose thumbnail images...", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key="file_uploader_tab1")
                    if uploaded_files:
                        for uploaded_file in uploaded_files:
                            file_id = f"{uploaded_file.name}_{uploaded_file.size}" # Unique ID for uploads
                            if file_id not in st.session_state.upload_items:
                                try:
                                    # Process and add to upload_items cache
                                    image_bytes = uploaded_file.getvalue(); img = Image.open(io.BytesIO(image_bytes)); img.verify(); img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                                    img_byte_arr = io.BytesIO(); img.save(img_byte_arr, format='JPEG', quality=85); processed_bytes = img_byte_arr.getvalue()
                                    st.session_state.upload_items[file_id] = { 'name': uploaded_file.name, 'original_bytes': image_bytes, 'processed_bytes': processed_bytes, 'label': None, 'reason': "N/A", 'status': 'uploaded', 'source': 'upload'}
                                except Exception as e: st.error(f"Error reading {uploaded_file.name}: {e}"); st.session_state.upload_items[file_id] = {'status': 'error', 'error_msg': str(e), 'name': uploaded_file.name, 'source': 'upload'}

                    # Analyze All button specific to uploads
                    items_to_analyze_upload = any(item.get('status') == 'uploaded' for item in st.session_state.upload_items.values())
                    if st.button("🧠 Analyze All Uploads", key="analyze_all_uploads", on_click=analyze_all_uploads_callback, disabled=not items_to_analyze_upload): st.rerun()
                    if st.button("Clear Uploads", key="clear_uploads_tab1"): st.session_state.upload_items = {}; st.rerun()


                with tab2:
                    # === Fetch from Channel Tab ===
                    st.header("Fetch from Channel URL")
                    channel_url = st.text_input("Enter YouTube Channel URL:", key="channel_url_input")

                    # Date Filtering Inputs
                    st.markdown("**Filter by Upload Date (Optional):**")
                    col_d1, col_d2 = st.columns(2)
                    today = date.today()
                    one_month_ago = today.replace(month=today.month-1) if today.month > 1 else today.replace(year=today.year-1, month=12)
                    start_date = col_d1.date_input("Start Date", value=None, key="start_date") # Default to None (all time)
                    end_date = col_d2.date_input("End Date", value=None, key="end_date") # Default to None (all time)

                    if st.button("Fetch Thumbnails", key="fetch_button", disabled=not channel_url):
                        with st.spinner("Fetching video list... (may take a moment)"):
                             # Fetch ~50 initially, then filter
                            videos = fetch_channel_videos(channel_url, max_to_scan=50)
                            st.session_state.fetched_thumbnails = videos
                            st.session_state.selected_thumbnails = set() # Clear previous selection
                            st.rerun() # Rerun to display results

                    # Display Fetched Thumbnails & Selection
                    if 'fetched_thumbnails' in st.session_state and st.session_state.fetched_thumbnails:
                         st.markdown("---")
                         st.subheader("Fetched Thumbnails")

                         # Apply Filters
                         filtered_videos = st.session_state.fetched_thumbnails
                         if start_date:
                             filtered_videos = [v for v in filtered_videos if v['upload_date'] and v['upload_date'] >= start_date]
                         if end_date:
                             filtered_videos = [v for v in filtered_videos if v['upload_date'] and v['upload_date'] <= end_date]

                         # Apply Max 25 Limit *after* filtering
                         display_videos = filtered_videos[:25]
                         st.caption(f"Displaying {len(display_videos)} thumbnails (filtered from {len(filtered_videos)}, scanned up to 50).")

                         if display_videos:
                              # Select All / Deselect All Checkbox
                              select_all = st.checkbox("Select/Deselect All Displayed", key="select_all_fetched")
                              if select_all:
                                   st.session_state.selected_thumbnails.update(v['video_id'] for v in display_videos)
                              elif not select_all and st.session_state.get('prev_select_all', False): # If it was just unchecked
                                    # This naive approach might clear selections made while select_all was checked.
                                    # A more complex diff logic would be needed for perfect behavior.
                                    st.session_state.selected_thumbnails.clear()
                              st.session_state.prev_select_all = select_all # Remember state for next run


                              # Grid Display with Checkboxes
                              num_cols_fetch = 4
                              fetch_cols = st.columns(num_cols_fetch)
                              for i, video in enumerate(display_videos):
                                   with fetch_cols[i % num_cols_fetch]:
                                       st.markdown('<div class="thumbnail-grid-item">', unsafe_allow_html=True)
                                       thumb_url = video.get('thumbnail_url')
                                       if thumb_url: st.image(thumb_url, caption=video.get('title', video['video_id'])[:50] + "...", use_container_width=True)
                                       else: st.caption("No thumbnail")

                                       # Checkbox for selection
                                       is_selected = st.checkbox("Select", value=(video['video_id'] in st.session_state.selected_thumbnails), key=f"select_{video['video_id']}")
                                       if is_selected and video['video_id'] not in st.session_state.selected_thumbnails:
                                           st.session_state.selected_thumbnails.add(video['video_id'])
                                       elif not is_selected and video['video_id'] in st.session_state.selected_thumbnails:
                                           st.session_state.selected_thumbnails.discard(video['video_id'])

                                       st.markdown('</div>', unsafe_allow_html=True)

                              # Analyze Selected Button
                              st.markdown("---")
                              analyze_selected_disabled = not st.session_state.selected_thumbnails or not client
                              if st.button("✨ Analyze Selected Thumbnails", key="analyze_selected", on_click=analyze_selected_callback, disabled=analyze_selected_disabled, use_container_width=True):
                                   st.rerun() # Rerun to start processing items added to fetch_items cache

                         else:
                              st.info("No thumbnails match the selected date range.")
                    elif 'fetched_thumbnails' in st.session_state: # If fetch was attempted but no results
                         st.info("No videos found or fetched for the given URL.")


                # --- Common Analysis Display Area ---
                st.markdown("---")
                # This function now displays items from BOTH caches (upload_items and fetch_items)
                display_analysis_items(client)


         elif menu == "Library Explorer":
             library_explorer()

if __name__ == "__main__":
    main()
