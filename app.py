import streamlit as st
import os
import io
import base64
import zipfile
from datetime import datetime, date, timezone # Added timezone
from PIL import Image
from openai import OpenAI
import requests
# Removed: import yt_dlp
# Added: Google API Client libraries
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re
import pathlib
import time

# --- Configuration ---
LIBRARY_DIR = "thumbnail_library"
# (STANDARD_CATEGORIES list remains the same as the last version)
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
    page_title="Thumbnail Analyzer Pro (API)",
    page_icon="✨",
    layout="wide"
)

# ---------- Custom CSS ----------
# (CSS remains the same)
st.markdown("""<style> /* ... Existing CSS ... */ </style>""", unsafe_allow_html=True)

# ---------- Filesystem Library Functions ----------
# (sanitize_foldername, ensure_library_dir, create_predefined_category_folders,
#  save_image_to_category - single label, get_categories_from_folders,
#  get_images_in_category, delete_image_file, create_zip_from_folder
#  remain the same as the previous version)
def sanitize_foldername(name):
    name = name.strip(); name = re.sub(r'[<>:"/\\|?*.,;]+', '_', name); name = re.sub(r'_+', '_', name)
    if name.upper() in ["CON", "PRN", "AUX", "NUL"] or re.match(r"^(COM|LPT)[1-9]$", name.upper()): name = f"_{name}_"
    return name if name else "uncategorized"
def ensure_library_dir(): pathlib.Path(LIBRARY_DIR).mkdir(parents=True, exist_ok=True)
def create_predefined_category_folders(category_list):
    ensure_library_dir(); # st.sidebar.write("Checking folders...")
    for category_name in category_list:
        sanitized_name = sanitize_foldername(category_name)
        if not sanitized_name or sanitized_name in ["uncategorized", "other_unclear"]: continue
        folder_path = pathlib.Path(LIBRARY_DIR) / sanitized_name
        try: folder_path.mkdir(parents=True, exist_ok=True)
        except Exception: pass
def save_image_to_category(image_bytes, label, original_filename="thumbnail"):
    ensure_library_dir();
    if not label or label in ["Uncategorized", "Other / Unclear"]: return False, None
    base_filename, _ = os.path.splitext(original_filename); base_filename_sanitized = re.sub(r'[^\w\-]+', '_', base_filename).strip('_')[:50]; timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    sanitized_label = sanitize_foldername(label); category_path = pathlib.Path(LIBRARY_DIR) / sanitized_label; category_path.mkdir(parents=True, exist_ok=True)
    filename = f"{base_filename_sanitized}_{timestamp}.jpg"; filepath = category_path / filename; counter = 1
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
    sanitized_category = sanitize_foldername(category_name); category_path = pathlib.Path(LIBRARY_DIR) / sanitized_category; image_files = []
    if category_path.is_dir():
        for item in category_path.iterdir():
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'] and not item.name.startswith('.'): image_files.append(item)
    return sorted(image_files, key=lambda p: p.stat().st_mtime, reverse=True)
def delete_image_file(image_path_str):
    try:
        file_path = pathlib.Path(image_path_str)
        if file_path.is_file(): file_path.unlink(); st.toast(f"Deleted: {file_path.name}", icon="🗑️"); return True
        else: st.error(f"File not found: {file_path.name}"); return False
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

# ---------- API Key Setup Functions ----------
def get_api_key(key_name: str, service_name: str) -> str | None:
    """Gets API key from secrets or sidebar input."""
    api_key = None
    if hasattr(st, 'secrets') and key_name in st.secrets:
        api_key = st.secrets[key_name]
    else:
        st.sidebar.warning(f"{service_name} API Key not found in Secrets. Using input below.", icon="⚠️")
        api_key = st.sidebar.text_input(f"Enter {service_name} API key:", type="password", key=f"api_key_input_{key_name}")

    if not api_key:
        st.sidebar.error(f"{service_name} API key is required.")
        return None
    return api_key

def setup_openai_client(api_key: str) -> OpenAI | None:
    """Initializes OpenAI client."""
    if not api_key: return None
    try:
        client = OpenAI(api_key=api_key)
        # client.models.list() # Optional check
        return client
    except Exception as e:
        st.sidebar.error(f"Error initializing OpenAI client: {e}")
        return None

def setup_youtube_service(api_key: str):
    """Initializes YouTube Data API service."""
    if not api_key: return None
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        return youtube
    except Exception as e:
        st.sidebar.error(f"Error initializing YouTube client: {e}. Check API Key/Quota.")
        return None


# ---------- Utility Function ----------
# (encode_image remains the same)
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# ---------- YouTube API Fetching Function ----------
def get_channel_id_from_url(youtube, channel_url):
    """Tries to extract channel ID from various URL formats using YouTube API."""
    # 1. Check for /channel/UC... format
    match = re.search(r"youtube\.com/channel/([\w-]+)", channel_url)
    if match:
        return match.group(1)

    # 2. Check for /@handle format
    match = re.search(r"youtube\.com/@([\w.-]+)", channel_url)
    if match:
        handle = match.group(1)
        try:
            search_response = Youtube().list(
                q=handle, # Search by handle
                part='id',
                type='channel',
                maxResults=1
            ).execute()
            if search_response.get('items'):
                return search_response['items'][0]['id']['channelId']
            else:
                # Sometimes handle search fails, try search with '@'
                search_response = Youtube().list(q=f"@{handle}", part='id', type='channel', maxResults=1).execute()
                if search_response.get('items'):
                    return search_response['items'][0]['id']['channelId']

        except HttpError as e:
            st.warning(f"API error searching for handle '{handle}': {e}. Trying other methods.")
        except Exception as e:
            st.warning(f"Error searching for handle '{handle}': {e}. Trying other methods.")


    # 3. Check for /c/VanityName or /user/Username (less reliable)
    match = re.search(r"youtube\.com/(?:c/|user/)([\w.-]+)", channel_url)
    if match:
        name = match.group(1)
        # Search API is generally better for vanity names than channels.list(forUsername=...)
        try:
            search_response = Youtube().list(
                q=name,
                part='id',
                type='channel',
                maxResults=1
            ).execute()
            if search_response.get('items'):
                return search_response['items'][0]['id']['channelId']
        except HttpError as e:
            st.warning(f"API error searching for name '{name}': {e}.")
        except Exception as e:
            st.warning(f"Error searching for name '{name}': {e}.")

    # 4. Fallback: If URL itself is the ID (unlikely but possible)
    if channel_url.startswith("UC") and len(channel_url) == 24:
         return channel_url

    return None # Could not determine ID

def fetch_channel_videos_api(youtube, channel_url, max_results=25):
    """Fetches video data using YouTube Data API v3."""
    if not youtube:
        st.error("YouTube client not initialized.")
        return []

    st.write(f"Fetching channel info for URL: {channel_url}...")
    channel_id = get_channel_id_from_url(youtube, channel_url)

    if not channel_id:
        st.error(f"Could not determine Channel ID from URL: {channel_url}. Please provide a valid Channel URL (e.g., youtube.com/channel/UC..., youtube.com/@handle).")
        return []

    st.write(f"Found Channel ID: {channel_id}. Fetching latest {max_results} videos...")

    try:
        # Get the uploads playlist ID
        channel_response = youtube.channels().list(
            part='contentDetails',
            id=channel_id
        ).execute()

        if not channel_response.get('items'):
            st.error(f"Could not find channel details for ID: {channel_id}")
            return []

        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        # Get videos from the uploads playlist
        videos = []
        next_page_token = None
        fetched_count = 0

        while fetched_count < max_results: # Fetch until we reach max_results
            results_this_page = min(max_results - fetched_count, 50) # Fetch up to 50 per API call

            playlist_response = youtube.playlistItems().list(
                part='snippet',
                playlistId=uploads_playlist_id,
                maxResults=results_this_page,
                pageToken=next_page_token
            ).execute()

            for item in playlist_response.get('items', []):
                snippet = item.get('snippet', {})
                video_id = snippet.get('resourceId', {}).get('videoId')
                published_at_str = snippet.get('publishedAt') # ISO 8601 format (UTC)
                upload_date_obj = None
                if published_at_str:
                    try:
                        # Parse ISO 8601 string and convert to date object
                        upload_date_obj = datetime.fromisoformat(published_at_str.replace('Z', '+00:00')).date()
                    except ValueError:
                        upload_date_obj = None

                # Get best thumbnail URL
                thumbnails = snippet.get('thumbnails', {})
                thumb_url = thumbnails.get('maxres', {}).get('url') or \
                            thumbnails.get('standard', {}).get('url') or \
                            thumbnails.get('high', {}).get('url') or \
                            thumbnails.get('medium', {}).get('url') or \
                            thumbnails.get('default', {}).get('url')

                if video_id and thumb_url:
                    videos.append({
                        'video_id': video_id,
                        'title': snippet.get('title', 'N/A'),
                        'thumbnail_url': thumb_url,
                        'upload_date': upload_date_obj,
                        'webpage_url': f"https://www.youtube.com/watch?v={video_id}"
                    })
                    fetched_count += 1
                    if fetched_count >= max_results:
                        break # Stop if we've reached the desired number

            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break # No more pages

        st.write(f"Successfully fetched info for {len(videos)} videos.")
        return videos

    except HttpError as e:
        st.error(f"YouTube API Error: {e}. Check your API key, quota, and channel URL.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during API fetch: {e}")
        return []

# ---------- OpenAI Analysis Function (Single Label) ----------
def analyze_and_classify_thumbnail(client: OpenAI, image_bytes: bytes):
    """ Analyzes thumbnail for the single most relevant label. """
    # (Function remains same - requests single label)
    if not client: return "Uncategorized", "Client Error."
    base64_image = encode_image(image_bytes); image_data_uri = f"data:image/jpeg;base64,{base64_image}"
    # (Prompt definitions remain same)
    category_definitions_list = [ "Text-Dominant: ...","Minimalist / Clean: ...", ...] # Shortened for brevity
    category_definitions_text = "\n".join([f"- {cat.split(':')[0]}" for cat in category_definitions_list])
    valid_categories = set(STANDARD_CATEGORIES)
    try:
        response = client.chat.completions.create( model="gpt-4o", messages=[ { "role": "system", "content": f"Expert analyst: Identify the SINGLE most relevant category... Output ONLY the category name." }, { "role": "user", "content": [ { "type": "text", "text": f"Classify using ONLY these categories: {', '.join(valid_categories)}. Output the single best category name." }, { "type": "image_url", "image_url": {"url": image_data_uri, "detail": "low"} } ] } ], temperature=0.1, max_tokens=40 )
        result = response.choices[0].message.content.strip()
    except Exception as e: st.error(f"OpenAI Error: {e}"); return "Uncategorized", "API Error."
    # Validation
    label = "Uncategorized"; reason = "Reason not stored."
    if result:
        found = False
        for valid_cat in valid_categories:
            if valid_cat.strip().lower() == result.strip().lower(): label = valid_cat; found = True; break
        if not found: label = "Other / Unclear"; st.warning(f"AI suggested unknown: '{result}'. Using default.")
    else: label = "Uncategorized"; st.warning("AI empty response.")
    return label, reason


# ---------- Callbacks ----------
# (Callbacks add_to_library_callback, add_direct_to_library_callback,
#  analyze_all_uploads_callback, analyze_selected_callback remain the same logic,
#  but analyze_selected_callback downloads using requests)
# --- analyze_selected_callback (minor refinement) ---
def analyze_selected_callback():
    """Downloads and prepares selected fetched thumbnails for analysis."""
    if 'selected_thumbnails' in st.session_state and st.session_state.selected_thumbnails:
        if 'fetch_items' not in st.session_state: st.session_state.fetch_items = {}
        if 'fetched_thumbnails' not in st.session_state: st.session_state.fetched_thumbnails = []

        triggered_count = 0
        st.write("Queueing selected thumbnails for analysis...")
        progress_bar = st.progress(0)
        total_selected = len(st.session_state.selected_thumbnails)
        processed_ids = set() # Track IDs processed in this run

        for i, video_id in enumerate(list(st.session_state.selected_thumbnails)):
             if video_id in processed_ids: continue # Avoid double processing if selected multiple times somehow
             processed_ids.add(video_id)

             # Check existing status in fetch_items cache
             current_status = st.session_state.fetch_items.get(video_id, {}).get('status')
             if current_status not in [None, 'selected', 'error_download', 'error_processing']:
                 # st.caption(f"Skipping {video_id} (Status: {current_status}).")
                 continue

             video_data = next((item for item in st.session_state.fetched_thumbnails if item['video_id'] == video_id), None)
             if not video_data or not video_data.get('thumbnail_url'):
                 st.warning(f"Data/URL missing for {video_id}"); continue

             try:
                 # Download thumbnail using requests
                 response = requests.get(video_data['thumbnail_url'], stream=True, timeout=15); response.raise_for_status()
                 image_bytes = response.content
                 img = Image.open(io.BytesIO(image_bytes)); img.verify()
                 img = Image.open(io.BytesIO(image_bytes)).convert("RGB"); img_byte_arr = io.BytesIO()
                 img.save(img_byte_arr, format='JPEG', quality=85); processed_bytes = img_byte_arr.getvalue()

                 st.session_state.fetch_items[video_id] = {
                     'name': video_data.get('title', video_id), 'processed_bytes': processed_bytes,
                     'label': None, 'reason': "N/A", 'status': 'analyzing', 'source': 'fetch'
                 }
                 triggered_count += 1
             except requests.exceptions.RequestException as e: st.error(f"Download error {video_id}: {e}"); st.session_state.fetch_items[video_id] = {'status': 'error_download', 'name': video_data.get('title', video_id)}
             except Exception as e: st.error(f"Image processing error {video_id}: {e}"); st.session_state.fetch_items[video_id] = {'status': 'error_processing', 'name': video_data.get('title', video_id)}

             progress_bar.progress((i + 1) / total_selected)

        if triggered_count > 0: st.toast(f"Queued {triggered_count} thumbnail(s) for analysis.", icon="👍")
        # Clear selection after queueing
        st.session_state.selected_thumbnails = set()
        # Clear the select_all checkbox state if it exists
        if 'select_all_fetched' in st.session_state:
             st.session_state.select_all_fetched = False
        if 'prev_select_all' in st.session_state:
             st.session_state.prev_select_all = False

    else: st.toast("No thumbnails selected.", icon="🤔")


# (add_to_library_callback, add_direct_to_library_callback, analyze_all_uploads_callback remain same)
def add_to_library_callback(item_key, image_bytes, label, filename, source='upload'):
    success, _ = save_image_to_category(image_bytes, label, filename)
    cache_key_name = 'upload_items' if source == 'upload' else 'fetch_items'
    if success:
        if cache_key_name in st.session_state and item_key in st.session_state[cache_key_name]:
            st.session_state[cache_key_name][item_key]['status'] = 'added'
            st.toast(f"Saved to '{label}' folder!", icon="✅")
        else: st.warning(f"Cache status update failed for {filename}. File likely saved."); st.toast(f"Saved.", icon="✅")
    else: st.toast(f"Failed to save thumbnail.", icon="❌")

def add_direct_to_library_callback(file_id, image_bytes, selected_category, filename):
    success, _ = save_image_to_category(image_bytes, selected_category, filename)
    if success: st.session_state[f'direct_added_{file_id}'] = True; st.toast(f"Image added to '{selected_category}'!", icon="⬆️")
    else: st.toast(f"Failed to add image directly.", icon="❌")

def analyze_all_uploads_callback():
    if 'upload_items' in st.session_state:
        triggered_count = 0
        for item_id, item_data in st.session_state.upload_items.items():
            if item_data.get('status') == 'uploaded': st.session_state.upload_items[item_id]['status'] = 'analyzing'; triggered_count += 1
        if triggered_count > 0: st.toast(f"Triggered analysis for {triggered_count} upload(s).", icon="🧠")
        else: st.toast("No uploads awaiting analysis.", icon="🤷")


# ---------- Display and Process Analysis Items ----------
# (display_analysis_items remains the same - processes both caches)
def display_analysis_items(client: OpenAI):
    # ... (same logic as previous version, iterates through upload_items and fetch_items) ...
    st.subheader("Analysis Queue & Results")
    # ... rest of function ...


# ---------- Library Explorer ----------
# (library_explorer remains the same)
def library_explorer():
    # ... (same logic including delete confirmation) ...


# ---------- Delete Confirmation Dialog Function ----------
# (display_delete_confirmation remains the same)
def display_delete_confirmation():
     # ... (same logic) ...


# ---------- Main App ----------
def main():
    ensure_library_dir()
    create_predefined_category_folders(STANDARD_CATEGORIES)

    # --- Initialize Session State ---
    if 'selected_category_folder' not in st.session_state: st.session_state.selected_category_folder = None
    if 'upload_items' not in st.session_state: st.session_state.upload_items = {}
    if 'fetch_items' not in st.session_state: st.session_state.fetch_items = {}
    if 'fetched_thumbnails' not in st.session_state: st.session_state.fetched_thumbnails = []
    if 'selected_thumbnails' not in st.session_state: st.session_state.selected_thumbnails = set()
    if 'confirm_delete_path' not in st.session_state: st.session_state.confirm_delete_path = None
    if 'prev_select_all' not in st.session_state: st.session_state.prev_select_all = False

    # --- Sidebar Setup ---
    with st.sidebar:
        # ... (Same sidebar title/info) ...
        openai_api_key = get_api_key("OPENAI_API_KEY", "OpenAI")
        youtube_api_key = get_api_key("YOUTUBE_API_KEY", "YouTube Data v3") # Get YouTube key

        client_openai = setup_openai_client(openai_api_key)
        client_youtube = setup_youtube_service(youtube_api_key) # Setup YouTube service

        menu = st.radio( "Navigation", ["Analyze Thumbnails", "Library Explorer"], key="nav_menu", label_visibility="collapsed" )
        st.markdown("---")
        st.info(f"Library stored in './{LIBRARY_DIR}'")
        with st.expander("Standard Categories"):
             st.caption("\n".join(f"- {cat}" for cat in STANDARD_CATEGORIES if cat != "Other / Unclear"))
        st.caption("Uses OpenAI & YouTube Data API.")

    # --- Main Content Area ---
    if st.session_state.confirm_delete_path:
         display_delete_confirmation() # Show confirmation dialog first if active
    else:
         if menu == "Analyze Thumbnails":
             # Check both clients
             if not client_openai: st.error("❌ OpenAI client not initialized. Check API key.")
             # YouTube client needed only for fetching tab, allow uploads without it maybe?
             # else:

             tab1, tab2 = st.tabs(["⬆️ Upload Files", "📡 Fetch from Channel URL"])

             with tab1:
                 # === Upload Files Tab ===
                 st.header("Upload Files")
                 st.caption("Upload images, click 'Analyze All Uploads', then 'Add to Library'.")
                 # File Uploader Logic
                 uploaded_files = st.file_uploader( "Choose images...", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key="file_uploader_tab1")

                 # --- FIX: Indentation Error Fix Start ---
                 if uploaded_files: # This block needs indentation
                     for uploaded_file in uploaded_files:
                         file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                         if file_id not in st.session_state.upload_items:
                             try:
                                 image_bytes = uploaded_file.getvalue(); img = Image.open(io.BytesIO(image_bytes)); img.verify()
                                 img = Image.open(io.BytesIO(image_bytes)).convert("RGB"); img_byte_arr = io.BytesIO()
                                 img.save(img_byte_arr, format='JPEG', quality=85); processed_bytes = img_byte_arr.getvalue()
                                 st.session_state.upload_items[file_id] = { 'name': uploaded_file.name, 'original_bytes': image_bytes, 'processed_bytes': processed_bytes, 'label': None, 'reason': "N/A", 'status': 'uploaded', 'source': 'upload'}
                             except Exception as e: st.error(f"Error reading {uploaded_file.name}: {e}"); st.session_state.upload_items[file_id] = {'status': 'error', 'error_msg': str(e), 'name': uploaded_file.name, 'source': 'upload'}
                 # --- FIX: Indentation Error Fix End ---

                 # Control buttons for Upload tab
                 upload_ctrl_cols = st.columns(2)
                 with upload_ctrl_cols[0]:
                      items_to_analyze_upload = any(item.get('status') == 'uploaded' for item in st.session_state.upload_items.values())
                      analyze_uploads_disabled = not items_to_analyze_upload or not client_openai
                      if st.button("🧠 Analyze All Uploads", key="analyze_all_uploads", on_click=analyze_all_uploads_callback, disabled=analyze_uploads_disabled, use_container_width=True): st.rerun()
                 with upload_ctrl_cols[1]:
                      if st.button("Clear Uploads List", key="clear_uploads_tab1", use_container_width=True): st.session_state.upload_items = {}; st.rerun()


             with tab2:
                 # === Fetch from Channel Tab ===
                 st.header("Fetch from Channel URL")
                 st.caption("Enter channel URL, filter (optional), fetch, select thumbnails, then analyze.")
                 if not client_youtube: # Check if youtube client is ready
                     st.warning("YouTube API client not initialized. Please provide a valid YouTube Data API key in the sidebar or secrets.", icon="🔑")
                 else:
                     channel_url = st.text_input("YouTube Channel URL (e.g., youtube.com/channel/..., youtube.com/@handle)", key="channel_url_input")
                     col_d1, col_d2, col_n = st.columns([2,2,1])
                     start_date = col_d1.date_input("Start Date (Optional)", value=None, key="start_date")
                     end_date = col_d2.date_input("End Date (Optional)", value=None, key="end_date")
                     max_fetch = 25 # Hardcoded user request

                     if col_n.button("Fetch Thumbnails", key="fetch_button", disabled=not channel_url, use_container_width=True):
                         with st.spinner("Fetching video list via YouTube API..."):
                              videos = fetch_channel_videos_api(client_youtube, channel_url, max_results=max_fetch) # Fetch exactly max_fetch
                              st.session_state.fetched_thumbnails = videos
                              st.session_state.selected_thumbnails = set()
                              st.session_state.fetch_items = {} # Clear previous fetch analysis items
                              st.rerun()

                     # Display Fetched Thumbnails & Selection
                     if 'fetched_thumbnails' in st.session_state and st.session_state.fetched_thumbnails:
                          st.markdown("---"); st.subheader("Fetched Thumbnails")
                          # Apply Filters LOCALLY after fetching (API fetch limit already applied)
                          filtered_videos = st.session_state.fetched_thumbnails
                          if start_date: filtered_videos = [v for v in filtered_videos if v['upload_date'] and v['upload_date'] >= start_date]
                          if end_date: filtered_videos = [v for v in filtered_videos if v['upload_date'] and v['upload_date'] <= end_date]
                          display_videos = filtered_videos # Display all filtered results (up to max_fetch)
                          st.caption(f"Displaying {len(display_videos)} thumbnails (fetched max {max_fetch}, filtered by date).")

                          if display_videos:
                               # Select/Deselect All
                               current_selection_ids = {v['video_id'] for v in display_videos if v['video_id'] in st.session_state.selected_thumbnails}
                               all_displayed_selected = len(current_selection_ids) == len(display_videos)
                               new_select_all_state = st.checkbox("Select/Deselect All Displayed", value=all_displayed_selected, key="select_all_fetched")

                               # Update selections based on checkbox change more robustly
                               if new_select_all_state != all_displayed_selected: # If state changed
                                   if new_select_all_state: # Check all displayed
                                        for v in display_videos: st.session_state.selected_thumbnails.add(v['video_id'])
                                   else: # Uncheck all displayed
                                        for v in display_videos: st.session_state.selected_thumbnails.discard(v['video_id'])
                                   st.rerun() # Rerun to update checkbox states

                               # Grid Display with Checkboxes
                               num_cols_fetch = 4; fetch_cols = st.columns(num_cols_fetch)
                               for i, video in enumerate(display_videos):
                                   with fetch_cols[i % num_cols_fetch]:
                                       st.markdown('<div class="thumbnail-grid-item">', unsafe_allow_html=True)
                                       thumb_url = video.get('thumbnail_url')
                                       if thumb_url: st.image(thumb_url, caption=video.get('title', video['video_id'])[:50] + "...", use_container_width=True)
                                       else: st.caption("No thumbnail")
                                       is_selected = st.checkbox("Select", value=(video['video_id'] in st.session_state.selected_thumbnails), key=f"select_{video['video_id']}")
                                       # Logic moved to Select All for simplicity or use on_change if needed
                                       st.markdown('</div>', unsafe_allow_html=True)

                               # Analyze Selected Button
                               st.markdown("---")
                               analyze_selected_disabled = not st.session_state.selected_thumbnails or not client_openai
                               if st.button("✨ Analyze Selected Thumbnails", key="analyze_selected", on_click=analyze_selected_callback, disabled=analyze_selected_disabled, use_container_width=True):
                                    st.rerun() # Rerun to start processing items added to fetch_items cache
                          else: st.info("No fetched thumbnails match the selected date range.")
                     elif 'fetched_thumbnails' in st.session_state: st.info("No videos found or fetch failed for the URL.")

             # --- Common Analysis Display Area ---
             st.markdown("---")
             display_analysis_items(client_openai) # Pass OpenAI client here


         elif menu == "Library Explorer":
             library_explorer()

if __name__ == "__main__":
    main()
