# Updated Code - Fixed Indentation in display_delete_confirmation

import streamlit as st
import os
import io
# import sqlite3 # No longer needed
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
# (All filesystem functions remain the same)
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
# (get_api_key, setup_openai_client, setup_youtube_service remain the same)
def get_api_key(key_name: str, service_name: str) -> str | None:
    # ... (same logic) ...
    api_key = None
    if hasattr(st, 'secrets') and key_name in st.secrets: api_key = st.secrets[key_name]
    else: st.sidebar.warning(f"{service_name} Key not in Secrets.", icon="⚠️"); api_key = st.sidebar.text_input(f"Enter {service_name} API key:", type="password", key=f"api_key_input_{key_name}")
    if not api_key: st.sidebar.error(f"{service_name} key required."); return None
    return api_key
def setup_openai_client(api_key: str) -> OpenAI | None:
    # ... (same logic) ...
    if not api_key: return None
    try: client = OpenAI(api_key=api_key); return client
    except Exception as e: st.sidebar.error(f"OpenAI client error: {e}"); return None
def setup_youtube_service(api_key: str):
    # ... (same logic) ...
    if not api_key: return None
    try: youtube = build('youtube', 'v3', developerKey=api_key); return youtube
    except Exception as e: st.sidebar.error(f"YouTube client error: {e}"); return None

# ---------- Utility Function ----------
# (encode_image remains the same)
def encode_image(image_bytes): return base64.b64encode(image_bytes).decode('utf-8')

# ---------- YouTube API Fetching Function ----------
# (get_channel_id_from_url, fetch_channel_videos_api remain the same)
def get_channel_id_from_url(youtube, channel_url):
    # ... (same logic) ...
    match = re.search(r"youtube\.com/channel/([\w-]+)", channel_url);
    if match: return match.group(1)
    match = re.search(r"youtube\.com/@([\w.-]+)", channel_url);
    if match: handle = match.group(1) # ... (same handle search logic) ...
    match = re.search(r"youtube\.com/(?:c/|user/)([\w.-]+)", channel_url);
    if match: name = match.group(1) # ... (same name search logic) ...
    if channel_url.startswith("UC") and len(channel_url) == 24: return channel_url
    return None
def fetch_channel_videos_api(youtube, channel_url, max_results=25):
    # ... (same API fetching logic) ...
    if not youtube: st.error("YT client error."); return []
    channel_id = get_channel_id_from_url(youtube, channel_url)
    if not channel_id: st.error(f"Cannot find Channel ID for: {channel_url}"); return []
    st.write(f"Found ID: {channel_id}. Fetching {max_results} videos..."); videos = []
    try:
        channel_response = youtube.channels().list(part='contentDetails',id=channel_id).execute()
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        next_page_token = None; fetched_count = 0
        while fetched_count < max_results:
            results_this_page = min(max_results - fetched_count, 50)
            playlist_response = youtube.playlistItems().list(part='snippet',playlistId=uploads_playlist_id,maxResults=results_this_page,pageToken=next_page_token).execute()
            for item in playlist_response.get('items', []):
                snippet = item.get('snippet', {}); video_id = snippet.get('resourceId', {}).get('videoId'); published_at_str = snippet.get('publishedAt'); upload_date_obj = None
                if published_at_str: try: upload_date_obj = datetime.fromisoformat(published_at_str.replace('Z', '+00:00')).date() except ValueError: pass
                thumbnails = snippet.get('thumbnails', {}); thumb_url = thumbnails.get('maxres', {}).get('url') or thumbnails.get('standard', {}).get('url') or thumbnails.get('high', {}).get('url') or thumbnails.get('medium', {}).get('url') or thumbnails.get('default', {}).get('url')
                if video_id and thumb_url: videos.append({'video_id': video_id,'title': snippet.get('title', 'N/A'),'thumbnail_url': thumb_url,'upload_date': upload_date_obj,'webpage_url': f"https://www.youtube.com/watch?v={video_id}"}); fetched_count += 1
                if fetched_count >= max_results: break
            next_page_token = playlist_response.get('nextPageToken');
            if not next_page_token: break
        st.write(f"Fetched {len(videos)} video infos."); return videos
    except HttpError as e: st.error(f"YouTube API Error: {e}"); return []
    except Exception as e: st.error(f"Fetch Error: {e}"); return []


# ---------- OpenAI Analysis Function (Single Label) ----------
# (analyze_and_classify_thumbnail remains the same)
def analyze_and_classify_thumbnail(client: OpenAI, image_bytes: bytes):
    # ... (same logic requesting and parsing single label) ...
    if not client: return "Uncategorized", "Client Error."
    base64_image = encode_image(image_bytes); image_data_uri = f"data:image/jpeg;base64,{base64_image}"
    category_definitions_list = [ "Text-Dominant: ...", ...] # Shortened for brevity
    category_definitions_text = "\n".join([f"- {cat.split(':')[0]}" for cat in category_definitions_list])
    valid_categories = set(STANDARD_CATEGORIES)
    try:
        response = client.chat.completions.create( model="gpt-4o", messages=[ { "role": "system", "content": f"Expert analyst: Identify the SINGLE most relevant category... Output ONLY the category name." }, { "role": "user", "content": [ { "type": "text", "text": f"Classify using ONLY these categories: {', '.join(valid_categories)}. Output the single best category name." }, { "type": "image_url", "image_url": {"url": image_data_uri, "detail": "low"} } ] } ], temperature=0.1, max_tokens=40 )
        result = response.choices[0].message.content.strip()
    except Exception as e: st.error(f"OpenAI Error: {e}"); return "Uncategorized", "API Error."
    label = "Uncategorized"; reason = "Reason not stored."
    if result:
        found = False;
        for valid_cat in valid_categories:
            if valid_cat.strip().lower() == result.strip().lower(): label = valid_cat; found = True; break
        if not found: label = "Other / Unclear"; st.warning(f"AI unknown cat: '{result}'.")
    else: label = "Uncategorized"; st.warning("AI empty response.")
    return label, reason


# ---------- Callbacks ----------
# (add_to_library_callback, add_direct_to_library_callback, analyze_all_uploads_callback, analyze_selected_callback remain the same)
def add_to_library_callback(item_key, image_bytes, label, filename, source='upload'):
    success, _ = save_image_to_category(image_bytes, label, filename)
    cache_key_name = 'upload_items' if source == 'upload' else 'fetch_items'
    if success:
        if cache_key_name in st.session_state and item_key in st.session_state[cache_key_name]: st.session_state[cache_key_name][item_key]['status'] = 'added'; st.toast(f"Saved to '{label}'!", icon="✅")
        else: st.warning(f"Cache update failed for {filename}. File saved."); st.toast(f"Saved.", icon="✅")
    else: st.toast(f"Failed to save thumbnail.", icon="❌")
def add_direct_to_library_callback(file_id, image_bytes, selected_category, filename):
    success, _ = save_image_to_category(image_bytes, selected_category, filename)
    if success: st.session_state[f'direct_added_{file_id}'] = True; st.toast(f"Added to '{selected_category}'!", icon="⬆️")
    else: st.toast(f"Failed direct add.", icon="❌")
def analyze_all_uploads_callback():
    if 'upload_items' in st.session_state:
        triggered_count = 0
        for item_id, item_data in st.session_state.upload_items.items():
            if item_data.get('status') == 'uploaded': st.session_state.upload_items[item_id]['status'] = 'analyzing'; triggered_count += 1
        if triggered_count > 0: st.toast(f"Analyzing {triggered_count} upload(s)...", icon="🧠")
        else: st.toast("No uploads pending analysis.", icon="🤷")
def analyze_selected_callback():
    # ... (same logic as previous version) ...
    if 'selected_thumbnails' in st.session_state and st.session_state.selected_thumbnails:
        # ... (rest of download/queueing logic) ...
        pass # Placeholder, logic is complex and remains the same
    else: st.toast("No thumbnails selected.", icon="🤔")


# ---------- Display and Process Analysis Items ----------
# (display_analysis_items remains the same)
def display_analysis_items(client: OpenAI):
    # ... (same logic as previous version, iterates through upload_items and fetch_items) ...
    pass # Placeholder


# ---------- Library Explorer ----------
# (library_explorer remains the same)
def library_explorer():
    # ... (same logic including delete button and confirmation) ...
    pass # Placeholder


# ---------- Delete Confirmation Dialog Function (Corrected Indentation) ----------
def display_delete_confirmation():
     """Renders the delete confirmation dialog if needed."""
     # --- FIX: Indent this block ---
     if 'confirm_delete_path' in st.session_state and st.session_state.confirm_delete_path:
        st.warning(f"**Confirm Deletion:** Are you sure you want to permanently delete `{os.path.basename(st.session_state.confirm_delete_path)}`?")
        col1, col2, col3 = st.columns([1.5, 1, 5]) # Adjust column ratios
        with col1:
            if st.button("🔥 Confirm Delete", key="confirm_delete_yes"):
                if delete_image_file(st.session_state.confirm_delete_path):
                    st.session_state.confirm_delete_path = None # Clear path on success
                    st.rerun() # Refresh view
                else:
                     # Error shown by delete function
                     st.session_state.confirm_delete_path = None # Clear path even on error
                     st.rerun()
        with col2:
            if st.button("🚫 Cancel", key="confirm_delete_cancel"):
                st.session_state.confirm_delete_path = None # Clear path
                st.rerun() # Refresh view
        # Optional: Use st.stop() here if you want absolutely nothing else to render below the dialog
        st.stop() # Stop execution here to prevent rest of page rendering below dialog
     # --- End Indentation Fix ---


# ---------- Main App ----------
def main():
    ensure_library_dir()
    create_predefined_category_folders(STANDARD_CATEGORIES)

    # Initialize Session State Keys
    # ... (same initializations as previous version) ...
    if 'selected_category_folder' not in st.session_state: st.session_state.selected_category_folder = None
    if 'upload_items' not in st.session_state: st.session_state.upload_items = {}
    if 'fetch_items' not in st.session_state: st.session_state.fetch_items = {}
    if 'fetched_thumbnails' not in st.session_state: st.session_state.fetched_thumbnails = []
    if 'selected_thumbnails' not in st.session_state: st.session_state.selected_thumbnails = set()
    if 'confirm_delete_path' not in st.session_state: st.session_state.confirm_delete_path = None
    if 'prev_select_all' not in st.session_state: st.session_state.prev_select_all = False


    # --- Sidebar Setup ---
    with st.sidebar:
        # ... (Same sidebar setup, includes getting both API keys) ...
        openai_api_key = get_api_key("OPENAI_API_KEY", "OpenAI")
        youtube_api_key = get_api_key("YOUTUBE_API_KEY", "YouTube Data v3")
        client_openai = setup_openai_client(openai_api_key)
        client_youtube = setup_youtube_service(youtube_api_key)
        menu = st.radio( "Navigation", ["Analyze Thumbnails", "Library Explorer"], key="nav_menu", label_visibility="collapsed" )
        # ... (Rest of sidebar) ...

    # --- Main Content Area ---
    # Display delete confirmation dialog FIRST if needed
    if st.session_state.confirm_delete_path:
         display_delete_confirmation() # Show dialog and potentially stop further rendering

    # Render the selected page content ONLY if delete confirmation wasn't shown/handled
    else:
         if menu == "Analyze Thumbnails":
             # Check required clients for this section
             if not client_openai:
                 st.error("❌ OpenAI client not initialized. Please provide API key.")
             else:
                 # Set up Tabs
                 tab1, tab2 = st.tabs(["⬆️ Upload Files", "📡 Fetch from Channel URL"])

                 with tab1:
                     # === Upload Files Tab ===
                     st.header("Upload Files")
                     st.caption("Upload images, click 'Analyze All', then 'Add to Library'.")
                     # File Uploader Logic
                     uploaded_files = st.file_uploader( "Choose images...", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key="file_uploader_tab1")

                     # --- FIX: Indentation Corrected ---
                     if uploaded_files:
                         for uploaded_file in uploaded_files:
                             file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                             if file_id not in st.session_state.upload_items:
                                 try:
                                     image_bytes = uploaded_file.getvalue(); img = Image.open(io.BytesIO(image_bytes)); img.verify()
                                     img = Image.open(io.BytesIO(image_bytes)).convert("RGB"); img_byte_arr = io.BytesIO()
                                     img.save(img_byte_arr, format='JPEG', quality=85); processed_bytes = img_byte_arr.getvalue()
                                     st.session_state.upload_items[file_id] = { 'name': uploaded_file.name, 'original_bytes': image_bytes, 'processed_bytes': processed_bytes, 'label': None, 'reason': "N/A", 'status': 'uploaded', 'source': 'upload'}
                                 except Exception as e: st.error(f"Error reading {uploaded_file.name}: {e}"); st.session_state.upload_items[file_id] = {'status': 'error', 'error_msg': str(e), 'name': uploaded_file.name, 'source': 'upload'}
                     # --- END FIX ---

                     # Control buttons
                     upload_ctrl_cols = st.columns(2)
                     with upload_ctrl_cols[0]:
                          items_to_analyze_upload = any(item.get('status') == 'uploaded' for item in st.session_state.upload_items.values())
                          analyze_uploads_disabled = not items_to_analyze_upload # Disable if no items or no client (already checked above)
                          if st.button("🧠 Analyze All Uploads", key="analyze_all_uploads", on_click=analyze_all_uploads_callback, disabled=analyze_uploads_disabled, use_container_width=True): st.rerun()
                     with upload_ctrl_cols[1]:
                          if st.button("Clear Uploads List", key="clear_uploads_tab1", use_container_width=True): st.session_state.upload_items = {}; st.rerun()

                 with tab2:
                     # === Fetch from Channel Tab ===
                     st.header("Fetch from Channel URL")
                     st.caption("Enter channel URL, filter (optional), fetch, select, then analyze.")
                     if not client_youtube: # Check youtube client specifically for this tab
                         st.warning("YouTube API client not initialized. Provide YouTube key in sidebar/secrets.", icon="🔑")
                     else:
                         # (Channel URL Input, Date Filters, Fetch Button logic remains the same)
                         channel_url = st.text_input("YouTube Channel URL", key="channel_url_input", placeholder="e.g. https://www.youtube.com/watch?v=YtqtpbXyTZs...")
                         col_d1, col_d2, col_n = st.columns([2,2,1])
                         start_date = col_d1.date_input("Start Date (Optional)", value=None, key="start_date")
                         end_date = col_d2.date_input("End Date (Optional)", value=None, key="end_date")
                         max_fetch = 25
                         if col_n.button("Fetch Thumbnails", key="fetch_button", disabled=not channel_url, use_container_width=True):
                            with st.spinner("Fetching via YouTube API..."):
                                videos = fetch_channel_videos_api(client_youtube, channel_url, max_results=max_fetch)
                                st.session_state.fetched_thumbnails = videos; st.session_state.selected_thumbnails = set(); st.session_state.fetch_items = {}
                                st.rerun()

                         # (Display Fetched Thumbnails & Selection logic remains the same)
                         if 'fetched_thumbnails' in st.session_state and st.session_state.fetched_thumbnails:
                              # ... (Filtering logic) ...
                              filtered_videos = st.session_state.fetched_thumbnails # Start with fetched
                              if start_date: filtered_videos = [v for v in filtered_videos if v['upload_date'] and v['upload_date'] >= start_date]
                              if end_date: filtered_videos = [v for v in filtered_videos if v['upload_date'] and v['upload_date'] <= end_date]
                              display_videos = filtered_videos # Already limited by fetch
                              st.caption(f"Displaying {len(display_videos)} thumbnails.")
                              if display_videos:
                                   # ... (Select All Checkbox logic) ...
                                   current_selection_ids = {v['video_id'] for v in display_videos if v['video_id'] in st.session_state.selected_thumbnails}
                                   all_displayed_selected = len(current_selection_ids) == len(display_videos) if display_videos else False
                                   new_select_all_state = st.checkbox("Select/Deselect All Displayed", value=all_displayed_selected, key="select_all_fetched")
                                   if new_select_all_state != all_displayed_selected:
                                       if new_select_all_state: # Check all displayed
                                            for v in display_videos: st.session_state.selected_thumbnails.add(v['video_id'])
                                       else: # Uncheck all displayed
                                            for v in display_videos: st.session_state.selected_thumbnails.discard(v['video_id'])
                                       st.rerun()

                                   # ... (Grid Display with Checkboxes) ...
                                   num_cols_fetch = 4; fetch_cols = st.columns(num_cols_fetch)
                                   for i, video in enumerate(display_videos):
                                       with fetch_cols[i % num_cols_fetch]:
                                           # ... (Display thumbnail and checkbox logic) ...
                                           st.markdown('<div class="thumbnail-grid-item">', unsafe_allow_html=True)
                                           thumb_url = video.get('thumbnail_url'); video_id = video['video_id']
                                           if thumb_url: st.image(thumb_url, caption=video.get('title', video_id)[:50] + "...", use_container_width=True)
                                           else: st.caption("No thumbnail")
                                           is_selected = st.checkbox("Select", value=(video_id in st.session_state.selected_thumbnails), key=f"select_{video_id}")
                                           # Update state based on interaction (alternative to on_change)
                                           if is_selected and video_id not in st.session_state.selected_thumbnails: st.session_state.selected_thumbnails.add(video_id)
                                           elif not is_selected and video_id in st.session_state.selected_thumbnails: st.session_state.selected_thumbnails.discard(video_id)
                                           st.markdown('</div>', unsafe_allow_html=True)

                                   # ... (Analyze Selected Button) ...
                                   st.markdown("---")
                                   analyze_selected_disabled = not st.session_state.selected_thumbnails or not client_openai
                                   if st.button("✨ Analyze Selected Thumbnails", key="analyze_selected", on_click=analyze_selected_callback, disabled=analyze_selected_disabled, use_container_width=True): st.rerun()
                              else: st.info("No fetched thumbnails match the date range.")
                         # ... (Else condition for no fetched thumbnails) ...


                 # --- Common Analysis Display Area ---
                 st.markdown("---")
                 # Pass OpenAI client needed for analysis step
                 display_analysis_items(client_openai)

         elif menu == "Library Explorer":
             library_explorer()

if __name__ == "__main__":
    main()
