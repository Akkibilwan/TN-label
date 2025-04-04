import streamlit as st
import os
import io
import sqlite3
import base64
import zipfile
from datetime import datetime
from PIL import Image
from openai import OpenAI
import re # For parsing multiple labels

# Set page configuration
st.set_page_config(
    page_title="Thumbnail Analyzer",
    page_icon="🎬",
    layout="wide"
)

# ---------- Custom CSS ----------
# (CSS remains mostly the same, minor adjustments possible)
st.markdown("""
<style>
    .main { background-color: #0f0f0f; color: #f1f1f1; }
    /* ... other styles ... */
    .stButton>button { /* Default button style */
        background-color: #444; color: #f1f1f1; border: 1px solid #666;
        border-radius: 4px; padding: 8px 16px; font-weight: 500;
        margin-top: 5px; margin-right: 5px;
    }
     .stButton>button:hover { opacity: 0.8; border-color: #888; }
    /* Specific button styles */
    button:has(span:contains("✅ Add to Library")) { background-color: #4CAF50; border: none;} /* Green */
    button:disabled:has(span:contains("✔️ Added")) { background-color: #3a7d3d; border: none; opacity: 0.7;} /* Darker Green */
    button:has(span:contains("⬆️ Add Uploaded Image")) { background-color: #ffc107; border: none; color: #333;} /* Yellow */
    .stDownloadButton>button { background-color: #007bff; border: none; } /* Blue */
    button:has(span:contains("⬅️ Back")) { background-color: #6c757d; border: none; } /* Grey */
    button:has(span:contains("Clear Uploads")) { background-color: #dc3545; border: none; } /* Red */

    /* Category buttons in library explorer */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
         background-color: #333; color: #f1f1f1; border: 1px solid #555;
    }
     div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
         border-color: #aaa; color: #fff; background-color: #444;
     }

    .thumbnail-container, .db-thumbnail-container { border: 1px solid #303030; border-radius: 8px; padding: 15px; background-color: #181818; margin-bottom: 15px; height: 100%; display: flex; flex-direction: column;}
    .analysis-box { border: 1px dashed #444; padding: 10px; margin-top: 10px; border-radius: 4px; background-color: #202020;}
    .stExpander > div:first-child > button { color: #f1f1f1 !important; }
    img { border-radius: 4px; }
    div[data-testid="stImage"]{ display: flex; justify-content: center; margin-bottom: 10px; }
    .db-thumbnail-container .stImage { flex-grow: 1; }
    .multi-label-tag {
        background-color: #555; color: #eee; padding: 2px 6px; margin: 2px;
        border-radius: 3px; font-size: 0.8em; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ---------- SQLite Database Functions ----------
DB_NAME = "thumbnails.db"
LABEL_DELIMITER = ";;" # Define a delimiter for multiple labels

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create table if not exists
    c.execute("""
        CREATE TABLE IF NOT EXISTS thumbnails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB NOT NULL,
            label TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Check and add 'reason' column if it doesn't exist (Migration)
    c.execute("PRAGMA table_info(thumbnails)")
    columns = [column[1] for column in c.fetchall()]
    if 'reason' not in columns:
        st.warning("Database schema outdated. Adding 'reason' column...")
        c.execute("ALTER TABLE thumbnails ADD COLUMN reason TEXT")
        st.success("Database schema updated.")

    # Optional: Index on label
    c.execute("CREATE INDEX IF NOT EXISTS idx_label ON thumbnails (label)")
    conn.commit()
    conn.close()


def store_thumbnail_record(image_bytes, label_str, reason): # label is now potentially delimited string
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO thumbnails (image, label, reason)
            VALUES (?, ?, ?)
        """, (sqlite3.Binary(image_bytes), label_str, reason))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Database Error: Failed to store record. {e}")
        return False
    finally:
        conn.close()

def get_labels():
    """Gets unique individual labels from the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT label FROM thumbnails WHERE label IS NOT NULL AND label != '' AND label != 'Uncategorized'")
    all_labels_str = [row[0] for row in c.fetchall()]
    conn.close()

    unique_labels = set()
    for label_str in all_labels_str:
        # Split the potentially delimited string and add individual labels
        labels = [lbl.strip() for lbl in label_str.split(LABEL_DELIMITER) if lbl.strip()]
        unique_labels.update(labels)

    # Exclude the generic catch-all if present among specifics
    if "Other / Unclear" in unique_labels and len(unique_labels) > 1:
        pass # Keep it if it's the only one, otherwise let specific labels dominate maybe? Or always show? Let's always show for now.
        # unique_labels.discard("Other / Unclear")

    return sorted(list(unique_labels))


def get_records_by_label(selected_label):
    """Gets records where the label column CONTAINS the selected label."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    # Use LIKE clause to find records containing the label within the delimited string
    # Need to handle the delimiter correctly in the LIKE pattern
    like_pattern = f"%{LABEL_DELIMITER}{selected_label}{LABEL_DELIMITER}%" # Match label surrounded by delimiters
    like_pattern_start = f"{selected_label}{LABEL_DELIMITER}%" # Match label at the start
    like_pattern_end = f"%{LABEL_DELIMITER}{selected_label}" # Match label at the end
    exact_match_pattern = f"{selected_label}" # Match if it's the only label

    query = """
        SELECT id, image, label, reason, created_at
        FROM thumbnails
        WHERE label = ? OR label LIKE ? OR label LIKE ? OR label LIKE ?
        ORDER BY created_at DESC
    """
    c.execute(query, (exact_match_pattern, like_pattern_start, like_pattern_end, like_pattern))
    # Simpler but potentially less precise: c.execute("SELECT ... WHERE label LIKE ?", (f'%{selected_label}%',))

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
            return None

    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.sidebar.error(f"Error initializing OpenAI client: {e}. Check API key and permissions.")
        return None


# ---------- Utility Function ----------
# (encode_image function remains the same)
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# ---------- OpenAI Analysis & Classification Function (Modified for Multi-Label) ----------
def analyze_and_classify_thumbnail(client: OpenAI, image_bytes: bytes):
    if not client:
        return [], "OpenAI client not initialized." # Return empty list for labels

    base64_image = encode_image(image_bytes)
    image_data_uri = f"data:image/jpeg;base64,{base64_image}"

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
                    "content": f"You are an expert analyst of YouTube thumbnail visual styles. Analyze the provided thumbnail image and identify the **top 1 to 3 most relevant** visual style/layout categories based on its dominant features, using the provided definitions. If only one category clearly applies, list only that one. If unsure, use 'Other / Unclear'. Respond ONLY in the specific format 'Labels: <Category 1>{LABEL_DELIMITER}<Category 2>{LABEL_DELIMITER}<Category 3>\\nReason: <Brief overall reason covering the chosen labels>'."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Classify this thumbnail using ONLY these definitions, providing 1 to 3 relevant labels:\n{category_definitions_text}\n\nStrictly follow the output format:\nLabels: <Label 1>{LABEL_DELIMITER}<Label 2>{LABEL_DELIMITER}<Label 3>\nReason: <Your brief overall reason>"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri, "detail": "low"}
                        }
                    ]
                }
            ],
            temperature=0.3, # Slightly higher temp may allow for multiple labels more easily
            max_tokens=150 # Increased slightly for potentially longer label list + reason
        )
        result = response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error during OpenAI analysis: {e}")
        return [], "Analysis failed due to an API error." # Return empty list

    # Parse the multi-label output
    labels = []
    reason = "Analysis could not determine a category or reason."
    try:
        lines = result.split('\n')
        # Find the Labels line
        labels_line = ""
        reason_line = ""
        if len(lines) >= 1 and lines[0].startswith("Labels:"):
            labels_line = lines[0].replace("Labels:", "").strip()
            if len(lines) >= 2 and lines[1].startswith("Reason:"):
                reason_line = lines[1].replace("Reason:", "").strip()
        elif len(lines) >= 2 and lines[1].startswith("Labels:"): # Handle potential preamble line
             labels_line = lines[1].replace("Labels:", "").strip()
             if len(lines) >= 3 and lines[2].startswith("Reason:"):
                 reason_line = lines[2].replace("Reason:", "").strip()


        if labels_line:
            # Split labels by delimiter and validate
            raw_labels = [lbl.strip() for lbl in labels_line.split(LABEL_DELIMITER) if lbl.strip()]
            for raw_label in raw_labels:
                found = False
                for valid_cat in valid_categories:
                    if valid_cat.lower() == raw_label.lower():
                        labels.append(valid_cat) # Keep official casing
                        found = True
                        break
                if not found:
                    st.warning(f"AI returned unrecognized label: '{raw_label}'. Discarding.")

            # Ensure max 3 labels, prioritize the first ones returned by AI
            labels = labels[:3]

        if reason_line:
            reason = reason_line
        elif labels: # If we got labels but no reason line
            reason = "No specific reason provided by AI."

    except Exception as parse_error:
        st.warning(f"Could not parse AI multi-label response: '{result}'. Error: {parse_error}")
        labels = [] # Reset labels on parse error
        reason = "Failed to parse the analysis result format."

    if not labels: # If parsing failed or AI returned nothing valid
        labels = ["Uncategorized"]
        reason = "Could not determine valid categories."


    return labels, reason # Return list of labels

# ---------- Callback function for Add to Library button ----------
def add_to_library_callback(file_id, image_bytes, labels_list, reason):
    """Stores the record (joining labels) and updates session state."""
    label_str = LABEL_DELIMITER.join(labels_list) # Join list into delimited string for DB
    if store_thumbnail_record(image_bytes, label_str, reason):
        st.session_state[f'upload_cache_{file_id}']['status'] = 'added' # Update status in cache
        st.toast(f"Thumbnail added to library with labels: {', '.join(labels_list)}!", icon="✅")
    else:
        st.toast(f"Failed to add thumbnail to library.", icon="❌")

# ---------- Callback function for Direct Add in Library Explorer ----------
def add_direct_to_library_callback(file_id, image_bytes, selected_category):
    """Stores a manually uploaded record with a default reason."""
    reason = "Manually added to category."
    label_str = selected_category # Store only the selected category as label
    if store_thumbnail_record(image_bytes, label_str, reason):
        st.session_state[f'direct_added_{file_id}'] = True
        st.toast(f"Image added directly to '{selected_category}' category!", icon="⬆️")
        # No need to rerun here, list should update on next interaction or button click
    else:
        st.toast(f"Failed to add image directly to library.", icon="❌")


# ---------- Upload and Process Function (Modified for State) ----------
def upload_and_process(client: OpenAI):
    st.header("Upload & Analyze Thumbnails")
    st.info("Upload images, get AI category suggestions (up to 3), and click '✅ Add to Library' to save.")

    # --- Initialize session state for upload cache if not present ---
    if 'upload_cache' not in st.session_state:
        st.session_state.upload_cache = {} # Dict to store {file_id: {data}}

    # --- File Uploader ---
    uploaded_files = st.file_uploader(
        "Choose thumbnail images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="file_uploader" # Keep key for potential reset later
    )

    # --- Process newly uploaded files ---
    newly_uploaded_ids = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_id = uploaded_file.id if hasattr(uploaded_file, 'id') else uploaded_file.name
            newly_uploaded_ids.append(file_id)
            # If file is not already in cache, add it with initial state
            if file_id not in st.session_state.upload_cache:
                try:
                    image_bytes = uploaded_file.getvalue()
                    # Prepare display image and processed bytes (JPEG)
                    display_image = Image.open(io.BytesIO(image_bytes))
                    img_byte_arr = io.BytesIO()
                    processed_image = display_image
                    if processed_image.mode in ('RGBA', 'LA') or (processed_image.mode == 'P' and 'transparency' in processed_image.info):
                         processed_image = processed_image.convert('RGB')
                    processed_image.save(img_byte_arr, format='JPEG', quality=85)
                    processed_image_bytes = img_byte_arr.getvalue()

                    st.session_state.upload_cache[file_id] = {
                        'name': uploaded_file.name,
                        'original_bytes': image_bytes, # Store original for display
                        'processed_bytes': processed_image_bytes, # Store JPEG for analysis/DB
                        'labels': None, # Analysis result
                        'reason': None, # Analysis reason
                        'status': 'uploaded' # Status: uploaded, analyzing, analyzed, added, error
                    }
                except Exception as e:
                     st.error(f"Error reading file {uploaded_file.name}: {e}")
                     st.session_state.upload_cache[file_id] = {'status': 'error', 'error_msg': str(e), 'name': uploaded_file.name}


    # --- Display and Process items from Cache ---
    if st.session_state.upload_cache:
        st.markdown("---")
        # Button to clear the upload cache/session
        if st.button("Clear Uploads and Analyses", key="clear_uploads"):
            st.session_state.upload_cache = {}
            st.rerun()

        num_columns = 3
        cols = st.columns(num_columns)
        col_index = 0

        # Keep track of items still present (in case files are de-selected in uploader)
        active_ids = set(st.session_state.upload_cache.keys()) # Check which ones we still need to show

        # Iterate through cached items
        for file_id, item_data in list(st.session_state.upload_cache.items()):
             # If file is no longer in the latest upload list (if uploader is active), remove from cache
             # This handles file de-selection in the uploader widget
             # if uploaded_files is not None and file_id not in newly_uploaded_ids:
             #     del st.session_state.upload_cache[file_id]
             #     continue # Skip to next cached item

             with cols[col_index % num_columns]:
                 with st.container():
                     st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
                     try:
                         # Display image from cached original bytes
                         display_image = Image.open(io.BytesIO(item_data['original_bytes']))
                         st.image(display_image, caption=f"{item_data['name']}", use_container_width=True)

                         analysis_placeholder = st.empty()

                         # Process based on status
                         if item_data['status'] == 'uploaded':
                             analysis_placeholder.info("Ready to analyze...")
                             if st.button("Analyze", key=f"analyze_{file_id}"):
                                 st.session_state.upload_cache[file_id]['status'] = 'analyzing'
                                 st.rerun()

                         elif item_data['status'] == 'analyzing':
                             with analysis_placeholder.container():
                                 with st.spinner(f"Analyzing {item_data['name']}..."):
                                     labels, reason = analyze_and_classify_thumbnail(client, item_data['processed_bytes'])
                                     st.session_state.upload_cache[file_id]['labels'] = labels
                                     st.session_state.upload_cache[file_id]['reason'] = reason
                                     st.session_state.upload_cache[file_id]['status'] = 'analyzed'
                                     st.rerun() # Rerun immediately to show results

                         elif item_data['status'] in ['analyzed', 'added']:
                             with analysis_placeholder.container():
                                 st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                                 labels = item_data.get('labels', ['Uncategorized'])
                                 reason = item_data.get('reason', 'N/A')
                                 # Display multiple labels using tags
                                 label_tags_html = "".join([f'<span class="multi-label-tag">{lbl}</span>' for lbl in labels])
                                 st.markdown(f"**Suggested:** {label_tags_html}", unsafe_allow_html=True)
                                 st.markdown(f"**Reason:** _{reason}_")

                                 is_added = (item_data['status'] == 'added')

                                 st.button(
                                     "✅ Add to Library" if not is_added else "✔️ Added",
                                     key=f'btn_add_{file_id}',
                                     on_click=add_to_library_callback,
                                     args=(file_id, item_data['processed_bytes'], labels, reason),
                                     disabled=is_added or "Uncategorized" in labels # Disable if added or only 'Uncategorized'
                                 )
                                 st.markdown('</div>', unsafe_allow_html=True)

                         elif item_data['status'] == 'error':
                             analysis_placeholder.error(f"Error: {item_data.get('error_msg', 'Unknown processing error')}")

                     except Exception as e:
                          st.error(f"Error displaying item {item_data.get('name', file_id)}: {e}")
                     finally:
                          st.markdown('</div>', unsafe_allow_html=True) # Close thumbnail-container

             col_index += 1

    elif not st.session_state.upload_cache:
        st.markdown("<p style='text-align: center; font-style: italic;'>Upload some thumbnails to get started!</p>", unsafe_allow_html=True)


# ---------- Function to create Zip File in Memory ----------
# (create_zip_in_memory function remains the same)
def create_zip_in_memory(records):
    zip_buffer = io.BytesIO()
    filenames_in_zip = set()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for record in records:
            rec_id = record['id']
            image_blob = record['image']
            # Use the full label string for context, but sanitize for filename
            label_str_sanitized = re.sub(r'[^\w\-]+', '_', record['label']).strip('_')
            created_at_str = record['created_at'] if record['created_at'] else datetime.now().isoformat()
            try:
                 try: timestamp_dt = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S.%f')
                 except ValueError: timestamp_dt = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')
                 timestamp_str_fmt = timestamp_dt.strftime('%Y%m%d_%H%M%S')
            except (ValueError, TypeError): timestamp_str_fmt = "unknown_time"

            base_filename = f"{label_str_sanitized[:30]}_{rec_id}_{timestamp_str_fmt}" # Limit label part length
            filename_in_zip = f"{base_filename}.jpg"
            counter = 1
            while filename_in_zip in filenames_in_zip:
                 filename_in_zip = f"{base_filename}_{counter}.jpg"
                 counter += 1
            filenames_in_zip.add(filename_in_zip)

            try:
                zip_file.writestr(filename_in_zip, image_blob)
            except Exception as zip_err:
                st.warning(f"Could not add image ID {rec_id} to zip: {zip_err}")

    zip_buffer.seek(0)
    return zip_buffer


# ---------- Library Explorer (Modified for Direct Upload) ----------
def library_explorer():
    st.header("Thumbnail Library Explorer")
    st.markdown("Browse saved thumbnails by category. You can also directly upload to a selected category.")
    labels = get_labels()

    if "selected_label" not in st.session_state:
        st.session_state.selected_label = None

    # Category Selection Grid
    if st.session_state.selected_label is None:
        st.markdown("### Select a Category to View")
        if not labels:
            st.info("The library is empty. Add some analyzed thumbnails first!")
            return
        cols_per_row = 4
        num_labels = len(labels)
        num_rows = (num_labels + cols_per_row - 1) // cols_per_row
        for i in range(num_rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < num_labels:
                    label = labels[idx]
                    if cols[j].button(label, key=f"btn_lib_{label.replace(' ', '_')}", use_container_width=True):
                        st.session_state.selected_label = label
                        st.rerun()
    else:
        # Display Selected Category Content
        selected_category = st.session_state.selected_label
        st.markdown(f"### Category: **{selected_category}**")

        # --- Top Bar: Back Button, Direct Upload Expander, Download Button ---
        top_cols = st.columns([0.25, 0.45, 0.3]) # Adjust ratios
        with top_cols[0]:
            if st.button("⬅️ Back to Categories", key="back_button", use_container_width=True):
                st.session_state.selected_label = None
                st.rerun()

        # --- Direct Upload Section (inside expander) ---
        with top_cols[1]:
             with st.expander(f"⬆️ Add Image Directly to '{selected_category}'"):
                 direct_uploaded_file = st.file_uploader(
                     f"Upload image for '{selected_category}'",
                     type=["jpg", "jpeg", "png"],
                     key=f"direct_upload_{selected_category.replace(' ', '_')}"
                 )
                 if direct_uploaded_file:
                     file_id = direct_uploaded_file.id if hasattr(direct_uploaded_file, 'id') else direct_uploaded_file.name
                     # Use session state to track added status for direct uploads too
                     if f'direct_added_{file_id}' not in st.session_state:
                         st.session_state[f'direct_added_{file_id}'] = False

                     is_added = st.session_state[f'direct_added_{file_id}']

                     st.image(direct_uploaded_file, width=150)
                     # Prepare bytes (convert to JPEG)
                     img_bytes_direct = direct_uploaded_file.getvalue()
                     img_direct = Image.open(io.BytesIO(img_bytes_direct))
                     img_byte_arr_direct = io.BytesIO()
                     if img_direct.mode in ('RGBA', 'LA') or (img_direct.mode == 'P' and 'transparency' in img_direct.info):
                          img_direct = img_direct.convert('RGB')
                     img_direct.save(img_byte_arr_direct, format='JPEG', quality=85)
                     processed_bytes_direct = img_byte_arr_direct.getvalue()

                     st.button(
                         f"⬆️ Add Uploaded Image" if not is_added else "✔️ Added",
                         key=f"btn_direct_add_{file_id}",
                         on_click=add_direct_to_library_callback,
                         args=(file_id, processed_bytes_direct, selected_category),
                         disabled=is_added
                     )


        records = get_records_by_label(selected_category)

        if records:
            # --- Download Button ---
             with top_cols[2]:
                  zip_buffer = create_zip_in_memory(records)
                  st.download_button(
                      label=f"⬇️ Download ({len(records)}) Zip",
                      data=zip_buffer,
                      file_name=f"{selected_category.replace('/', '_').replace(' ', '_')}_thumbnails.zip",
                      mime="application/zip",
                      key=f"download_{selected_category.replace(' ', '_')}",
                      use_container_width=True
                  )

             # --- Thumbnail Display Grid ---
             st.markdown("---")
             num_records = len(records)
             cols_per_row_thumbs = 4
             thumb_cols = st.columns(cols_per_row_thumbs)
             col_idx = 0

             for record in records:
                 with thumb_cols[col_idx % cols_per_row_thumbs]:
                     rec_id = record['id']
                     image_blob = record['image']
                     label_str = record['label'] # Full label string from DB
                     reason = record['reason']
                     created_at_str = record['created_at']
                     # (Timestamp parsing logic remains the same)
                     try:
                        try: created_at = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S.%f')
                        except ValueError: created_at = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')
                        display_time = created_at.strftime('%Y-%m-%d %H:%M')
                     except (ValueError, TypeError): display_time = created_at_str

                     with st.container():
                         st.markdown('<div class="db-thumbnail-container">', unsafe_allow_html=True)
                         try:
                             image = Image.open(io.BytesIO(image_blob))
                             st.image(image, caption=f"ID: {rec_id} | {display_time}", use_container_width=True)
                             # Display multiple labels using tags
                             labels = [lbl.strip() for lbl in label_str.split(LABEL_DELIMITER) if lbl.strip()]
                             label_tags_html = "".join([f'<span class="multi-label-tag">{lbl}</span>' for lbl in labels])
                             st.markdown(f"**Labels:** {label_tags_html}", unsafe_allow_html=True)

                             with st.expander("Show Reason"):
                                 st.markdown(f"_{reason}_")
                         except Exception as img_err:
                              st.warning(f"Could not load image for ID {rec_id}: {img_err}")
                         st.markdown('</div>', unsafe_allow_html=True)
                 col_idx += 1
        else:
            st.info(f"No thumbnails found for the category: '{selected_category}'. You can add one directly using the expander above.")


# ---------- Main App ----------
def main():
    # Initialize DB Schema (handles migration check)
    init_db()

    # --- Initialize Session State Keys ---
    if 'selected_label' not in st.session_state:
        st.session_state.selected_label = None
    if 'upload_cache' not in st.session_state:
        st.session_state.upload_cache = {} # Initialize upload cache
    # 'added_{file_id}' and 'direct_added_{file_id}' are created dynamically

    # --- Sidebar Setup ---
    with st.sidebar:
        st.markdown(
            '<div style="display: flex; align-items: center; padding: 10px 0;">'
            '<span style="color: #FF0000; font-size: 28px; font-weight: bold; margin-right: 5px;">▶️</span>'
            '<h1 style="margin: 0; color: #f1f1f1; font-size: 24px;">Thumbnail Analyzer</h1></div>',
            unsafe_allow_html=True
        )
        st.markdown('<p style="color: #aaaaaa; margin-top: 0;">Analyze & Explore Thumbnails using AI</p>', unsafe_allow_html=True)
        st.markdown("---")

        client = setup_openai_client() # Initialize client

        menu = st.radio(
            "Navigation",
            ["Upload & Analyze", "Library Explorer"],
            key="nav_menu",
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.info("Thumbnails added to the library are stored locally in `thumbnails.db`.")
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
