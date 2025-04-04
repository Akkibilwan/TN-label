import streamlit as st
import os
import io
import sqlite3
import base64
import zipfile # Added for zip functionality
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
    p, li, div[data-testid="stMarkdownContainer"] { color: #aaaaaa; } /* Target markdown text */
    .stButton>button { background-color: #4CAF50; color: white; border: none; border-radius: 4px; padding: 8px 16px; font-weight: 500; margin-top: 5px; margin-right: 5px;} /* Green add button */
    .stDownloadButton>button { background-color: #007bff; color: white; border: none; border-radius: 4px; padding: 8px 16px; font-weight: 500; margin-top: 5px;} /* Blue download button */
    /* Style category buttons in library explorer */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
         background-color: #333;
         color: #f1f1f1;
         border: 1px solid #555;
    }
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
         border-color: #aaa;
         color: #fff;
    }
    /* Style back button differently */
    button[kind="secondary"]:has(span:contains("⬅️ Back")) {
         background-color: #555;
         border: 1px solid #777;
    }
    .stButton>button:hover, .stDownloadButton>button:hover { opacity: 0.8; }
    .thumbnail-container, .db-thumbnail-container { border: 1px solid #303030; border-radius: 8px; padding: 15px; background-color: #181818; margin-bottom: 15px; height: 100%; display: flex; flex-direction: column;}
    .analysis-box { border: 1px dashed #444; padding: 10px; margin-top: 10px; border-radius: 4px; background-color: #202020;}
    .stExpander > div:first-child > button { color: #f1f1f1 !important; }
    img { border-radius: 4px; } /* Slightly round image corners */
    div[data-testid="stImage"]{ /* Center image within its container */
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }
    .db-thumbnail-container .stImage { flex-grow: 1; } /* Make image take available space */
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
    # Optional: Index on label for faster lookups
    c.execute("CREATE INDEX IF NOT EXISTS idx_label ON thumbnails (label)")
    conn.commit()
    conn.close()

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
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Ensure we don't get None or empty strings if they somehow get saved
    c.execute("SELECT DISTINCT label FROM thumbnails WHERE label IS NOT NULL AND label != '' AND label != 'Uncategorized'")
    labels = sorted([row[0] for row in c.fetchall()])
    conn.close()
    return labels

def get_records_by_label(label):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # Return rows as dict-like objects
    c = conn.cursor()
    c.execute("SELECT id, image, label, reason, created_at FROM thumbnails WHERE label=? ORDER BY created_at DESC", (label,)) # Order by newest first
    records = c.fetchall()
    conn.close()
    return records

# ---------- OpenAI API Setup ----------
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
        # Quick check to see if key is likely valid (optional, remove if causing issues)
        # client.models.list()
        return client
    except Exception as e:
        st.sidebar.error(f"Error initializing OpenAI client: {e}. Check API key and permissions.")
        return None

# ---------- Utility Function ----------
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# ---------- OpenAI Analysis & Classification Function ----------
def analyze_and_classify_thumbnail(client: OpenAI, image_bytes: bytes):
    if not client:
        return "Uncategorized", "OpenAI client not initialized."

    base64_image = encode_image(image_bytes)
    image_data_uri = f"data:image/jpeg;base64,{base64_image}" # Assumes JPEG conversion happened before

    # Define categories clearly for the prompt
    category_definitions_text = """
    1.  **Text-Dominant:** Large, bold typography is the main focus, covering a significant portion. Minimal imagery.
    2.  **Minimalist / Clean:** Uncluttered, simple background, limited colors, clean font, few elements, lots of negative space.
    3.  **Face-Focused:** A close-up, expressive human face is the largest or most central element.
    4.  **Before & After:** Clearly divided layout showing two distinct states side-by-side.
    5.  **Comparison / Versus:** Layout structured (often split-screen) comparing items/ideas.
    6.  **Collage / Multi-Image:** Composed of multiple distinct images arranged together.
    7.  **Image-Focused:** A single, high-quality photo/illustration is dominant; text is secondary.
    8.  **Branded:** Most defining trait is prominent, consistent channel logos, colors, or fonts (use if branding is stronger than other elements).
    9.  **Curiosity Gap / Intrigue:** Deliberately obscures info (blurring, question marks, arrows).
    10. **Other / Unclear:** Doesn't fit well into other categories or mixes styles heavily.
    """
    valid_categories = {
        "Text-Dominant", "Minimalist / Clean", "Face-Focused",
        "Before & After", "Comparison / Versus", "Collage / Multi-Image",
        "Image-Focused", "Branded", "Curiosity Gap / Intrigue", "Other / Unclear"
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Ensure this model is available to your API key
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert analyst of YouTube thumbnail visual styles. Analyze the provided thumbnail image and classify it into ONE category based on its MOST dominant visual style/layout feature, using the provided definitions. Respond ONLY in the specific format 'Label: <Category Name>\\nReason: <Brief one or two sentence reason>'."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Classify this thumbnail using ONLY these definitions:\n{category_definitions_text}\n\nStrictly follow the output format:\nLabel: <Category Name>\nReason: <Your brief reason>"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri, "detail": "low"}
                        }
                    ]
                }
            ],
            temperature=0.2, # Lower temperature for more deterministic category output
            max_tokens=100
        )
        result = response.choices[0].message.content.strip()

    except Exception as e:
        # Check for specific OpenAI errors if needed (e.g., authentication, rate limits)
        st.error(f"Error during OpenAI analysis: {e}")
        result = "Label: Uncategorized\nReason: Analysis failed due to an API error."

    # Parse the output carefully
    label = "Uncategorized"
    reason = "Analysis could not determine a category or reason."
    try:
        lines = result.split('\n') # Split strictly by newline
        if len(lines) >= 1 and lines[0].startswith("Label:"):
            label_candidate = lines[0].replace("Label:", "").strip()
            # Validate against known categories (case-insensitive check first)
            found = False
            for cat in valid_categories:
                 if cat.lower() == label_candidate.lower():
                     label = cat # Use the official casing
                     found = True
                     break
            if not found:
                 # Optional: Add a log or warning if model returns unexpected category
                 st.warning(f"AI returned unrecognized category: '{label_candidate}'. Classifying as 'Other / Unclear'.")
                 label = "Other / Unclear"

        # Look for Reason specifically on the next line if available
        if len(lines) >= 2 and lines[1].startswith("Reason:"):
            reason = lines[1].replace("Reason:", "").strip()
        # If reason wasn't found on line 2, but we got a valid label, provide default reason
        elif label not in ["Uncategorized", "Other / Unclear"]:
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
        st.toast(f"Thumbnail added to '{label}' category!", icon="✅")
    else:
        # Error message is shown by store_thumbnail_record
         st.toast(f"Failed to add thumbnail to library.", icon="❌")


# ---------- Upload and Process Function ----------
def upload_and_process(client: OpenAI):
    st.header("Upload & Analyze Thumbnails")
    st.info("Upload up to 10 thumbnail images (JPG, JPEG, PNG). Click '✅ Add to Library' to save the analysis.")
    uploaded_files = st.file_uploader(
        "Choose thumbnail images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("Please upload a maximum of 10 images at once.")
            uploaded_files = uploaded_files[:10] # Process only the first 10

        num_columns = 3 # Adjust number of columns for display
        cols = st.columns(num_columns)
        col_index = 0

        # Initialize session state keys for added status if they don't exist
        for uploaded_file in uploaded_files:
             file_id = uploaded_file.id if hasattr(uploaded_file, 'id') else uploaded_file.name
             if f'added_{file_id}' not in st.session_state:
                 st.session_state[f'added_{file_id}'] = False

        for uploaded_file in uploaded_files:
            file_id = uploaded_file.id if hasattr(uploaded_file, 'id') else uploaded_file.name

            with cols[col_index % num_columns]:
                # Use a container for each item to group elements better
                with st.container():
                    st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
                    try:
                        image_bytes = uploaded_file.getvalue()
                        display_image = Image.open(io.BytesIO(image_bytes))

                        st.image(display_image, caption=f"{uploaded_file.name}", use_container_width=True)

                        # Prepare bytes for analysis/storage (convert to JPEG)
                        img_byte_arr = io.BytesIO()
                        processed_image = display_image
                        if processed_image.mode in ('RGBA', 'LA') or (processed_image.mode == 'P' and 'transparency' in processed_image.info):
                             processed_image = processed_image.convert('RGB')
                        processed_image.save(img_byte_arr, format='JPEG', quality=85)
                        processed_image_bytes = img_byte_arr.getvalue()

                        # --- Analysis Section ---
                        analysis_placeholder = st.empty() # Placeholder for results + button

                        with st.spinner(f"Analyzing {uploaded_file.name}..."):
                            label, reason = analyze_and_classify_thumbnail(client, processed_image_bytes)

                        # Populate the placeholder
                        with analysis_placeholder.container():
                            st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                            st.markdown(f"**Suggested Category:** `{label}`")
                            st.markdown(f"**Reason:** _{reason}_")

                            # --- Add to Library Button ---
                            is_added = st.session_state[f'added_{file_id}'] # Get current status

                            st.button(
                                "✅ Add to Library" if not is_added else "✔️ Added",
                                key=f'btn_add_{file_id}',
                                on_click=add_to_library_callback,
                                args=(file_id, processed_image_bytes, label, reason),
                                disabled=is_added or label == "Uncategorized" or not label # Disable if added or uncategorized
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                    finally:
                         st.markdown('</div>', unsafe_allow_html=True) # Close thumbnail-container

            col_index += 1

# ---------- Function to create Zip File in Memory ----------
def create_zip_in_memory(records):
    zip_buffer = io.BytesIO()
    filenames_in_zip = set() # To prevent duplicate names if timestamps are identical

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for record in records:
            rec_id = record['id']
            image_blob = record['image']
            label = record['label'].replace('/', '_').replace(' ', '_')
            # Handle potential None or invalid timestamp strings
            created_at_str = record['created_at'] if record['created_at'] else datetime.now().isoformat()
            try:
                 # Try parsing with microseconds first, then without
                 try:
                     timestamp_dt = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S.%f')
                 except ValueError:
                     timestamp_dt = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')
                 timestamp_str_fmt = timestamp_dt.strftime('%Y%m%d_%H%M%S')
            except (ValueError, TypeError):
                 timestamp_str_fmt = "unknown_time" # Fallback

            # Create a unique filename
            base_filename = f"{label}_{rec_id}_{timestamp_str_fmt}"
            filename_in_zip = f"{base_filename}.jpg"
            counter = 1
            while filename_in_zip in filenames_in_zip: # Ensure uniqueness
                 filename_in_zip = f"{base_filename}_{counter}.jpg"
                 counter += 1
            filenames_in_zip.add(filename_in_zip)

            try:
                zip_file.writestr(filename_in_zip, image_blob)
            except Exception as zip_err:
                st.warning(f"Could not add image ID {rec_id} to zip: {zip_err}")

    zip_buffer.seek(0)
    return zip_buffer


# ---------- Library Explorer ----------
def library_explorer():
    st.header("Thumbnail Library Explorer")
    st.markdown("Browse thumbnails saved to the library by category, and download categories as Zip files.")
    labels = get_labels()
    if not labels:
        st.info("The library is empty. Upload, analyze, and add some thumbnails first!")
        return

    if "selected_label" not in st.session_state:
        st.session_state.selected_label = None

    # Category Selection Grid
    if st.session_state.selected_label is None:
        st.markdown("### Select a Category to View")
        cols_per_row = 4 # Adjust as needed
        num_labels = len(labels)
        num_rows = (num_labels + cols_per_row - 1) // cols_per_row

        for i in range(num_rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < num_labels:
                    label = labels[idx]
                    if cols[j].button(label, key=f"btn_{label.replace(' ', '_')}", use_container_width=True):
                        st.session_state.selected_label = label
                        st.rerun()
    else:
        # Display Selected Category and Thumbnails
        selected_category = st.session_state.selected_label
        st.markdown(f"### Category: **{selected_category}**")

        # --- Top Bar: Back Button and Download Button ---
        top_cols = st.columns([0.3, 0.7]) # Adjust ratio as needed
        with top_cols[0]:
             if st.button("⬅️ Back to Categories", key="back_button", use_container_width=True):
                 st.session_state.selected_label = None
                 st.rerun()

        records = get_records_by_label(selected_category)

        if records:
            with top_cols[1]: # Place download button next to back button
                 zip_buffer = create_zip_in_memory(records)
                 st.download_button(
                     label=f"⬇️ Download {selected_category} ({len(records)}) as Zip",
                     data=zip_buffer,
                     file_name=f"{selected_category.replace('/', '_').replace(' ', '_')}_thumbnails.zip",
                     mime="application/zip",
                     key=f"download_{selected_category.replace(' ', '_')}",
                     use_container_width=True # Make button fill column width
                 )

            # --- Thumbnail Display Grid ---
            st.markdown("---") # Separator
            num_records = len(records)
            cols_per_row_thumbs = 4 # Adjust number of thumbnails per row
            thumb_cols = st.columns(cols_per_row_thumbs)
            col_idx = 0

            for record in records:
                with thumb_cols[col_idx % cols_per_row_thumbs]:
                     rec_id = record['id']
                     image_blob = record['image']
                     label = record['label']
                     reason = record['reason']
                     created_at_str = record['created_at']
                     try:
                        # Try parsing with microseconds first, then without
                        try:
                             created_at = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S.%f')
                        except ValueError:
                             created_at = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')
                        display_time = created_at.strftime('%Y-%m-%d %H:%M')
                     except (ValueError, TypeError):
                         display_time = created_at_str # Fallback

                     # Use a container for better spacing within columns
                     with st.container():
                         st.markdown('<div class="db-thumbnail-container">', unsafe_allow_html=True)
                         try:
                             image = Image.open(io.BytesIO(image_blob))
                             st.image(image, caption=f"ID: {rec_id} | {display_time}", use_container_width=True)
                             # Use st.expander for reason to save space initially
                             with st.expander("Show Reason"):
                                 st.markdown(f"_{reason}_")
                         except Exception as img_err:
                              st.warning(f"Could not load image for ID {rec_id}: {img_err}")
                         st.markdown('</div>', unsafe_allow_html=True)
                col_idx += 1
        else:
            st.info(f"No thumbnails found for the category: '{selected_category}'.")


# ---------- Main App ----------
def main():
    init_db() # Initialize the database schema if it doesn't exist

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

        # Setup OpenAI Client - place key input logically here
        client = setup_openai_client() # Initialize client

        # Main navigation
        menu = st.radio(
            "Navigation",
            ["Upload & Analyze", "Library Explorer"],
            key="nav_menu",
            label_visibility="collapsed" # Hide the "Navigation" label itself
        )
        st.markdown("---")
        st.info("Thumbnails added to the library are stored locally in a `thumbnails.db` file in the same directory as the script.")
        # Add credits or notes if desired
        st.caption(f"Using OpenAI model for analysis. Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Main Content Area ---
    if menu == "Upload & Analyze":
        if not client:
            st.error("❌ OpenAI client not initialized. Please provide a valid API key in the sidebar.")
        else:
            upload_and_process(client)
    elif menu == "Library Explorer":
        library_explorer()

if __name__ == "__main__":
    # Initialize session state keys if they don't exist
    if 'selected_label' not in st.session_state:
        st.session_state.selected_label = None
    # Other session state keys ('added_{file_id}') are created dynamically when needed
    main()
