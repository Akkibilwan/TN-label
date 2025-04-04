import streamlit as st
import os
import io
import json
import re
import requests
import sqlite3
from datetime import datetime
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import openai
import base64

# Initialize SQLite
conn = sqlite3.connect("thumbnail_analysis.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS thumbnail_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    video_id TEXT,
    source TEXT,
    openai_description TEXT,
    vision_result TEXT,
    final_analysis TEXT,
    final_prompt TEXT,
    categories TEXT
);
""")
conn.commit()

def save_analysis_to_db(video_id, source, openai_description, vision_result, final_analysis, final_prompt, categories):
    timestamp = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO thumbnail_analysis (timestamp, video_id, source, openai_description, vision_result, final_analysis, final_prompt, categories)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        video_id,
        source,
        openai_description,
        json.dumps(vision_result) if vision_result else None,
        final_analysis,
        final_prompt,
        ", ".join(categories)
    ))
    conn.commit()

# Streamlit UI
st.title("🎬 YouTube Thumbnail Analyzer with AI & SQLite")

# Google Vision Setup
vision_client = None
if 'GOOGLE_CREDENTIALS' in st.secrets:
    credentials_dict = st.secrets["GOOGLE_CREDENTIALS"]
    if isinstance(credentials_dict, str):
        credentials_dict = json.loads(credentials_dict)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)

# OpenAI setup (v1.x syntax)
openai.api_key = st.secrets.get("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=openai.api_key)

# Choose input method
input_option = st.radio("Select input method:", ["Upload Image", "YouTube URL"], horizontal=True)

image_bytes = None
image = None
video_info = {"id": "uploaded_image", "url": None, "title": "Uploaded Thumbnail"}

if input_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a thumbnail image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
        image_bytes = img_byte_arr.getvalue()
else:
    youtube_url = st.text_input("Enter YouTube video URL:")
    if youtube_url:
        video_id_match = re.search(r"(?<=v=)[^&#]+", youtube_url)
        if video_id_match:
            video_id = video_id_match.group(0)
            video_info = {
                "id": video_id,
                "url": youtube_url,
                "title": f"Thumbnail for {video_id}"
            }
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            resp = requests.get(thumbnail_url)
            if resp.status_code == 200:
                image_bytes = resp.content
                image = Image.open(io.BytesIO(image_bytes))

if image_bytes and image:
    st.image(image, caption=video_info["title"], use_column_width=True)

    # Analyze image with OpenAI (v1.x)
    base64_img = base64.b64encode(image_bytes).decode("utf-8")
    openai_desc = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this thumbnail in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }
        ]
    ).choices[0].message.content

    # Google Vision Analysis
    vision_result = {}
    if vision_client:
        image_v = vision.Image(content=image_bytes)
        label_response = vision_client.label_detection(image=image_v)
        vision_result["labels"] = [label.description for label in label_response.label_annotations]

    # Generate structured analysis
    final_analysis = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You're a thumbnail analysis expert."},
            {"role": "user", "content": f"Analyze this thumbnail based on Vision AI: {json.dumps(vision_result)} and OpenAI vision description: {openai_desc}"}
        ]
    ).choices[0].message.content

    # Generate detailed prompt
    final_prompt = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You're a creative prompt engineer."},
            {"role": "user", "content": f"Write a detailed DALL-E prompt to recreate this thumbnail: {final_analysis}"}
        ]
    ).choices[0].message.content

    # Categorize thumbnail
    category_response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You're an expert in categorizing YouTube thumbnails."},
            {"role": "user", "content": f"From the following analysis, return 3 high-level thumbnail categories this image fits into: {final_analysis}. Only return a JSON list of 3 category names."}
        ]
    ).choices[0].message.content

    try:
        categories = json.loads(category_response)
    except:
        categories = ["Uncategorized"]

    st.subheader("🧠 Final Analysis")
    st.markdown(final_analysis)

    st.subheader("🏷️ Categories")
    st.write(categories)

    st.subheader("🖊️ Prompt to Recreate Thumbnail")
    st.text_area("Prompt", final_prompt, height=150)

    save_analysis_to_db(
        video_id=video_info.get("id", "uploaded_image"),
        source=input_option,
        openai_description=openai_desc,
        vision_result=vision_result,
        final_analysis=final_analysis,
        final_prompt=final_prompt,
        categories=categories
    )

# History tab
with st.expander("📜 View Past Analyses"):
    cursor.execute("SELECT timestamp, video_id, source, final_prompt, categories FROM thumbnail_analysis ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    for row in rows:
        st.markdown(f"**Date**: {row[0]}  |  **Video ID**: {row[1]}  |  **Source**: {row[2]}")
        st.markdown(f"**Categories**: {row[4]}")
        st.markdown(f"**Prompt**: {row[3][:200]}...")
        st.markdown("---")

# Browse by Category
with st.expander("🗂️ Browse by Category"):
    cursor.execute("SELECT DISTINCT categories FROM thumbnail_analysis")
    all_categories = sorted(set(
        cat.strip()
        for row in cursor.fetchall()
        for cat in row[0].split(',') if cat.strip()
    ))

    selected_category = st.selectbox("Select a category", all_categories)
    if selected_category:
        cursor.execute("SELECT timestamp, video_id, final_prompt FROM thumbnail_analysis WHERE categories LIKE ? ORDER BY id DESC", (f"%{selected_category}%",))
        filtered = cursor.fetchall()
        for f in filtered:
            st.markdown(f"**{f[0]}** | **Video ID**: {f[1]}")
            st.markdown(f"Prompt: {f[2][:200]}...")
            st.markdown("---")
