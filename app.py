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
    final_prompt TEXT
);
""")
conn.commit()

def save_analysis_to_db(video_id, source, openai_description, vision_result, final_analysis, final_prompt):
    timestamp = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO thumbnail_analysis (timestamp, video_id, source, openai_description, vision_result, final_analysis, final_prompt)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        video_id,
        source,
        openai_description,
        json.dumps(vision_result) if vision_result else None,
        final_analysis,
        final_prompt
    ))
    conn.commit()

# Main Streamlit app
st.title("🎬 YouTube Thumbnail Analyzer with AI & SQLite")

# Setup API clients
vision_client = None
openai_client = None

if 'GOOGLE_CREDENTIALS' in st.secrets:
    credentials_dict = st.secrets["GOOGLE_CREDENTIALS"]
    if isinstance(credentials_dict, str):
        credentials_dict = json.loads(credentials_dict)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)

openai.api_key = st.secrets.get("OPENAI_API_KEY")
openai_client = openai

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

    # Analyze image with OpenAI Vision (basic simulation)
    base64_img = base64.b64encode(image_bytes).decode("utf-8")
    openai_desc = openai_client.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this thumbnail in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]}
        ]
    )["choices"][0]["message"]["content"]

    # Analyze with Vision API
    vision_result = {}
    if vision_client:
        image_v = vision.Image(content=image_bytes)
        label_response = vision_client.label_detection(image=image_v)
        vision_result["labels"] = [label.description for label in label_response.label_annotations]

    # Generate summary analysis
    messages = [
        {"role": "system", "content": "You're a thumbnail analysis expert."},
        {"role": "user", "content": f"Analyze this thumbnail based on Vision AI: {json.dumps(vision_result)} and your description: {openai_desc}"}
    ]
    final_analysis = openai_client.ChatCompletion.create(model="gpt-4o", messages=messages)["choices"][0]["message"]["content"]

    # Generate prompt paragraph
    prompt_message = [
        {"role": "system", "content": "You're an AI prompt designer."},
        {"role": "user", "content": f"Create a detailed prompt to recreate this thumbnail based on: {final_analysis}"}
    ]
    final_prompt = openai_client.ChatCompletion.create(model="gpt-4o", messages=prompt_message)["choices"][0]["message"]["content"]

    st.subheader("🧠 Final Analysis")
    st.markdown(final_analysis)

    st.subheader("🖊️ Prompt to Recreate Thumbnail")
    st.text_area("Prompt", final_prompt, height=150)

    # Save everything to SQLite
    save_analysis_to_db(
        video_id=video_info.get("id", "uploaded_image"),
        source=input_option,
        openai_description=openai_desc,
        vision_result=vision_result,
        final_analysis=final_analysis,
        final_prompt=final_prompt
    )

# View Past Analyses
with st.expander("📜 View Past Analyses"):
    cursor.execute("SELECT timestamp, video_id, source, final_prompt FROM thumbnail_analysis ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    for row in rows:
        st.markdown(f"**Date**: {row[0]}  |  **Video ID**: {row[1]}  |  **Source**: {row[2]}")
        st.markdown(f"**Prompt**: {row[3][:200]}...")
        st.markdown("---")
