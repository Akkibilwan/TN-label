import streamlit as st
from openai import OpenAI
import base64
import io
import os

# --- Configuration ---
# It's better to use environment variables or Streamlit secrets for API keys
# For local testing, using text_input is okay, but less secure.
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE" # Or set it in your environment

# --- Category Definitions (as provided by user, with additions) ---
CATEGORY_DEFINITIONS = """
* **Text-Dominant:** Large, bold typography is the main focus, covering a significant portion of the thumbnail. Minimal imagery or image is secondary. Goal: Grab attention with a compelling text hook.
* **Minimalist / Clean:** Uncluttered impression, simple background, limited color palette, clean font, few elements, lots of negative space. Goal: Appear modern, professional.
* **Face-Focused:** A close-up of a person's face showing clear emotion is the largest or most central element. Goal: Create human connection, convey emotion.
* **Before & After:** Clearly divided layout (usually vertically) showing two distinct states of the same subject/scene side-by-side. Goal: Demonstrate transformation or results.
* **Comparison / Versus:** Layout clearly structured (often split-screen) to visually place two or more competing items/ideas against each other. Goal: Highlight differences or choices.
* **Collage / Multi-Image:** Composed of multiple distinct images arranged together (grid, overlapping etc.). Goal: Hint at variety of content within the video.
* **Image-Focused:** A single, high-quality photograph, illustration, or graphic is the dominant element, carrying the visual appeal. Text is minimal/secondary. Goal: Impress with strong visuals.
* **Branded:** The *most defining characteristic* is the consistent and prominent use of specific channel logos, color schemes, or unique font styles that make it instantly recognizable *primarily* due to branding elements. Goal: Build brand recognition.
* **Curiosity Gap / Intrigue:** Deliberately obscures information using blurring, question marks, arrows pointing to something hidden, or incomplete visuals. Goal: Make the viewer click to find out more.
* **Other / Unclear:** Does not fit well into any of the above categories or combines multiple styles without a single dominant one.
"""

# --- OpenAI API Call Function ---
def get_thumbnail_category(api_key: str, image_bytes: bytes) -> str:
    """
    Analyzes the thumbnail image using OpenAI multimodal model and returns the category.
    """
    try:
        client = OpenAI(api_key=api_key)

        # Encode image to Base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_data_uri = f"data:image/jpeg;base64,{base64_image}" # Assuming jpeg, adjust if needed

        prompt_messages = [
            {
                "role": "system",
                "content": "You are an expert visual analyst specializing in YouTube thumbnail styles. Your task is to analyze the provided thumbnail image and classify it into ONE of the following visual style/layout categories based on its most dominant features. Respond with only the single category name and nothing else."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please classify the following thumbnail into one category based on these definitions:\n\n{CATEGORY_DEFINITIONS}\n\nRespond with only the single category name."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                           "url": image_data_uri
                        }
                    }
                ]
            }
        ]

        # Use the appropriate multimodal model name (check OpenAI documentation for the latest)
        # Common options include "gpt-4-vision-preview", "gpt-4o", or newer versions.
        # Using "gpt-4o" as a likely candidate for strong vision capabilities.
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=prompt_messages,
            max_tokens=50 # Keep response short, just the category name
        )

        category = response.choices[0].message.content.strip()
        # Basic validation to ensure it's one of the expected categories (optional but good)
        defined_categories = [
            "Text-Dominant", "Minimalist / Clean", "Face-Focused",
            "Before & After", "Comparison / Versus", "Collage / Multi-Image",
            "Image-Focused", "Branded", "Curiosity Gap / Intrigue", "Other / Unclear"
            ]
        # Simple check if the response *contains* a known category name
        found_category = "Other / Unclear" # Default
        for defined_cat in defined_categories:
            if defined_cat.lower() in category.lower():
                 # Return the official category name for consistency
                found_category = defined_cat
                break
        return found_category

    except Exception as e:
        st.error(f"An error occurred while contacting OpenAI: {e}")
        return "Error during analysis"

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("🖼️ Thumbnail Visual Style Categorizer (using OpenAI)")
st.markdown("Upload a thumbnail image, and OpenAI will categorize its visual style based on predefined definitions.")

st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.header("Category Definitions")
st.sidebar.markdown(CATEGORY_DEFINITIONS.replace("*","").replace("\n\n","\n").replace(":","**:\n")) # Format for sidebar display

uploaded_file = st.file_uploader("Choose a thumbnail image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = io.BytesIO(uploaded_file.getvalue())
    st.image(image, caption=f"Uploaded Thumbnail: {uploaded_file.name}", width=300)

    if api_key:
        st.markdown("---")
        st.subheader("Analysis Result")
        with st.spinner("🧠 Analyzing thumbnail with OpenAI..."):
            # Read image bytes for API call
            image_bytes_for_api = uploaded_file.getvalue()
            category_result = get_thumbnail_category(api_key, image_bytes_for_api)

            if "Error" not in category_result:
                st.success(f"Predicted Category: **{category_result}**")
            else:
                st.error(category_result) # Display the error message from the function

    elif not api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar to enable analysis.")

else:
    st.info("Upload a thumbnail image to begin.")

st.markdown("---")
st.caption("Note: Analysis relies on OpenAI's multimodal understanding. Ensure you are using an appropriate model like GPT-4o or GPT-4V. Accuracy depends on the model's interpretation and the clarity of category definitions.")
