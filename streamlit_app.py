import streamlit as st
import requests
import os
from dotenv import load_dotenv
from PIL import Image
import openai

load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]


st.title("🧠 AI Product Image Generator")

# Upload or paste URL
img_source = st.radio("Choose image input method:", ["Upload", "Paste URL"])

img = None
if img_source == "Upload":
    uploaded = st.file_uploader("Upload product image", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        img_path = os.path.join("references", uploaded.name)
        img.save(img_path)
        st.image(img, caption="Uploaded Image", use_column_width=True)

elif img_source == "Paste URL":
    url = st.text_input("Paste image URL here")
    if url:
        try:
            response = requests.get(url)
            img = Image.open(response.raw)
            filename = url.split("/")[-1].split("?")[0]
            img_path = os.path.join("references", filename)
            img.save(img_path)
            st.image(img, caption="Downloaded Image", use_column_width=True)
        except:
            st.error("Invalid image URL")

# Prompt + Generate
prompt = st.text_area("Describe the image you'd like to create:", height=100)
num_outputs = st.slider("How many variations?", 1, 4, 1)

if st.button("Generate Image"):
    if prompt and img:
        with st.spinner("Generating..."):
            try:
                response = openai.Image.create(
                    prompt=prompt,
                    n=num_outputs,
                    size="1024x1024"
                )
                st.success("Done!")
                for i, data in enumerate(response['data']):
                    url = data['url']
                    st.image(url, caption=f"Variant {i+1}")
                    with open(f"outputs/output_{i+1}.txt", "w") as f:
                        f.write(url)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Upload image and enter prompt first.")
