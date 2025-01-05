import os
import urllib.request
import logging
import streamlit as st
from PIL import Image
from gfpgan_inference import enhance_with_gfpgan
import io

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

weights_path = "gfpgan/weights/GFPGANv1.4.pth"
download_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

if not os.path.exists(weights_path):
    logging.info(f"Model not found. Downloading to {weights_path}...")
    try:
        urllib.request.urlretrieve(download_url, weights_path)
        logging.info(f"Model downloaded successfully to {weights_path}.")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
else:
    logging.info(f"Model found at: {weights_path}")

st.title("Portrait Enhancer")
st.write("Enhance face image resolution using GFPGAN!")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
rescaling_factor = st.selectbox("Choose rescaling factor (upscale)", [1, 2, 3, 4], index=1)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Enhance Image"):
        with st.spinner("Enhancing image..."):
            try:
                enhanced_image = enhance_with_gfpgan(image, upscale=rescaling_factor)
                if enhanced_image:
                    logging.info("Image enhancement successful.")
                    st.image(enhanced_image, caption="Enhanced Image", use_container_width=True)

                    img_byte_arr = io.BytesIO()
                    enhanced_image.save(img_byte_arr, format="JPEG", quality=95)
                    img_byte_arr = img_byte_arr.getvalue()

                    st.download_button(
                        label="Download Enhanced Image",
                        data=img_byte_arr,
                        file_name="enhanced_image.jpg",
                        mime="image/jpeg"
                    )
                else:
                    logging.warning("Image enhancement failed.")
                    st.error("Image enhancement failed. Please try again.")
            except Exception as e:
                logging.error(f"Error during image enhancement: {e}")
                st.error(f"An error occurred during the enhancement process: {e}")






