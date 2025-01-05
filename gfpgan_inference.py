import os
import numpy as np
import cv2
from PIL import Image
from gfpgan import GFPGANer
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def enhance_with_gfpgan(image, upscale=2):
    try:
        version = '1.4'
        arch = 'clean'
        channel_multiplier = 2
        model_path = os.path.abspath("gfpgan/weights/GFPGANv1.4.pth")
        bg_upsampler = None

        logging.info(f"Checking GFPGAN model path: {model_path}")
        if not os.path.exists(model_path):
            logging.error(f"Model path does not exist: {model_path}")
            return None

        logging.info(f"Initializing GFPGAN model (version: {version}, arch: {arch})")
        restorer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler
        )
        logging.info("GFPGAN model initialized successfully")

        logging.info(f"Processing input image...")
        input_img = np.array(image)
        input_img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

        if input_img_bgr is None:
            logging.error(f"Failed to process the image.")
            return None

        logging.info(f"Enhancing image with upscale factor {upscale}")
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5
        )

        if restored_img is not None:
            logging.info("Enhancement process completed successfully")
            restored_img_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
            enhanced_image = Image.fromarray(restored_img_rgb)
            return enhanced_image
        else:
            logging.warning("Restored image is None, enhancement failed")
            return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        return None








