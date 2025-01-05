import os
import cv2
from PIL import Image
from gfpgan import GFPGANer
import logging
import urllib.request
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

realesrgan_weights_path = "gfpgan/weights/RealESRGAN_x2plus.pth"
realesrgan_download_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

if not os.path.exists(realesrgan_weights_path):
    logging.info(f"Realesrgan not found. Downloading to {realesrgan_weights_path}...")
    try:
        urllib.request.urlretrieve(realesrgan_download_url, realesrgan_weights_path)
        logging.info(f"Realesrgan downloaded successfully to {realesrgan_weights_path}.")
    except Exception as e:
        logging.error(f"Failed to download Realesrgan: {e}")
else:
    logging.info(f"Realesrgan found at: {realesrgan_weights_path}")

def enhance_with_gfpgan(input_image, upscale=2):
    try:
        logging.info("Initializing RealESRGAN model...")
        realesrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path=realesrgan_weights_path,
            model=realesrgan_model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=False)
        logging.info("RealESRGAN model initialized successfully")

        version = '1.4'
        arch = 'clean'
        channel_multiplier = 2
        model_path = os.path.abspath("gfpgan/weights/GFPGANv1.4.pth")

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

        logging.info(f"Checking input image path: {input_image}")
        if not os.path.exists(input_image):
            logging.error(f"Input image path does not exist: {input_image}")
            return None

        logging.info(f"Reading input image: {input_image}")
        input_img = cv2.imread(input_image, cv2.IMREAD_COLOR)
        if input_img is None:
            logging.error(f"Failed to read the image: {input_image}")
            return None

        logging.info(f"Enhancing image with upscale factor {upscale}")
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5
        )

        if restored_img is not None:
            logging.info("Enhancement process completed successfully")
            enhanced_image = Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
            return enhanced_image
        else:
            logging.warning("Restored image is None, enhancement failed")
            return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        return None







