# Portrait Quality Enhancement with Streamlit  
Link - [Portrait enhancer](https://portraitenhancer.streamlit.app/)

This project is a web-based application designed to enhance the quality of portrait images using GFPGAN. The app is built with Streamlit and provides an easy-to-use interface for improving image sharpness, reducing noise, and restoring old photos. 

---

## Features  

- **Upload Images:** Upload portrait images directly from your device.  
- **Face Restoration with GFPGAN:** Automatically enhance image quality using GFPGAN.  
- **Preview Results:** Instantly preview the enhanced image in the app.  
- **Download Enhanced Images:** Save the restored images to your device.  
---
## Installation  

Follow these steps to run the project locally:
1. Clone the repository: 
```bash  
   git clone https://github.com/Zakmar2004/Super_resolution_project.git  
   cd Super_resolution_project
```
2. Install dependencies:
```bash
   pip install -r requirements.txt  
```
3. Run the Streamlit app:
```bash
   streamlit run main.py  
```
---
## Structure  

- **`enhance_with_gfpgan` function:**  
  Processes an input image and generates an enhanced version using the GFPGAN model.  

- **`main.py`:**  
  The main script creates a Streamlit-based web application, integrating the enhancement functionality and providing an interactive user interface.  
---
## Acknowledgments

This project integrates the GFPGAN framework developed by Tencent ARC Lab. For more details, visit the [GFPGAN repository](https://github.com/TencentARC/GFPGAN).