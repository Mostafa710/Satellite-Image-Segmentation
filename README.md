# 🛰️ Satellite Image Segmentation

## 📝 Project Overview
This repository contains an end-to-end machine learning project for segmenting water bodies from satellite images. The project demonstrates a full development cycle, from data preprocessing and model experimentation to deployment as a web application using Flask. The project is divided into three main phases:

1. **From Scratch Model**: Custom segmentation model using **TensorFlow**.  
2. **Pretrained Model**: Utilizing a pretrained **UNet** model with a **MobileNetV2** encoder, developed with **PyTorch** and the `segmentation-models-pytorch` library.  
3. **Deployment**: **Flask** web application to serve the best-performing model, allowing users to upload satellite images and receive water segmentation masks.

---

## ✨ Key Features

### Custom Data Preprocessing
The preprocessing pipeline handles 12-channel satellite images and applies data augmentation.

### Advanced Feature Engineering
New features, including the **Normalized Difference Water Index (NDWI)** and **Normalized Difference Vegetation Index (NDVI)**, are generated to improve model accuracy.

### Model Optimization
The pipeline removes redundant or less important channels (such as Blue, Green, and Red) to focus on more informative spectral bands for water detection.

### Efficient Deployment
The final model is deployed via a lightweight Flask API, making it easy to use and showcase. The web application is configured to accept satellite images with the `.tif` and `.tiff` extensions for segmentation.

---

## 📂 Project Structure

```
├── app.py                     # The Flask application to serve the model
├── model/                     # Directory for the trained model file
│   └── UNet_with_MobileNetV2_backbone.pth
├── static/                    # Static files for the web interface
│   ├── script.js
│   └── style.css
├── templates/                 # HTML templates
│   └── index.html
├── requirements.txt           # List of project dependencies
├── satellite-water-segmentation.ipynb               # Jupyter notebook for the "from scratch" model phase
├── satellite-water-segmentation-pre-trained.ipynb   # Jupyter notebook for the pretrained model phase
└── utils.py                   # Utility functions for data preprocessing and model inference
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```
git clone https://github.com/Mostafa710/Satellite-Image-Segmentation.git
cd Satellite-Image-Segmentation
```

### 2. Install dependencies
```
pip install -r requirements.txt
```
The dependencies include `flask`, `torch`, `torchvision`, `segmentation-models-pytorch`, `tifffile`, and `pillow`.

### 3. Running the application
To start the web application, simply run the `app.py` file from your terminal:
```
python app.py
```
The application will start, and you can access it by navigating to `http://127.0.0.1:5000` in your web browser. You will be able to upload a satellite image (`.tif` or `.tiff`), and the application will display the segmented water mask.

---

## 🖼️ Example

Upload a satellite image through the app interface and get:
- The original image (visualized as an RGB composite)
- A predicted water segmentation mask
<img width="1093" height="379" alt="Screenshot 2025-08-19 165525" src="https://github.com/user-attachments/assets/f9eda0b7-3422-4a7e-8072-810b7f538f78" />

---

## 💻 Tech Stack

- **Languages:** Python, HTML, CSS, JavaScript
- **Machine Learning:** TensorFlow, PyTorch, `segmentation-models-pytorch`, `torchvision`
- **Data Handling:** NumPy, Pillow, `tifffile`
- **Web Framework:** Flask

---

## 📬 Contact

For questions or collaboration, feel free to connect:

[LinkedIn](https://www.linkedin.com/in/mostafa-mamdouh-80b110228) | [Email](mailto:mostafamamdouh710@gmail.com)

---




