# 🛰️ Satellite Image Segmentation

## 📝 Project Overview
This repository contains an end-to-end machine learning project for segmenting water bodies from satellite images. The project demonstrates a full development cycle, from data preprocessing and model experimentation to deployment as a web application using Flask. The project is divided into three main phases:

1. **From Scratch Model**: Custom segmentation model using **TensorFlow**.  
2. **Pretrained Model**: Utilizing a pretrained **UNet** model with a **MobileNetV2** encoder, developed with **PyTorch** and the `segmentation-models-pytorch` library.  
3. **Deployment**: **Flask** web application to serve the best-performing model, allowing users to upload satellite images and receive water segmentation masks.

---

## ✨ Key Features

### Custom Data preprocessing
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
Make sure you have Python 3.8+ installed, then run:
```
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```
streamlit run app.py
```

---


