
# FloraVision: Deep Learning Image Classifier

FloraVision is a command-line tool built with **PyTorch** that leverages **Transfer Learning** to identify flower species. It automates the entire process: from downloading the raw dataset to training a deep neural network and performing real-time inference.



## 📁 Project Structure

* **`train.py`**: The primary script for training the neural network.
* **`predict.py`**: The interface for loading a saved model and classifying new images.
* **`model_functions.py`**: Contains the architecture logic and training loops.
* **`model_functions_for_prediction.py`**: Handles image preprocessing (normalization, resizing) for inference.
* **`Dataset/`**:
    * **`dataset.py`**: An automation script that downloads the 102-category flower dataset and creates the `cat_to_name.json` mapping.

---

## 🚀 Quick Start Guide

### 1. Initialize the Dataset
Before training, you must pull the data. This script downloads, extracts, and organizes the images automatically:
```bash
python Dataset/dataset.py

```

*This will create a `/flowers` directory in your root folder.*

### 2. Train the Neural Network

Train a model using a pre-trained architecture (like VGG16 or DenseNet).

```bash
python train.py ./flowers --arch "vgg16" --learning_rate 0.001 --hidden_units 512 --epochs 10 --gpu

```

*The script saves a `checkpoint.pth` file upon completion.*

### 3. Predict Flower Species

Use your trained checkpoint to identify an image file:

```bash
python predict.py path/to/image.jpg checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu

```

---

## 🛠️ Technical Requirements

The project requires the following Python libraries:

* `torch` & `torchvision`
* `PIL` (Pillow)
* `numpy`
* `matplotlib`
* `requests` (for dataset downloading)

## ✍️ Author

**Ghousiya Begum*

[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ghousiya-begum-a9b634258/)  [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ghousiya47)


