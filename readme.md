#  AncientEye – Ancient Egypt Monuments Classifier

A deep learning project to **classify Ancient Egyptian monuments** using a custom **Convolutional Neural Network (CNN)**.  
The model is trained on a dataset of statues, pyramids, temples, and other famous monuments.

---

##  Features
- Classifies **21 categories** of Ancient Egyptian monuments.
- **Streamlit web app** for easy image upload and predictions.
- Provides **historical information** about each classified monument.
- Built with **TensorFlow / Keras** from scratch.

---

## 📂 Project Structure
AncientEye/
│── train.py # Model training script
│── app.py # Streamlit app
│── models/
│ ├── egypt_cnn.h5 # Trained model (not uploaded due to size)
│ └── class_indices.json# Class index mapping
│── requirements.txt
│── README.md


---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mahmoudelgendy001/AncientEye.git
   cd AncientEye
Create a virtual environment & install dependencies:

## Create a virtual environment & install dependencies:

python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
🗂️ Dataset
We used the Ancient Egypt Monuments Classification Dataset.
Download it here:
👉 [Ancient Egypt Monuments Classification Dataset](https://www.kaggle.com/datasets/monaabdelrazek/finaldataset)

After download, place the folders inside data/:

data/
├── train/
├── val/
└── test/

#  Training the Model
Run the training script:

python train.py
This will:

Train the CNN model.

Save best weights to models/egypt_cnn.h5.

Save class mappings in models/class_indices.json.

# 🚀Running the Streamlit App
After training (or downloading the model):


streamlit run app.py
Upload an image of a monument.

Get the predicted class with confidence score.

See a short historical description of the monument.

##  Notes
The dataset must be downloaded manually from Kaggle (link above).

The trained model (egypt_cnn.h5) is not uploaded (file size >100MB). You can train it locally or host it externally (Google Drive, Hugging Face, etc.).

The app supports Arabic historical descriptions.

#  Author
Mahmoud Elgendy

GitHub: Mahmoudelgendy001