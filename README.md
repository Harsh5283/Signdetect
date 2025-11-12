# 🧠 Sign Language Detection using OpenCV

## 🔍 Project Overview
This project performs *real-time sign language detection* using a webcam and *OpenCV*.  
It helps recognize hand gestures and convert them into corresponding sign labels (like A, B, C, etc.).  

The workflow includes:
- Collecting image data for each sign
- Training a deep learning model
- Testing it in real-time using a webcam

---

## 🚀 Features
- Real-time webcam gesture capture  
- Dataset creation tool using dataCollection.py  
- Model storage and management in Models/ folder  
- Easy integration for new gesture labels  
- Real-time prediction with confidence score display  

---

## 🗂 Project Structure
MINI project/
├─ DATA/
│ ├─ A/
│ ├─ B/
│ ├─ C/
│ └─ ... (Each folder contains images for one sign)
├─ Models/
│ └─ (Trained model files)
├─ dataCollection.py
├─ test.py
├─ hello.py
└─ Readme.md


---

## ⚙ Requirements
Install dependencies using pip:

```bash
pip install opencv-python numpy tensorflow scikit-learn matplotlib pillow

🧩 How It Works
⿡ Data Collection (dataCollection.py)


Opens the webcam using OpenCV (cv2.VideoCapture(0)).


Captures hand gesture images when you press a key.


Saves them in labeled folders under DATA/ (e.g., DATA/A, DATA/B, etc.).


You can capture multiple signs for each letter.


Example Command:
python dataCollection.py


⿢ Model Training (Optional - if not pre-trained)
If you don’t have a trained model, create one using a simple CNN or transfer learning.
Example structure (conceptual):
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

Save the model to the Models/ folder:
model.save('Models/sign_model.h5')


⿣ Real-Time Detection (test.py)


Loads the trained model (sign_model.h5).


Captures live video from webcam.


Preprocesses each frame (resize, normalize).


Predicts the sign and displays the result on screen using cv2.putText.


Example Command:
python test.py

You can press q to quit the webcam window.

🧠 Codebase Explanation
dataCollection.py


Captures live frames from your webcam.


Saves each captured frame into a folder (DATA/A, DATA/B, etc.).


Used for building a labeled dataset.


test.py


Loads the trained CNN model.


Reads video frames in real time.


Detects the region of interest (hand area).


Predicts the gesture and displays it live.


hello.py


Helper/test script (e.g., to test camera, print system info, etc.).


You can use it for debugging or simple model tests.



⚡ Quick Commands (Windows PowerShell)
# Activate virtual environment (optional)
.\venv\Scripts\Activate

# Initialize git (if not already)
git init

# Run data collection
python dataCollection.py

# Train model (if train.py exists)
python train.py

# Test model in real time
python test.py


🧾 Troubleshooting
❌ git : The term 'git' is not recognized
✅ Solution:
Install Git using Winget:
winget install --id Git.Git -e --source winget

Then restart VS Code or your terminal.

⚠ f2py.exe not on PATH
This is a harmless warning from Python — you can ignore it or add this folder to PATH:
C:\Users\<YourUser>\AppData\Roaming\Python\Python3x\Scripts


🧠 Tips for Better Accuracy


Capture diverse samples (lighting, angles, background).


Use at least 200–500 images per class.


Normalize input images before training.


Use data augmentation for better generalization.


Try transfer learning (MobileNetV2 / ResNet50).



🧰 Future Enhancements


Add more signs (A–Z, 0–9, common gestures).


Build GUI using Tkinter / Streamlit.


Integrate text-to-speech output for detected signs.


Computer-Network Project by:
Shailendra Mani Pandey, Suyash Shukla, Vaibhav Singh
🎓 CSE Student | Developer | Tech Enthusiast
