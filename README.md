# Emotion Analysis Project

This project predicts emotions from text using a Python deep learning model and a C++ client for inference.  
It is designed to be **easy to use** and **portable** for any user.

---

## 📁 Project Structure

```
Emotion Analysis/
├── python_ml_server/
│   ├── scripts/
│   │   ├── train.py         # Train the emotion classifier
│   │   └── app.py           # Streamlit web app for emotion prediction
│   └── model/               # Model files (created after training)
│       ├── model.keras
│       ├── tokenizer.pkl
│       ├── label_encoder.pkl
│       ├── saved_model/     # TensorFlow SavedModel for C++ client
│       ├── word_index.txt   # Vocabulary for C++ client
│       └── labels.txt       # Labels for C++ client
├── cpp_client/
│   ├── main.cpp
│   ├── TextPreprocessor.cpp
│   ├── TextPreprocessor.h
│   ├── LabelUtils.cpp
│   ├── LabelUtils.h
│   ├── CMakeLists.txt
│   ├── include/             # (Create this and add TensorFlow C headers here)
│   └── lib/                 # (Create this and add TensorFlow C library here)
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/emotion-analysis.git
cd "Emotion Analysis"
```

---

### 2. Python: Train & Run the Web App

#### a. Install Python dependencies

```sh
pip install -r requirements.txt
```
If `requirements.txt` is missing, install manually:
```sh
pip install tensorflow scikit-learn pandas numpy streamlit
```

#### b. Train the model (optional, skip if model files exist)

```sh
cd python_ml_server/scripts
python train.py
```

#### c. Run the Streamlit web app

```sh
streamlit run app.py
```
Open the link shown in your terminal to use the web interface.

---

### 3. C++ Client: Native Inference

#### a. Download TensorFlow C Library

- Go to: https://www.tensorflow.org/install/lang_c
- Download the C library for your OS (Windows/Linux/Mac).
- Extract:
  - If `cpp_client/include/` or `cpp_client/lib/` do not exist, create them.
  - Copy the `include` folder from the TensorFlow C download to `cpp_client/include`
  - Copy the `lib` folder (or just `tensorflow.dll`/`libtensorflow.so`) to `cpp_client/lib`

#### b. Build the C++ client

```sh
cd cpp_client
mkdir build
cd build
cmake ..
cmake --build .
```

#### c. Run the C++ client

```sh
./main.exe      # On Windows
./main          # On Linux/Mac
```
Enter your text when prompted and see the predicted emotion.

---

## 🛠️ Troubleshooting

- **TensorFlow include error in C++:**  
  Make sure you have copied the TensorFlow C headers and library to the correct folders as described above.
- **Model files missing:**  
  Run the Python training script first, or download the provided model files if available.
- **Python errors:**  
  Ensure all dependencies are installed and you are using Python 3.7+.

---

## 🧹 .gitignore Example

Add a `.gitignore` file to avoid pushing unnecessary files:

```
__pycache__/
*.pyc
build/
dist/
*.dll
*.so
*.dylib
.DS_Store
Thumbs.db
cpp_client/include/
cpp_client/lib/
```

---

## 📢 Credits

- Created by Sitanshu & Yash for a semester project.
- Uses TensorFlow, Streamlit, and C++.

---

## ❓ Need Help?

If you get stuck, read the error messages carefully and check the steps above.  
If you’re still lost, open an issue on GitHub or ask a friend!

---

**Now even a monkey can run this project! 🐒**