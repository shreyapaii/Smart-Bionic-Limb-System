# 🦾 AI-Based Low-Cost Bionic Limb Gesture Recognition System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Format](#dataset-format)
- [Model Training](#model-training)
- [Running the App](#running-the-app)
- [Screenshots](#screenshots)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements an **intelligent gesture recognition system** using EMG (Electromyography) muscle signals to predict hand gestures for controlling low-cost prosthetic and bionic limbs. The system combines Machine Learning with a modern healthcare-themed web interface to provide:

- Real-time gesture prediction from EMG signals
- Patient rehabilitation monitoring
- Progress tracking and analytics
- High-accuracy classification using Random Forest

### Problem Statement

Thousands of civilians and soldiers lose limbs due to accidents and conflict situations. Traditional prosthetics are expensive, fragile, and difficult to maintain. This system aims to create an **intelligent, affordable, and healthcare-oriented software** that predicts hand movements from EMG muscle signals.

## ✨ Features

### Core Functionality
- ✅ **8-Channel EMG Signal Processing**
- ✅ **Random Forest ML Classifier**
- ✅ **Real-Time Gesture Prediction**
- ✅ **Multi-Class Classification** (rest, fist, wrist flexion, extension, open palm, etc.)

### Healthcare Dashboard
- 📊 **Model Performance Metrics** (Accuracy, Precision, Recall, F1-Score)
- 🔲 **Confusion Matrix Visualization**
- 📈 **Feature Importance Analysis**
- 👤 **Patient Progress Dashboard**
- 🏥 **Rehabilitation Monitoring**
- 📉 **Progress Tracking Over Time**

### User Interface
- 🎨 **Modern Healthcare-Themed Design**
- 📁 **CSV File Upload**
- 💾 **Export Prediction Results**
- 🔍 **Interactive Visualizations**
- 📱 **Responsive Layout**

## 📁 Project Structure

```
bionic_limb_project/
│
├── app.py                      # Main Streamlit application
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── models/                    # Trained models directory
│   ├── random_forest_model.joblib
│   ├── label_encoder.joblib
│   ├── model_metadata.json
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── data/                      # Dataset directory
│   └── combined_emg_data.csv  # Your combined dataset
│
├── assets/                    # Images and resources
│   └── (gesture images, logos, etc.)
│
└── utils/                     # Utility functions
    └── (helper scripts if needed)
```

## 🚀 Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip** (Python package manager)
- **Git** (optional, for cloning)

### Step 1: Clone or Download the Project

```bash
# If using Git
git clone https://github.com/yourusername/bionic-limb-project.git
cd bionic-limb-project

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import streamlit; import sklearn; import pandas; print('All dependencies installed successfully!')"
```

## 📊 Dataset Format

Your CSV dataset should have the following structure:

| Column Name | Type    | Description                    |
|-------------|---------|--------------------------------|
| time        | float   | Timestamp of the measurement   |
| channel1    | float   | EMG signal from channel 1      |
| channel2    | float   | EMG signal from channel 2      |
| channel3    | float   | EMG signal from channel 3      |
| channel4    | float   | EMG signal from channel 4      |
| channel5    | float   | EMG signal from channel 5      |
| channel6    | float   | EMG signal from channel 6      |
| channel7    | float   | EMG signal from channel 7      |
| channel8    | float   | EMG signal from channel 8      |
| class       | string  | Gesture label (e.g., "fist")   |

### Supported Gesture Classes

- `rest` - Resting position
- `fist` - Closed fist
- `wrist_flexion` - Wrist flexion movement
- `wrist_extension` - Wrist extension movement
- `open_palm` - Open palm
- *(Add other gestures from your dataset)*

## 🎓 Model Training

### Step 1: Prepare Your Dataset

1. Place your combined CSV file in the `data/` directory
2. Ensure it's named `combined_emg_data.csv` or update the path in `train_model.py`

```python
# In train_model.py, update this line if needed:
DATA_PATH = "data/combined_emg_data.csv"
```

### Step 2: Run Training Script

```bash
python train_model.py
```

This will:
- Load and preprocess your EMG dataset
- Split data into training and testing sets (80/20)
- Train a Random Forest classifier
- Evaluate model performance
- Generate visualizations (confusion matrix, feature importance)
- Save the trained model in the `models/` directory

### Step 3: Check Training Results

After training completes, you'll see:

```
==============================================================
✓ Training pipeline completed successfully!
✓ Model accuracy: 92.45%
==============================================================
```

The following files will be created in `models/`:
- `random_forest_model.joblib` - Trained model
- `label_encoder.joblib` - Label encoder
- `model_metadata.json` - Model information and metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `feature_importance.png` - Feature importance plot

### Training Options

For faster development/testing with large datasets:

```python
# In train_model.py, modify:
trainer.run_complete_pipeline(
    test_size=0.2,
    sample_size=100000  # Use only 100k samples for testing
)
```

## 🖥️ Running the App

### Step 1: Ensure Model is Trained

Make sure you have trained the model first (see [Model Training](#model-training))

### Step 2: Launch Streamlit App

```bash
streamlit run app.py
```

### Step 3: Access the Application

The app will automatically open in your default browser at:
```
http://localhost:8501
```

If it doesn't open automatically, manually navigate to the URL shown in the terminal.

## 🎨 Using the Application

### 1. Home Page (🏠)
- Overview of the system
- Supported gestures
- System capabilities
- How it works

### 2. Real-Time Prediction (🔬)
1. Click "Browse files" to upload your EMG CSV data
2. Adjust the number of samples to predict
3. Click "🚀 Predict Gestures"
4. View results:
   - Detected gesture
   - Confidence scores
   - Gesture distribution chart
   - Detailed predictions table
5. Download predictions as CSV

### 3. Model Performance (📊)
- View accuracy, precision, recall, F1-score
- Explore confusion matrix
- Analyze feature importance
- Review model metadata

### 4. Patient Dashboard (👤)
- Select patient ID
- View session statistics
- Track accuracy over time
- Monitor gesture performance
- Review rehabilitation progress

### 5. Rehabilitation Monitor (📈)
- Weekly activity summary
- Activity heatmap
- Improvement trends
- AI-powered recommendations
- Exercise suggestions

### 6. About System (ℹ️)
- Project objectives
- Technical approach
- Technology stack
- Contact information

## 🔧 Technical Details

### Machine Learning Model

- **Algorithm:** Random Forest Classifier
- **Framework:** scikit-learn
- **Input Features:** 8 EMG channels (channel1-8)
- **Output:** Gesture classification (multi-class)
- **Training Split:** 80% training, 20% testing
- **Expected Accuracy:** >85%

### Model Parameters

```python
RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Maximum tree depth
    random_state=42,       # Reproducibility
    n_jobs=-1              # Use all CPU cores
)
```

### Performance Metrics

- **Accuracy:** Overall correct predictions
- **Precision:** Positive prediction accuracy
- **Recall:** True positive detection rate
- **F1-Score:** Harmonic mean of precision and recall

### Visualization Libraries

- **Plotly:** Interactive charts
- **Matplotlib:** Static plots
- **Seaborn:** Statistical visualizations

## 🐛 Troubleshooting

### Issue: "Model not loaded" error

**Solution:**
```bash
# Train the model first
python train_model.py

# Then run the app
streamlit run app.py
```

### Issue: Import errors

**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: Dataset file not found

**Solution:**
```bash
# Create data directory
mkdir data

# Place your CSV file in data/combined_emg_data.csv
# Or update DATA_PATH in train_model.py
```

### Issue: Out of memory during training

**Solution:**
```python
# In train_model.py, use sampling:
trainer.run_complete_pipeline(
    test_size=0.2,
    sample_size=100000  # Reduce this number
)
```

### Issue: Streamlit port already in use

**Solution:**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

## 📚 Additional Resources

### Learning Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [EMG Signal Processing](https://www.ncbi.nlm.nih.gov/pmc/articles/EMG/)
- [Random Forest Algorithm](https://scikit-learn.org/stable/modules/ensemble.html#forest)

### Dataset Sources
- UCI Machine Learning Repository
- PhysioNet EMG Datasets
- Open BCI Community

### Related Projects
- OpenBionics
- Thalmic Labs Myo
- EMG-based HCI Systems

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is developed for **educational and research purposes**. 

⚠️ **Important:** This is a demonstration project for learning purposes. Before deploying in any medical or clinical setting, ensure compliance with medical device regulations in your region (FDA, CE marking, etc.).

## 👥 Authors

- **Your Name** - Initial Development
- **Your Team** - ML Mini Project

## 🙏 Acknowledgments

- Healthcare ML research community
- Open-source contributors
- Educational institutions supporting assistive technology research

## 📧 Contact

For questions or support:
- **Email:** your.email@example.com
- **GitHub Issues:** [Report a bug](https://github.com/yourusername/bionic-limb-project/issues)

---

<div align="center">

### 🌟 Making Advanced Prosthetic Control Accessible to All 🌟

**Built with ❤️ for Healthcare Innovation**

</div>