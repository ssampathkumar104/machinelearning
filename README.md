# Machine Learning Classification Pipeline

A Python-based end-to-end machine learning project that demonstrates classification pipelines using scikit-learn, XGBoost, and TensorFlow. This repository includes implementations for training Random Forest classifiers on fertilizer recommendation datasets with complete data preprocessing, model training, and evaluation workflows.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Stack](#stack)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Architecture](#pipeline-architecture)
- [Results](#results)

## 📖 Overview

This project demonstrates best practices for building scalable ML pipelines with scikit-learn. It includes:
- Data acquisition from Google Drive
- Automated data preprocessing and cleaning
- Custom transformer implementations
- Model training with hyperparameter tuning
- Comprehensive evaluation metrics

The project focuses on a fertilizer recommendation classification task using custom transformer classes that integrate seamlessly with scikit-learn's pipeline architecture.

## ✨ Features

- **Google Drive Integration**: Direct data loading from Google Drive using OAuth2 or public URLs
- **Custom Transformers**: Reusable, modular preprocessing components (`NullChecker`, `LabelEncoderTransformer`, `StandardScalerTransformer`)
- **Multiple ML Algorithms**: RandomForest and XGBoost implementations
- **Complete Pipeline**: End-to-end workflow from data ingestion to model evaluation
- **Flexible Authentication**: Two approaches—OAuth2 flow and direct URL access

## 📁 Project Structure

```
.
├── RandomForestClassifier.py      # Main pipeline with Google Drive OAuth2 authentication
├── RandomForestClassifier2.py     # Simplified variant using direct Google Drive URL
├── requirements.txt               # Python dependencies
├── input.txt                      # Configuration file (stores Google Drive file_id)
├── .devcontainer/                 # Dev container setup for consistent environments
└── README.md                       # This file
```

## 🛠 Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.7+ |
| **ML Frameworks** | scikit-learn 1.3.1, XGBoost 1.7.6, TensorFlow |
| **Data Processing** | pandas, NumPy |
| **Visualization** | matplotlib, seaborn 0.12.2 |
| **Cloud Integration** | Google Drive API (googleapiclient 2.98.0) |
| **Authentication** | google-auth 2.23.0, google-auth-oauthlib 1.0.0 |

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Google account (for accessing datasets on Google Drive)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ssampathkumar104/machinelearning.git
cd machinelearning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Google Drive credentials (for `RandomForestClassifier.py`):
   - Download `credentials.json` from [Google Cloud Console](https://console.cloud.google.com/)
   - Place it in the project root directory
   - Create `input.txt` with your Google Drive file ID:
     ```
     file_id=YOUR_DRIVE_FILE_ID
     ```

## 🚀 Usage

### Option 1: Using OAuth2 Authentication (RandomForestClassifier.py)

```bash
python RandomForestClassifier.py
```

This script:
- Authenticates with Google Drive using OAuth2
- Downloads the CSV file specified in `input.txt`
- Displays data overview (columns, shape, missing values, duplicates)
- Trains a Random Forest classifier with custom preprocessing steps
- Outputs accuracy and classification report

### Option 2: Using Direct Google Drive URL (RandomForestClassifier2.py)

```bash
python RandomForestClassifier2.py
```

This simplified version:
- Uses a hardcoded public Google Drive file URL (no authentication needed)
- Executes the same preprocessing and training pipeline
- Provides the same evaluation metrics

## 🔄 Pipeline Architecture

Both implementations follow this modular pipeline structure:

```
Raw Data (CSV from Google Drive)
    ↓
[NullChecker] → Handle missing values (fill with 0)
    ↓
[LabelEncoderTransformer] → Encode categorical variables
    ↓
[StandardScalerTransformer] → Normalize numeric features
    ↓
[DataSplitter] → Split into train/test sets (75/25)
    ↓
[ModelTrainer] → Train RandomForest/XGBoost classifier
    ↓
[ModelSaver] → Persist model predictions
    ↓
Evaluation Metrics (Accuracy, Classification Report)
```

### Custom Transformers

- **NullChecker**: Fills null values with 0
- **LabelEncoderTransformer**: Converts categorical features to numeric labels
- **StandardScalerTransformer**: Standardizes numeric features to mean=0, std=1
- **DataSplitter**: Splits data into train/test sets (configurable ratio)
- **ModelTrainer**: Trains ensemble models with configurable hyperparameters
- **ModelSaver**: Persists model predictions to pickle files

## 📊 Results

The model outputs:
- **Accuracy Score**: Overall classification accuracy on the test set
- **Classification Report**: Precision, recall, and F1-score per class
- **Saved Model**: Predictions saved to `final_rf_model.pkl`

Example output:
```
Accuracy: 0.92
Classification Report:
              precision    recall  f1-score   support
           0       0.91      0.93      0.92       245
           1       0.93      0.91      0.92       255
```

## 🔧 Configuration

Edit pipeline parameters in the `ModelTrainer` initialization:
```python
ModelTrainer(
    n_estimators=50,        # Number of trees in the forest
    max_depth=10,           # Maximum tree depth
    min_samples_split=2,    # Minimum samples to split a node
    min_samples_leaf=1,     # Minimum samples in leaf node
    criterion='gini',       # Split quality criterion
    bootstrap=True          # Use bootstrap samples
)
```

## 📝 Notes

- Both scripts target the fertilizer recommendation classification task (`target_feature='fertilizer_name'`)
- The project is designed for educational purposes and demonstrates ML best practices
- Custom transformers are compatible with scikit-learn's `Pipeline` and `GridSearchCV`
- TensorFlow is listed in requirements but not currently used; reserved for future deep learning implementations

## 🤝 Contributing

Contributions welcome! Feel free to:
- Improve model performance
- Add additional preprocessing techniques
- Extend to other ML algorithms
- Enhance documentation

## 📄 License

This project is open source and available for educational use.

---

**Last Updated**: November 2024
