# Antimicrobial Peptide (AMP) Predictor

A machine learning pipeline that classifies peptide sequences as **Antimicrobial (AMP)** or **Non-AMP** using sequence-derived features (amino acid composition, dipeptide composition, and physicochemical properties), with an interactive Streamlit prediction app.

## 📋 Table of Contents
- [Overview](#overview)
- [Pipeline Stages](#pipeline-stages)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

## 🔬 Overview

The pipeline builds a labeled dataset of AMP (positive) and non-AMP (negative) peptide sequences, extracts **424 features** per sequence (20 amino acid composition + 400 dipeptide composition + 6 physicochemical properties + length), trains and compares three classifiers (Random Forest, SVM, Neural Network/MLP), and saves the best model. A Streamlit app then lets users input any peptide sequence and get a live AMP/Non-AMP prediction with confidence score.

## 🧬 Pipeline Stages

1. **Download negative set** — Automatically fetches ~1500 non-AMP reviewed sequences (length 10–50) from UniProt
2. **Load sequences** — Parses positive (AMP) and negative FASTA files, filters by valid amino acids and length (5–60)
3. **Feature extraction** — Computes AAC (20), DPC (400), molecular weight, aromaticity, instability index, isoelectric point, GRAVY (hydrophobicity), net charge at pH 7
4. **Train & evaluate** — Trains Random Forest, SVM, and MLP; generates confusion matrices, ROC curves, accuracy comparison; saves best model, scaler, and feature columns

## ⚙️ Installation

### Prerequisites
- Python **3.10.11**
- Git
- Internet connection (for UniProt download)

### Step 1: Clone the repository
```bash
git clone https://github.com/Pummyrathore41/Antimicrobial-Peptide-Predictor.git
cd Antimicrobial-Peptide-Predictor
```

### Step 2: Create a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Add the positive (AMP) dataset
Download AMP sequences from the **APD3 database** (https://aps.unmc.edu/database), and save as:
data/amp_sequences.fasta
> The negative (non-AMP) dataset is downloaded automatically from UniProt on first run.

## ▶️ Usage

### 1. Train the model
```bash
python amp_predictor.py
```
This will:
- Download non-AMP sequences (first run only)
- Build the feature dataset (`results/feature_dataset.csv`)
- Train Random Forest, SVM, and MLP models
- Save plots, model comparison CSV, and the best model to `results/`

### 2. Launch the prediction app
```bash
streamlit run app.py
```
Enter any peptide sequence (standard amino acids only, length 5–60) and click **Predict** to get an AMP / Non-AMP classification with confidence score and computed properties.

### Deactivate the environment (when done)
```bash
deactivate
```

## 📊 Results

After training, check the `results/` folder for:

| File | Description |
|---|---|
| `feature_dataset.csv` | Full extracted feature dataset (424 features per sequence) |
| `confusion_matrix_Random_Forest.png` | Confusion matrix — Random Forest |
| `confusion_matrix_SVM.png` | Confusion matrix — SVM |
| `confusion_matrix_Neural_Network_(MLP).png` | Confusion matrix — MLP |
| `roc_comparison.png` | ROC curve comparison across all 3 models |
| `model_comparison.csv` | Accuracy comparison table |
| `best_model.pkl` | Best-performing trained model (used by the app) |
| `scaler.pkl` | Fitted StandardScaler |
| `feature_columns.pkl` | Ordered feature column names |

### Sample Console Output
Downloading non-AMP sequences from UniProt...

Saved negative set to data/non_amp_sequences.fasta
Loading sequences...

Positive (AMP) sequences: 1480

Negative (non-AMP) sequences: 1452

Feature dataset saved. Shape: (2932, 426)
Training Random Forest...

Random Forest Accuracy: 0.9437

precision    recall  f1-score   support

Non-AMP       0.94      0.94      0.94       291

AMP       0.94      0.95      0.94       296
Training SVM...

SVM Accuracy: 0.9215
Training Neural Network (MLP)...

Neural Network (MLP) Accuracy: 0.9352
Best model: Random Forest (Accuracy: 0.9437)
All results, plots, and models saved in 'results/' folder.

### Streamlit App Preview
🧬 Antimicrobial Peptide Predictor
Peptide Sequence: KWKLFKKIEKVGRNVRDGIIKAGPAVAVVGQAATVVK
✅ This sequence is predicted to be an Antimicrobial Peptide (AMP)

Confidence: 96.40%
▼ View calculated properties

{

"Length": 37,

"Molecular Weight": 3987.92,

"Isoelectric Point": 11.32,

"Hydrophobicity (GRAVY)": 0.292,

"Net Charge (pH 7)": 8.51,

"Aromaticity": 0.054

}

## 📁 Project Structure
Antimicrobial-Peptide-Predictor/

├── amp_predictor.py       # Data download, feature extraction, training pipeline

├── app.py                  # Streamlit prediction app

├── requirements.txt        # Python dependencies

├── data/

│   ├── amp_sequences.fasta      # User-provided (from APD3)

│   └── non_amp_sequences.fasta  # Auto-downloaded from UniProt

├── results/                 # Auto-generated: models, plots, datasets

└── README.md

## 🛠️ Tech Stack

- **Biopython** — sequence parsing & physicochemical property calculation
- **scikit-learn** — Random Forest, SVM, MLP classifiers
- **pandas / numpy** — data handling
- **matplotlib / seaborn** — visualization (confusion matrices, ROC curves)
- **Streamlit** — interactive web app
- **joblib** — model persistence

## 📄 License

This project is open source. Add your preferred license here (e.g., MIT).
