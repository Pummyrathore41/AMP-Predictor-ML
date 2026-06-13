import os
import requests
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ----------------------------
# CONFIG
# ----------------------------
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

POS_FASTA = os.path.join(DATA_DIR, "amp_sequences.fasta")
NEG_FASTA = os.path.join(DATA_DIR, "non_amp_sequences.fasta")

# ----------------------------
# STEP 1: Download negative (non-AMP) set from UniProt automatically
# ----------------------------
def download_negative_set(n_sequences=1500):
    if os.path.exists(NEG_FASTA):
        print("Negative set already exists, skipping download.")
        return

    print("Downloading non-AMP sequences from UniProt...")
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": "reviewed:true AND length:[10 TO 50] NOT (keyword:Antimicrobial)",
        "format": "fasta",
        "size": n_sequences
    }
    response = requests.get(url, params=params)
    response.raise_for_status()

    with open(NEG_FASTA, "w") as f:
        f.write(response.text)

    print(f"Saved negative set to {NEG_FASTA}")

# ----------------------------
# STEP 2: Load sequences from FASTA
# ----------------------------
def load_sequences(fasta_path, label):
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        # Keep only standard amino acids, reasonable length
        if all(aa in AMINO_ACIDS for aa in seq) and 5 <= len(seq) <= 60:
            sequences.append((seq, label))
    return sequences

# ----------------------------
# STEP 3: Feature extraction
# ----------------------------
def extract_features(seq):
    features = {}

    # Amino acid composition (20 features)
    for aa in AMINO_ACIDS:
        features[f"AAC_{aa}"] = seq.count(aa) / len(seq)

    # Dipeptide composition (400 features)
    for aa1 in AMINO_ACIDS:
        for aa2 in AMINO_ACIDS:
            dipep = aa1 + aa2
            count = sum(1 for i in range(len(seq)-1) if seq[i:i+2] == dipep)
            features[f"DPC_{dipep}"] = count / (len(seq)-1) if len(seq) > 1 else 0

    # Physicochemical properties using BioPython
    analysis = ProteinAnalysis(seq)
    features["length"] = len(seq)
    features["molecular_weight"] = analysis.molecular_weight()
    features["aromaticity"] = analysis.aromaticity()
    features["instability_index"] = analysis.instability_index()
    features["isoelectric_point"] = analysis.isoelectric_point()
    features["gravy"] = analysis.gravy()  # hydrophobicity
    features["net_charge_pH7"] = analysis.charge_at_pH(7.0)

    return features

def build_dataset():
    print("Loading sequences...")
    pos_seqs = load_sequences(POS_FASTA, 1)
    neg_seqs = load_sequences(NEG_FASTA, 0)
    print(f"Positive (AMP) sequences: {len(pos_seqs)}")
    print(f"Negative (non-AMP) sequences: {len(neg_seqs)}")

    all_data = pos_seqs + neg_seqs
    rows = []
    for seq, label in all_data:
        feat = extract_features(seq)
        feat["label"] = label
        feat["sequence"] = seq
        rows.append(feat)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "feature_dataset.csv"), index=False)
    print(f"Feature dataset saved. Shape: {df.shape}")
    return df

# ----------------------------
# STEP 4: Train models
# ----------------------------
def train_and_evaluate(df):
    feature_cols = [c for c in df.columns if c not in ["label", "sequence"]]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(probability=True, kernel="rbf", random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }

    results = {}
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Non-AMP", "AMP"]))

        results[name] = {"model": model, "accuracy": acc}

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-AMP", "AMP"], yticklabels=["Non-AMP", "AMP"])
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{name.replace(' ', '_')}.png"))
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(1)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.figure(1)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_comparison.png"))
    plt.close()

    # Save best model + scaler + feature column order for the app
    best_model_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_model_name]["model"]
    print(f"\nBest model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")

    joblib.dump(best_model, os.path.join(RESULTS_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(RESULTS_DIR, "feature_columns.pkl"))

    # Save accuracy comparison
    acc_df = pd.DataFrame([{"Model": k, "Accuracy": v["accuracy"]} for k, v in results.items()])
    acc_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print("\nAll results, plots, and models saved in 'results/' folder.")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    if not os.path.exists(POS_FASTA):
        raise FileNotFoundError(
            f"'{POS_FASTA}' not found. Please download AMP sequences from APD3 "
            f"(https://aps.unmc.edu/database) and save as data/amp_sequences.fasta"
        )

    download_negative_set()
    df = build_dataset()
    train_and_evaluate(df)