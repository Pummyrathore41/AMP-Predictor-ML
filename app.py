import streamlit as st
import joblib
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Load saved model, scaler, and feature columns
model = joblib.load("results/best_model.pkl")
scaler = joblib.load("results/scaler.pkl")
feature_cols = joblib.load("results/feature_columns.pkl")

def extract_features(seq):
    features = {}
    for aa in AMINO_ACIDS:
        features[f"AAC_{aa}"] = seq.count(aa) / len(seq)

    for aa1 in AMINO_ACIDS:
        for aa2 in AMINO_ACIDS:
            dipep = aa1 + aa2
            count = sum(1 for i in range(len(seq)-1) if seq[i:i+2] == dipep)
            features[f"DPC_{dipep}"] = count / (len(seq)-1) if len(seq) > 1 else 0

    analysis = ProteinAnalysis(seq)
    features["length"] = len(seq)
    features["molecular_weight"] = analysis.molecular_weight()
    features["aromaticity"] = analysis.aromaticity()
    features["instability_index"] = analysis.instability_index()
    features["isoelectric_point"] = analysis.isoelectric_point()
    features["gravy"] = analysis.gravy()
    features["net_charge_pH7"] = analysis.charge_at_pH(7.0)

    return features

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="AMP Predictor", page_icon="🧬")
st.title("🧬 Antimicrobial Peptide Predictor")
st.write("Enter a peptide sequence (only standard amino acids: A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y)")

seq_input = st.text_input("Peptide Sequence", placeholder="e.g. KWKLFKKIEKVGRNVRDGIIKAGPAVAVVGQAATVVK")

if st.button("Predict"):
    seq = seq_input.strip().upper()

    if not seq:
        st.warning("Please enter a sequence.")
    elif not all(aa in AMINO_ACIDS for aa in seq):
        st.error("Invalid characters detected. Use only standard amino acid letters.")
    elif not (5 <= len(seq) <= 60):
        st.error("Sequence length should be between 5 and 60 amino acids.")
    else:
        feats = extract_features(seq)
        X = np.array([[feats[col] for col in feature_cols]])
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]

        if prediction == 1:
            st.success(f"✅ This sequence is predicted to be an **Antimicrobial Peptide (AMP)**")
            st.metric("Confidence", f"{probability[1]*100:.2f}%")
        else:
            st.info(f"❌ This sequence is predicted to be **Non-AMP**")
            st.metric("Confidence", f"{probability[0]*100:.2f}%")

        with st.expander("View calculated properties"):
            st.json({
                "Length": feats["length"],
                "Molecular Weight": round(feats["molecular_weight"], 2),
                "Isoelectric Point": round(feats["isoelectric_point"], 2),
                "Hydrophobicity (GRAVY)": round(feats["gravy"], 3),
                "Net Charge (pH 7)": round(feats["net_charge_pH7"], 2),
                "Aromaticity": round(feats["aromaticity"], 3)
            })