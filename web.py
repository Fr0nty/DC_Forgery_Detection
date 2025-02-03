import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                            classification_report, roc_curve, auc, 
                            precision_recall_curve)
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D,
                         Flatten, Dense, Dropout)
import seaborn as sns
import streamlit as st
from collections import defaultdict

# Step 1: Create folders for normal and anomaly cells
def create_folders():
    os.makedirs("normal", exist_ok=True)
    os.makedirs("anomaly", exist_ok=True)

# Step 2: Preprocess the image and create a grid of 64x64 cells
def preprocess_image(image_path, cell_size=64, anomaly_percentage=0.1):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or could not be loaded.")

    image = cv2.resize(image, (1280, 1280))
    cells = []
    for y in range(0, image.shape[0], cell_size):
        for x in range(0, image.shape[1], cell_size):
            cell = image[y:y + cell_size, x:x + cell_size]
            cells.append(cell)

    for idx, cell in enumerate(cells):
        cv2.imwrite(os.path.join("normal", f"cell_{idx}.png"), cell)

    num_anomalies = int(len(cells) * anomaly_percentage)
    anomaly_indices = np.random.choice(len(cells), num_anomalies, replace=False)

    for idx in anomaly_indices:
        inverted_cell = cv2.bitwise_not(cells[idx])
        cv2.imwrite(os.path.join("anomaly", f"anomaly_{idx}.png"), inverted_cell)

    return cells, anomaly_indices

# Step 3: Load data from the folders
def load_data():
    normal_cells = []
    anomaly_cells = []

    for filename in os.listdir("normal"):
        cell = cv2.imread(os.path.join("normal", filename))
        if cell is not None:
            normal_cells.append(cell)

    for filename in os.listdir("anomaly"):
        cell = cv2.imread(os.path.join("anomaly", filename))
        if cell is not None:
            anomaly_cells.append(cell)

    x = np.array(normal_cells + anomaly_cells)
    y = np.array([0] * len(normal_cells) + [1] * len(anomaly_cells))

    return x, y

# Enhanced Feature Extraction
def extract_features(cells):
    cells = np.array(cells).reshape((-1, 64, 64, 3))

    def build_feature_extractor():
        return Sequential([
            Conv2D(64, (5,5), activation='relu', input_shape=(64, 64, 3)),
            MaxPooling2D((3,3)),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Conv2D(256, (3,3), activation='relu'),
            GlobalAveragePooling2D(),
            Dense(128, activation='selu'),
            Dropout(0.4)
        ])

    feature_extractor = build_feature_extractor()
    features = feature_extractor.predict(cells)
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# Optimized Dendritic Cell Algorithm
def compute_pamp(features, mean_normal, inv_cov_normal):
    delta = features - mean_normal
    return np.sqrt(np.einsum('...i,ij,...j', delta, inv_cov_normal, delta))

def train_and_evaluate(x_train, x_test, y_train, y_test, x_test_images):
    # Class balancing
    normal_indices = np.where(y_train == 0)[0]
    anomaly_indices = np.where(y_train == 1)[0]
    min_samples = min(len(normal_indices), len(anomaly_indices))
    
    balanced_indices = np.concatenate([
        np.random.choice(normal_indices, min_samples),
        np.random.choice(anomaly_indices, min_samples)
    ])
    x_train = x_train[balanced_indices]
    y_train = y_train[balanced_indices]

    # Calculate normal statistics
    normal_mask = (y_train == 0)
    normal_features = x_train[normal_mask]
    mean_normal = np.mean(normal_features, axis=0)
    cov_normal = np.cov(normal_features, rowvar=False)
    inv_cov_normal = np.linalg.pinv(cov_normal)

    # Compute signals
    pamp_train = compute_pamp(x_train, mean_normal, inv_cov_normal)
    danger_train = pamp_train * 1.5  # Amplify danger signal
    safe_train = 1 / (1 + pamp_train**2)  # Quadratic suppression
    
    pamp_test = compute_pamp(x_test, mean_normal, inv_cov_normal)
    danger_test = pamp_test * 2
    safe_test = 1 / (1 + np.sqrt(pamp_test))

    # Signal normalization
    pamp_train = (pamp_train - np.min(pamp_train)) / (np.max(pamp_train) - np.min(pamp_train))
    danger_train = (danger_train - np.min(danger_train)) / (np.max(danger_train) - np.min(danger_train))
    
    pamp_test = (pamp_test - np.median(pamp_test)) / (np.percentile(pamp_test, 75) - np.percentile(pamp_test, 25))
    danger_test = (danger_test - np.median(danger_test)) / (np.percentile(danger_test, 75) - np.percentile(danger_test, 25))

    train_scores = (danger_train * 0.7) - (safe_train * 0.3)
    normal_scores = train_scores[y_train == 0]
    threshold = np.percentile(normal_scores, 97)  # 97th percentile of normal scores
    # Context-aware classification
    y_pred_test = []
    for i in range(len(x_test)):
        danger_context = np.mean(danger_test[i] > np.percentile(danger_train, 75))
        safe_context = np.mean(safe_test[i] < np.percentile(safe_train, 25))
        
        combined_score = (0.6 * danger_test[i]) + (0.3 * pamp_test[i]) - (0.4 * safe_test[i])
        
        # Adaptive decision making
        if combined_score > threshold:
            y_pred_test.append(1)
        elif danger_context > 0.8 and safe_context < 0.2:
            y_pred_test.append(1)
        else:
            y_pred_test.append(0)
    # Dendritic Cell parameters
    num_dcs = 200
    antigens_per_dc = 20
    anomaly_threshold = 0.7

    # Train DCs
    dc_assignments = [np.random.choice(len(x_train), antigens_per_dc, replace=False) 
                     for _ in range(num_dcs)]
    
    dc_contexts = []
    for indices in dc_assignments:
        pamp_sum = pamp_train[indices].sum()
        danger_sum = danger_train[indices].sum()
        safe_sum = safe_train[indices].sum()
        dc_contexts.append(1 if (pamp_sum + danger_sum) > safe_sum else 0)

    # Classify test instances with dynamic threshold
    normal_scores = []
    anomaly_scores = []
    y_pred_test = []
    
    for i in range(len(x_test)):
        score = (danger_test[i] * 0.8) - (safe_test[i] * 1.2)
        
        if y_test[i] == 0:
            normal_scores.append(score)
        else:
            anomaly_scores.append(score)
            
        y_pred_test.append(1 if score > anomaly_threshold else 0)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    class_report = classification_report(y_test, y_pred_test)

    return accuracy, conf_matrix, class_report, y_pred_test, x_test_images

# Streamlit Interface
def main():
    st.title("Immune-Inspired Art Forgery Detection")
    st.write("Dendritic Cell Algorithm for Art Authentication")

    # Reference image upload
    st.sidebar.header("Reference Masterpiece")
    ref_file = st.sidebar.file_uploader("Upload genuine artwork...", type=["jpg", "jpeg", "png"])
    if ref_file:
        with open("reference.jpg", "wb") as f:
            f.write(ref_file.getbuffer())
        st.sidebar.image(ref_file, caption="Reference Artwork", use_column_width=True)

    # Test image upload
    st.header("Artwork Analysis")
    test_file = st.file_uploader("Upload artwork for authentication...", type=["jpg", "jpeg", "png"])
    if test_file:
        with open("test.jpg", "wb") as f:
            f.write(test_file.getbuffer())
        st.image(test_file, caption="Submitted Artwork", use_column_width=True)

        if st.button("Analyze with Immune Defense System"):
            with st.spinner("Immune cells analyzing brushstrokes..."):
                try:
                    create_folders()
                    
                    # Process images
                    ref_cells, _ = preprocess_image("reference.jpg")
                    test_cells, _ = preprocess_image("test.jpg")

                    # Prepare dataset
                    X = np.array(ref_cells + test_cells)
                    y = np.array([0]*len(ref_cells) + [1]*len(test_cells))
                    
                    # Extract features
                    features = extract_features(X)

                    # Split data
                    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
                        features, y, np.arange(len(y)), test_size=0.2, random_state=42
                    )

                    # Store original images for visualization
                    X_test_images = X[indices_test]

                    # Train and evaluate
                    accuracy, cm, report, preds, test_images = train_and_evaluate(
                        X_train, X_test, y_train, y_test, X_test_images
                    )

                    # Display results
                    st.success("Immune Analysis Complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("System Accuracy", f"{accuracy:.2%}")
                    with col2:
                        verdict = "Genuine Artwork" if accuracy > 0.75 else "Potential Forgery"
                        st.metric("Expert Verdict", verdict)

                    # Confusion Matrix
                    st.subheader("Immune Response Analysis")
                    fig, ax = plt.subplots(figsize=(8,6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                                xticklabels=['Normal', 'Anomaly'],
                                yticklabels=['Normal', 'Anomaly'])
                    ax.set_xlabel("Predicted Labels")
                    ax.set_ylabel("Actual Labels")
                    st.pyplot(fig)

                    # Detailed Report
                    st.subheader("Forensic Breakdown")
                    st.code(report)

                    # Anomaly Visualization
                    st.subheader("Detected Anomalies")
                    anomaly_indices = np.where(np.array(preds) == 1)[0]
                    if len(anomaly_indices) > 0:
                        cols = st.columns(4)
                        for idx, col in zip(anomaly_indices[:4], cols):
                            cell = cv2.cvtColor(test_images[idx], cv2.COLOR_BGR2RGB)
                            col.image(cell, caption=f"Anomaly {idx+1}", use_column_width=True)
                    else:
                        st.info("No significant anomalies detected")

                except Exception as e:
                    st.error(f"Immune system error: {str(e)}")

if __name__ == "__main__":
    main()