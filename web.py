import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from imblearn.over_sampling import SMOTE
import seaborn as sns
import streamlit as st

# Step 1: Create folders for normal and anomaly cells
def create_folders():
    os.makedirs("normal", exist_ok=True)
    os.makedirs("anomaly", exist_ok=True)

# Step 2: Preprocess the image and create a grid of 64x64 cells
def preprocess_image(image_path, cell_size=64, anomaly_percentage=0.1):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or could not be loaded.")

    # Resize the image to 1280x1280
    image = cv2.resize(image, (1280, 1280))

    # Create a grid of 64x64 cells
    cells = []
    for y in range(0, image.shape[0], cell_size):
        for x in range(0, image.shape[1], cell_size):
            cell = image[y:y + cell_size, x:x + cell_size]
            cells.append(cell)

    # Save all cells in the normal folder
    for idx, cell in enumerate(cells):
        cv2.imwrite(os.path.join("normal", f"cell_{idx}.png"), cell)

    # Randomly select 10% of the cells to invert (simulate anomalies)
    num_anomalies = int(len(cells) * anomaly_percentage)
    anomaly_indices = np.random.choice(len(cells), num_anomalies, replace=False)

    # Invert the colors of the selected cells and save them in the anomaly folder
    for idx in anomaly_indices:
        inverted_cell = cv2.bitwise_not(cells[idx])
        cv2.imwrite(os.path.join("anomaly", f"anomaly_{idx}.png"), inverted_cell)

    return cells, anomaly_indices

# Step 3: Load data from the folders
def load_data():
    normal_cells = []
    anomaly_cells = []

    # Load normal cells
    for filename in os.listdir("normal"):
        cell = cv2.imread(os.path.join("normal", filename))
        normal_cells.append(cell)

    # Load anomaly cells
    for filename in os.listdir("anomaly"):
        cell = cv2.imread(os.path.join("anomaly", filename))
        anomaly_cells.append(cell)

    # Create labels (0 for normal, 1 for anomaly)
    x = np.array(normal_cells + anomaly_cells)
    y = np.array([0] * len(normal_cells) + [1] * len(anomaly_cells))

    return x, y

# Step 4: Extract features using a deeper CNN
def extract_features(cells):
    # Reshape cells to include channel dimension
    cells = np.array(cells).reshape((-1, 64, 64, 3))

    # Feature Extraction using a deeper CNN
    def build_feature_extractor():
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu')
        ])
        return model

    feature_extractor = build_feature_extractor()
    features = feature_extractor.predict(cells)

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features

# Dendritic Cell Algorithm (DCA) Implementation
class DendriticCell:
    def __init__(self):
        self.safe_signal = 0
        self.danger_signal = 0
        self.inflammatory_signal = 0
        self.antigen = None

    def process_signals(self, safe, danger, inflammatory):
        self.safe_signal += safe
        self.danger_signal += danger
        self.inflammatory_signal += inflammatory

    def classify_antigen(self):
        # Use a weighted sum of signals for classification
        if self.danger_signal > 1.5 * self.safe_signal:  # Adjusted threshold
            return 1  # Anomalous
        else:
            return 0  # Normal

def generate_signals(features, mean_normal, std_normal, epsilon=1e-10):
    # Generate signals based on feature distances
    safe_signal = np.exp(-((features - mean_normal) ** 2) / (2 * (std_normal ** 2 + epsilon)))  # Gaussian-like safe signal
    danger_signal = 1.95 - safe_signal  # Enhanced danger signal
    inflammatory_signal = random.random()  # Random inflammatory signal

    # Convert signals to scalars by taking the mean
    safe_signal = np.mean(safe_signal)
    danger_signal = np.mean(danger_signal)
    inflammatory_signal = np.mean(inflammatory_signal)

    return safe_signal, danger_signal, inflammatory_signal

def dca_anomaly_detection(x_data, mean_normal, std_normal):
    dendritic_cells = [DendriticCell() for _ in range(10)]  # Create 10 dendritic cells
    predictions = []

    for i, data_point in enumerate(x_data):
        safe, danger, inflammatory = generate_signals(data_point, mean_normal, std_normal)
        cell = dendritic_cells[i % len(dendritic_cells)]  # Round-robin assignment
        cell.process_signals(safe, danger, inflammatory)
        cell.antigen = data_point
        predictions.append(cell.classify_antigen())

    return predictions

# Step 5: Train and evaluate the model
def train_and_evaluate(x_train, x_test, y_train, y_test, x_test_images):
    # Use SMOTE to address class imbalance
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    # Compute mean and std of normal features
    normal_features = x_train_resampled[y_train_resampled == 0]
    mean_normal = np.mean(normal_features, axis=0)
    std_normal = np.std(normal_features, axis=0)

    # Run DCA on the test set
    predictions = dca_anomaly_detection(x_test, mean_normal, std_normal)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    return accuracy, conf_matrix, class_report, predictions

# Streamlit App
def main():
    st.title("Art Forgery Detection")
    st.write("Upload a reference image (original artwork) and an image to check for forgeries.")

    # File uploader for reference image
    st.sidebar.header("Reference Image")
    reference_file = st.sidebar.file_uploader("Upload the reference image...", type=["jpg", "jpeg", "png"])
    if reference_file is not None:
        # Save the reference file
        with open("reference_image.jpg", "wb") as f:
            f.write(reference_file.getbuffer())

        # Display the reference image
        st.sidebar.image(reference_file, caption="Reference Image", use_column_width=True)

    # File uploader for the image to check
    st.header("Image to Check")
    uploaded_file = st.file_uploader("Upload the image to check...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Save the uploaded file
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Run anomaly detection when the user clicks the button
        if st.button("Detect Forgeries"):
            with st.spinner("Processing..."):
                # Step 1: Create folders
                create_folders()

                # Step 2: Preprocess the reference image and create synthetic data
                reference_cells, _ = preprocess_image("reference_image.jpg")

                # Step 3: Preprocess the uploaded image and create synthetic data
                uploaded_cells, _ = preprocess_image("uploaded_image.jpg")

                # Step 4: Combine the data
                x = np.array(reference_cells + uploaded_cells)
                y = np.array([0] * len(reference_cells) + [1] * len(uploaded_cells))

                # Step 5: Extract features
                features = extract_features(x)

                # Step 6: Split the data into training and testing sets
                x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(
                    features, y, np.arange(len(y)), test_size=0.2, random_state=42
                )

                # Store the original image data for visualization
                x_test_images = x[indices_test]  # Use indices to get the corresponding original images

                # Step 7: Train and evaluate the model
                accuracy, conf_matrix, class_report, predictions = train_and_evaluate(x_train, x_test, y_train, y_test, x_test_images)

                # Display results
                st.success("Processing complete!")
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write("Confusion Matrix:")
                st.write(conf_matrix)
                st.write("Classification Report:")
                st.write(class_report)

                # Plot confusion matrix
                st.write("Confusion Matrix:")
                fig, ax = plt.subplots()
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

                # Plot ROC curve
                st.write("ROC Curve:")
                fpr, tpr, _ = roc_curve(y_test, predictions)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
                st.pyplot(fig)

                # Plot Precision-Recall curve
                st.write("Precision-Recall Curve:")
                precision, recall, _ = precision_recall_curve(y_test, predictions)
                fig, ax = plt.subplots()
                ax.plot(recall, precision, color='green', lw=2)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall Curve")
                st.pyplot(fig)

                # Show random samples
                st.write("Random Samples:")
                fig, axes = plt.subplots(2, 5, figsize=(12, 6))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(x_test_images[i])
                    ax.set_title(f"True: {y_test[i]}\nPred: {predictions[i]}")
                    ax.axis('off')
                st.pyplot(fig)

# Run the Streamlit app
if __name__ == "__main__":
    main()