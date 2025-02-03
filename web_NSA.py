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
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

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
class NegativeSelectionAlgorithm:
    def __init__(self, self_set, num_detectors=100, threshold=0.1):
        """
        Initialize the Negative Selection Algorithm.
        
        :param self_set: The self-set (normal data).
        :param num_detectors: Number of detectors to generate.
        :param threshold: Threshold for matching (smaller values mean stricter matching).
        """
        self.self_set = self_set
        self.num_detectors = num_detectors
        self.threshold = threshold
        self.detectors = self._generate_detectors()

    def _generate_detectors(self):
        """
        Generate detectors that do not match the self-set.
        """
        detectors = []
        num_features = self.self_set.shape[1]

        while len(detectors) < self.num_detectors:
            # Randomly generate a candidate detector
            detector = np.random.rand(num_features)

            # Check if the detector does not match any self-sample
            if not self._matches_self(detector):
                detectors.append(detector)

        return np.array(detectors)

    def _matches_self(self, detector):
        """
        Check if a detector matches any sample in the self-set.
        """
        for self_sample in self.self_set:
            distance = np.linalg.norm(detector - self_sample)
            if distance < self.threshold:
                return True  # Detector matches self
        return False  # Detector does not match self

    def classify(self, data):
        """
        Classify data points as self (0) or non-self (1).
        """
        predictions = []

        for sample in data:
            is_non_self = False

            # Check if the sample matches any detector
            for detector in self.detectors:
                distance = np.linalg.norm(sample - detector)
                if distance < self.threshold:
                    is_non_self = True
                    break

            predictions.append(1 if is_non_self else 0)

        return np.array(predictions)

def extract_features(images):
    """
    Extract features from 64x64 image cells.
    """
    features = []

    for image in images:
        # Ensure the image is in the correct shape (height × width × channels)
        if image.ndim == 2:
            # If grayscale, use as is
            gray_image = image
        elif image.shape[-1] == 3:
            # If RGB, convert to grayscale
            gray_image = rgb2gray(image)
        else:
            # If multi-channel, reduce to grayscale by averaging across channels
            gray_image = np.mean(image, axis=-1)

        # Ensure the image is 2D
        if gray_image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image, but got {gray_image.ndim} dimensions.")

        # Rescale grayscale image to [0, 255] and convert to uint8
        gray_image = (gray_image * 255).astype(np.uint8)

        # Compute texture features using GLCM
        glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        # Compute color histogram (RGB)
        if image.ndim == 3 and image.shape[-1] == 3:
            # If RGB, compute histogram for all 3 channels
            color_hist = np.histogram(image, bins=8, range=(0, 256))[0]
        else:
            # If grayscale, compute histogram for the single channel
            color_hist = np.histogram(gray_image, bins=8, range=(0, 256))[0]

        # Combine features
        feature_vector = np.hstack([contrast, correlation, energy, homogeneity, color_hist])
        features.append(feature_vector)

    return np.array(features)

def train_and_evaluate(x_train, x_test, y_train, y_test):
    # Extract features from the image cells
    x_train_features = extract_features(x_train)
    x_test_features = extract_features(x_test)

    # Normalize the features
    scaler = StandardScaler()
    x_train_features = scaler.fit_transform(x_train_features)
    x_test_features = scaler.transform(x_test_features)

    # Define the self-set (normal data)
    self_set = x_train_features[y_train == 0]

    # Initialize and train the Negative Selection Algorithm
    nsa = NegativeSelectionAlgorithm(self_set, num_detectors=100, threshold=0.1)

    # Classify the test set
    predictions = nsa.classify(x_test_features)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    return accuracy, conf_matrix, class_report, predictions
def determine_forgery(conf_matrix):
    # Extract values from the confusion matrix
    true_normal = conf_matrix[0, 0]  # True negatives
    false_anomaly = conf_matrix[0, 1]  # False positives
    false_normal = conf_matrix[1, 0]  # False negatives
    true_anomaly = conf_matrix[1, 1]  # True positives

    # Check the structure of the confusion matrix
    if true_normal > false_anomaly and true_anomaly > false_normal:
        return "Forgery"
    elif true_normal < false_anomaly and true_anomaly < false_normal:
        return "True Artwork"
    else:
        return "Inconclusive"
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
                accuracy, conf_matrix, class_report, predictions = train_and_evaluate(x_train, x_test, y_train, y_test)
                # Display forgery status
                st.write("### Forgery Detection Result")
                forgery_status = determine_forgery(conf_matrix)
                # Display results
                st.success("Processing complete!")
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write("Confusion Matrix:")
                st.write(conf_matrix)
                st.write("Classification Report:")
                st.write(class_report)

                if forgery_status == "True Artwork":
                    st.success("The uploaded image is a **True Artwork**.")
                elif forgery_status == "Forgery":
                    st.error("The uploaded image is a **Forgery**.")
                else:
                    st.warning("The result is **Inconclusive**, likely forgery.")
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

                

# Run the Streamlit app
if __name__ == "__main__":
    main()