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
import seaborn as sns

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

# Step 4: Extract features using a simple CNN
def extract_features(cells):
    # Reshape cells to include channel dimension
    cells = np.array(cells).reshape((-1, 64, 64, 3))

    # Feature Extraction using a simple CNN
    def build_feature_extractor():
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu')
        ])
        return model

    feature_extractor = build_feature_extractor()
    features = feature_extractor.predict(cells)

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features

# Step 5: Train and evaluate the model
def train_and_evaluate(x_train, x_test, y_train, y_test, x_test_images):
    # Use a simple classifier (e.g., Logistic Regression) for demonstration
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)
    model.fit(x_train, y_train)

    # Evaluate the model
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    evaluate_and_plot(x_test, y_test, model, x_test_images)

def plot_confusion_matrix(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def plot_roc_curve(y_test, probabilities):
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def plot_precision_recall(y_test, probabilities):
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()

def show_random_samples(x_test_images, y_test, predictions):
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test_images[i])
        plt.title(f"True: {y_test[i]}\nPred: {predictions[i]}")
        plt.axis('off')
    plt.suptitle("Model Predictions on Image Cells")
    plt.show()

def evaluate_and_plot(x_test, y_test, model, x_test_images):
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]  # Probability of anomaly
    
    plot_confusion_matrix(y_test, predictions)
    plot_roc_curve(y_test, probabilities)
    plot_precision_recall(y_test, probabilities)
    show_random_samples(x_test_images, y_test, predictions)

# Main function to run the entire pipeline
def main(image_path):
    # Step 1: Create folders
    create_folders()

    # Step 2: Preprocess the image and create synthetic data
    cells, anomaly_indices = preprocess_image(image_path)

    # Step 3: Load data from the folders
    x, y = load_data()

    # Step 4: Extract features
    features = extract_features(x)

    # Step 5: Split the data into training and testing sets
    x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(
        features, y, np.arange(len(y)), test_size=0.2, random_state=42
    )

    # Store the original image data for visualization
    x_test_images = x[indices_test]  # Use indices to get the corresponding original images

    # Step 6: Train and evaluate the model
    train_and_evaluate(x_train, x_test, y_train, y_test, x_test_images)

# Run the pipeline
image_path = "image.jpg"  # Replace with the path to your image
main(image_path)