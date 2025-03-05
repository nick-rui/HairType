import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------
# 1. Load and Prepare Data
# -----------------------

# Update paths as per your directory structure
csv_path = r"scripts/imagehairtype_2col.csv"
images_dir = r"images/Figaro-1k/Figaro-1k/Original/Testing"

# Read the CSV file
df = pd.read_csv(csv_path)

# Remove leading/trailing whitespace from column names
df.columns = df.columns.str.strip()
print("Columns in CSV:", df.columns.tolist())

# Select only relevant columns and drop rows with missing values
df = df[['Image name', 'Curl Diameter', 'Curl Frequency']].dropna()

# Ensure all filenames have a `.jpg` extension
df['Image name'] = df['Image name'].apply(lambda x: x.strip() + ".jpg" if not x.strip().endswith(".jpg") else x.strip())

# Convert "Curl Diameter" and "Curl Frequency" to float
df['Curl Diameter'] = df['Curl Diameter'].astype(float)
df['Curl Frequency'] = df['Curl Frequency'].astype(float)

# Define maximum values based on your domain
max_diameter = 3.0
max_frequency = 5.0  # Adjust if necessary

# Image dimensions
IMG_WIDTH, IMG_HEIGHT = 128, 128

def load_and_preprocess_image(image_filename):
    """
    Loads an image from the given directory, converts from BGR to RGB,
    resizes to IMG_WIDTH x IMG_HEIGHT, and normalizes pixel values.
    """
    path = os.path.normpath(os.path.join(images_dir, image_filename))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype('float32') / 255.0
    return image

images = []
labels = []

# Iterate through the DataFrame to load images and create target vectors
for idx, row in df.iterrows():
    image_name = row['Image name']
    try:
        img = load_and_preprocess_image(image_name)
        images.append(img)
        # Create a two-element vector [curl diameter, curl frequency]
        labels.append([row['Curl Diameter'], row['Curl Frequency']])
    except Exception as e:
        print(f"Skipping {image_name}: {e}")

X = np.array(images)
y = np.array(labels)  # Shape will be (num_samples, 2)

print(f"Loaded {X.shape[0]} images.")

if X.shape[0] == 0:
    raise RuntimeError("No images were successfully loaded. Check your file paths and CSV content.")

# -----------------------
# 2. Split and Scale Data
# -----------------------

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Define scaling factors for each output
scaling_factors = np.array([max_diameter, max_frequency])

# Scale labels to the [0, 1] range
y_train = y_train / scaling_factors
y_test = y_test / scaling_factors

# Further split X_train into training and validation sets (10% for validation)
X_train_new, X_val, y_train_new, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, shuffle=True
)

# -----------------------
# 3. Data Augmentation (for training data)
# -----------------------

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train_new)

train_generator = datagen.flow(X_train_new, y_train_new, batch_size=32, shuffle=True)

# -----------------------
# 4. Build the CNN Model
# -----------------------

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    # Final Dense layer now outputs 2 numbers (normalized to [0,1])
    Dense(2, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])
model.summary()

# -----------------------
# 5. Callbacks for Optimization
# -----------------------

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-4
)

# -----------------------
# 6. Train the Model
# -----------------------

history = model.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early_stopping, reduce_lr]
)

# -----------------------
# 7. Evaluate and Predict
# -----------------------

# Evaluate on test data
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE (scaled): {test_mae:.4f}")

# Predict on test data
predictions = model.predict(X_test)

# Scale predictions back to the original range for each target
for i in range(min(5, len(predictions))):
    pred_dia = predictions[i][0] * max_diameter
    pred_freq = predictions[i][1] * max_frequency
    actual_dia = y_test[i][0] * max_diameter
    actual_freq = y_test[i][1] * max_frequency
    print(f"Sample {i}: Predicted - Diameter: {pred_dia:.2f}, Frequency: {pred_freq:.2f} | Actual - Diameter: {actual_dia:.2f}, Frequency: {actual_freq:.2f}")


import matplotlib.pyplot as plt
y_pred_rescaled = predictions * scaling_factors
y_test_rescaled = y_test * scaling_factors
# Select a subset of test samples to display


# Select a subset of test samples to display
num_samples_to_display = min(5, len(y_pred_rescaled))

# Create a figure with multiple rows, 2 columns (image + text)
fig, axes = plt.subplots(num_samples_to_display, 2, figsize=(10, num_samples_to_display * 2.5))

for i in range(num_samples_to_display):
    image = X_test[i]  # Retrieve the image
    image_name = df.iloc[i]['Image name']  # Get corresponding image name

    # Show the image in the first column
    ax_img = axes[i, 0] if num_samples_to_display > 1 else axes[0]
    ax_img.imshow(image)
    ax_img.axis('off')  # Hide axis for better visualization

    # Display text information in the second column
    ax_text = axes[i, 1] if num_samples_to_display > 1 else axes[1]
    ax_text.axis('off')  # No axis for the text area
    ax_text.text(
        0, 0.5,  
        f"Image Name: {image_name}\n\n"
        f"Predicted:\n   - Diameter: {y_pred_rescaled[i][0]:.2f}\n   - Frequency: {y_pred_rescaled[i][1]:.2f}\n\n"
        f"Actual:\n   - Diameter: {y_test_rescaled[i][0]:.2f}\n   - Frequency: {y_test_rescaled[i][1]:.2f}",
        fontsize=10, verticalalignment="center"
    )

# Adjust layout for better spacing
plt.tight_layout()
plt.show()