import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# -----------------------
# 1. Load and Prepare Data
# -----------------------

csv_path = r"scripts/twistlength.csv"
images_dir = r"images/Figaro-1k/Figaro-1k/Original/Testing"

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df = df[['Image name', 'Twist Frequency', 'Length']].dropna()
df['Image name'] = df['Image name'].apply(lambda x: x.strip() + ".jpg" if not x.strip().endswith(".jpg") else x.strip())
df['Twist Frequency'] = df['Twist Frequency'].astype(float)
df['Length'] = df['Length'].astype(float)

IMG_WIDTH, IMG_HEIGHT = 128, 128

def load_and_preprocess_image(image_filename):
    path = os.path.join(images_dir, image_filename)
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
for idx, row in df.iterrows():
    image_name = row['Image name']
    try:
        img = load_and_preprocess_image(image_name)
        images.append(img)
        labels.append([row['Twist Frequency'], row['Length']])
    except Exception as e:
        print(f"Skipping {image_name}: {e}")

X = np.array(images)
y = np.array(labels)
print(f"Loaded {X.shape[0]} images.")

if X.shape[0] == 0:
    raise RuntimeError("No images were loaded. Check your file paths and CSV content.")

# -----------------------
# 2. Split and Standard Scale Data
# -----------------------

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Standard scaling: subtract mean and divide by std for each target variable
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

# Further split training set into training and validation
X_train_new, X_val, y_train_new, y_val = train_test_split(
    X_train, y_train_scaled, test_size=0.1, random_state=42, shuffle=True
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

# We now use a linear activation in the final layer because our targets are standardized.
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
    Dense(2, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])
model.summary()

# -----------------------
# 5. Callbacks for Optimization
# -----------------------

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-4)

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

test_loss, test_mae = model.evaluate(X_test, y_test_scaled)
print(f"Test MAE (scaled): {test_mae:.4f}")

predictions_scaled = model.predict(X_test)
# Inverse transform to get predictions in the original scale
predictions = scaler.inverse_transform(predictions_scaled)

for i in range(min(5, len(predictions))):
    pred_twist, pred_length = predictions[i]
    actual_twist, actual_length = y_test[i]
    print(f"Sample {i}: Predicted - Twist Frequency: {pred_twist:.2f}, Length: {pred_length:.2f} | Actual - Twist Frequency: {actual_twist:.2f}, Length: {actual_length:.2f}")

# Visualization (optional)
num_samples_to_display = min(5, len(predictions))
fig, axes = plt.subplots(num_samples_to_display, 2, figsize=(10, num_samples_to_display * 2.5))
for i in range(num_samples_to_display):
    image = X_test[i]
    ax_img = axes[i, 0] if num_samples_to_display > 1 else axes[0]
    ax_img.imshow(image)
    ax_img.axis('off')
    ax_text = axes[i, 1] if num_samples_to_display > 1 else axes[1]
    ax_text.axis('off')
    ax_text.text(
        0, 0.5,
        f"Predicted:\n   - Twist Frequency: {predictions[i][0]:.2f}\n   - Length: {predictions[i][1]:.2f}\n\n"
        f"Actual:\n   - Twist Frequency: {y_test[i][0]:.2f}\n   - Length: {y_test[i][1]:.2f}",
        fontsize=10, verticalalignment="center"
    )
plt.tight_layout()
plt.show()
