import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# Model Configuration
img_height, img_width = 224, 224  # Adjust size as necessary
batch_size = 32
num_classes = 3  # Number of categories: benign, malignant, non-nodule

# Data Augmentation and Data Loading for Multiclass Classification
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only normalization for validation

# Adjusted paths for loading dataset; ensure the paths are correct
train_generator = train_datagen.flow_from_directory(
    'data/preprocessed/train',  # Update this to your lung nodule training dataset path
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Categorical classification for three classes
    shuffle=True  # Shuffle training data
)

val_generator = val_datagen.flow_from_directory(
    'data/preprocessed/validation',  # Update this to your lung nodule validation dataset path
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical', 
    shuffle=False  # Do not shuffle validation data
)

# Model Definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')  # Softmax for multiclass classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Categorical cross-entropy for multiclass classification
              metrics=['accuracy'])

# Training
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Restore best weights
model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# Save Model
model.save('models/lung_nodule_detection_model.h5')  # Save your trained model
