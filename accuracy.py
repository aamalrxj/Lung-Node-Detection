import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the saved model
model = tf.keras.models.load_model('models/lung_nodule_detection_model.h5')

# Model Configuration
img_height, img_width = 224, 224  # Adjust size as necessary
batch_size = 32

# Data Loading for Training
train_datagen = ImageDataGenerator(rescale=1./255)  # Only normalization for training

# Create a training data generator (this assumes your data is still in the same structure)
train_generator = train_datagen.flow_from_directory(
    'data/preprocessed/train',  # Path to your training dataset
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Do not shuffle for evaluation
)

# Create a validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)  # Only normalization for validation

val_generator = val_datagen.flow_from_directory(
    'data/preprocessed/validation',  # Path to your validation dataset
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Do not shuffle for evaluation
)

# Evaluate the model on the training data
train_loss, train_accuracy = model.evaluate(train_generator)
print(f'Training Loss: {train_loss:.4f}')
print(f'Training Accuracy: {train_accuracy:.4f}')

# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(val_generator)
print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')
