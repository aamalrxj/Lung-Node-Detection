Lung Nodule Detection Using Deep Learning
A deep learning-based project that explores the classification of lung CT scan images using a Convolutional Neural Network (CNN).
The application uses TensorFlow and Keras for image classification and Flask to provide a simple web interface where users can upload CT scan images and view the model's predicted category.

Features
- Image Preprocessing: Resizes CT scan images to 224 × 224 pixels and normalizes pixel values.
- CNN Model: Uses a Convolutional Neural Network for multiclass image classification.
- Data Augmentation: Applies image transformations such as zooming, shearing, and horizontal flipping during training.
- Model Evaluation: Evaluates training and validation loss and accuracy.
- Web Interface: Provides a Flask-based interface for uploading CT scan images.
- Prediction Results: Displays the predicted classification with a corresponding description.

Technologies Used
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV |
| Data Processing | NumPy |
| Web Framework | Flask |
| Model Architecture | Convolutional Neural Network (CNN) |

Project Structure
The data, model, templates, and uploads directories are expected by the application but are not currently included in the repository.
1. Data Preparation: CT scan images are organized into categories and prepared for training.
2. Preprocessing: Images are resized and normalized to make them suitable for model input.
3. Model Training: A CNN is trained using the prepared dataset to learn image features.
4. Evaluation: The trained model is evaluated using training and validation datasets.
5. Prediction: Users upload a CT scan image through the Flask interface, and the model predicts its category.

Getting Started
1. Clone the repository
git clone https://github.com/aamalrxj/Lung-Node-Detection.git
cd Lung-Node-Detection


2. Create a virtual environment
python -m venv venv

Activate it:
venv\Scripts\activate
source venv/bin/activate


3. Install dependencies
pip install tensorflow flask numpy opencv-python


4. Prepare the dataset
Organize the CT scan images into training and validation directories.

5. Preprocess the images
python preprocess.py

6. Train the model
python train.py

The trained model is saved as:
models/lung_nodule_detection_model.h5


8. Evaluate the model
python accuracy.py


9. Run the Flask application
After placing the trained model and the required HTML templates in their expected locations:
python app.py


Model Architecture
The project uses a CNN architecture consisting of:
- Convolutional layers with ReLU activation.
- Max Pooling layers for dimensionality reduction.
- A Flatten layer.
- A fully connected Dense layer.
- A Softmax output layer for classification.
  
The model is trained using the Adam optimizer and categorical cross-entropy loss, with Early Stopping to help prevent overfitting.

Future Improvements
- Improve classification performance through model tuning.
- Explore transfer learning using pretrained CNN architectures.
- Add confusion matrix and precision, recall, and F1-score evaluation.
- Improve the web interface and prediction reporting.
- Add stronger input validation and secure image handling.
- Expand testing using independent CT scan datasets.


---

*An academic deep learning project exploring medical image classification using CNNs and Python.*
