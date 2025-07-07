# Image Classifier – Cats vs Dogs (TensorFlow CNN)

This project is a beginner-friendly image classifier that uses a Convolutional Neural Network (CNN) to distinguish between images of cats and dogs. It’s built with TensorFlow and structured using clean, object-oriented Python across multiple modules.

---

## Project Structure
image-classifier/
│
├── main.py # Entry point to train and test the model
├── train.py # Handles training logic
├── test.py # Runs predictions on new images
│
├── model/
│ └── classifier.py # ImageClassifier class (data loading, model building, training)
│
├── utils/
│ └── visualizer.py # Accuracy/Loss plotting
│
├── cats_vs_dogs_model.h5 # (Output) Saved trained model
└── README.md

---

## Features

✅ Loads and preprocesses the Cats vs Dogs dataset  
✅ Builds a custom CNN using TensorFlow  
✅ Trains and validates the model with live feedback  
✅ Saves the trained model to disk  
✅ Predicts new images (dog or cat)  
✅ Visualizes training accuracy and loss

---

## Technologies Used

- Python 3.10
- TensorFlow / Keras
- Matplotlib
- NumPy

---

## Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Vinush22/Image-Classifier.git
cd Image-Classifier
