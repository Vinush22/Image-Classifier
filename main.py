# main.py

from train import train_model
from test import predict_image

def main():
    #Train the model and save it
    train_model(epochs=5)

    # Predict a test image
    # Replace 'your_image.jpg' with a valid image path on your system
    predict_image(model_path="cats_vs_dogs_model.h5", img_path="your_image.jpg")

if __name__ == "__main__":
    main()
