# model/classifier.py

import tensorflow as tf
import os

class ImageClassifier:
    """
    ImageClassifier builds, trains, evaluates, and saves a CNN model for binary image classification (cats vs dogs).
    """

    def __init__(self, img_size=(180, 180), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.train_ds = None
        self.val_ds = None

    def load_data(self):
        """
        Downloads and loads the dataset from TensorFlow and splits it into training and validation datasets.
        """
        print("[INFO] Downloading dataset...")
        dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
        zip_path = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=dataset_url, extract=True)

        base_dir = os.path.join(os.path.dirname(zip_path), 'cats_and_dogs_filtered')
        train_dir = os.path.join(base_dir, 'train')
        val_dir = os.path.join(base_dir, 'validation')

        print("[INFO] Creating datasets...")
        self.train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=self.img_size, batch_size=self.batch_size)
        self.val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=self.img_size, batch_size=self.batch_size)

        print("[INFO] Classes found:", self.train_ds.class_names)

    def build_model(self):
        """
        Builds the CNN model architecture using Keras Sequential API.
        """
        print("[INFO] Building model...")
        self.model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255, input_shape=(*self.img_size, 3)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def train(self, epochs=5):
        """
        Trains the CNN model on the training dataset and validates on the validation dataset.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call build_model() first.")
        
        print(f"[INFO] Training for {epochs} epochs...")
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs
        )

    def save_model(self, path='cats_vs_dogs_model.h5'):
        """
        Saves the trained model to a file.
        """
        print(f"[INFO] Saving model to '{path}'...")
        self.model.save(path)

    def get_history(self):
        """
        Returns the training history object.
        """
        return self.history
