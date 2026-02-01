import numpy as np
import tensorflow as tf

IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "bishop",
    "king",
    "knight",
    "pawn",
    "queen",
    "rook"
]

def preprocess_image(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    return img_array

def load_and_preprocess_image(path: str):
    image = tf.keras.preprocessing.image.load_img(
        path, target_size=IMAGE_SIZE
    )

    return preprocess_image(image)

def classify(model, image_path: str):
    img = load_and_preprocess_image(image_path)

    predictions = model.predict(img)

    probabilities = predictions[0]
    class_index = int(np.argmax(probabilities))

    label = CLASS_NAMES[class_index]
    confidence = float(probabilities[class_index])

    return  label, confidence