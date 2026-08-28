import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image



MODEL_FILE = "waste_classifier.keras"
CLASSES_FILE = "classes.txt"
IMAGE_SIZE = (128, 128)



if not os.path.exists(MODEL_FILE):
    print("Model file not found!")
    print("Please run train_model.py first.")
    exit()



print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_FILE)



with open(CLASSES_FILE, "r") as file:
    class_names = [line.strip() for line in file.readlines()]

print("Model loaded successfully!")



def predict_image(image_path):

    if not os.path.exists(image_path):
        print("\nImage not found!")
        return

    img = image.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = float(np.max(prediction) * 100)

    print("\n========================================")
    print("           PREDICTION RESULT")
    print("========================================")

    print("Image:", image_path)
    print("Waste Type:", predicted_class)
    print("Confidence:", round(confidence, 2), "%")

    print("========================================")



print("\n========================================")
print("      SMART WASTE CLASSIFICATION")
print("========================================")

while True:

    image_path = input(
    "\nEnter image path (or type 'exit' to quit): "
    ).strip().strip('"')

    if image_path.lower() == "exit":
        print("\nProgram finished.")
        break

    predict_image(image_path)
