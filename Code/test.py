from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load the trained model
model = load_model('sign_language_model.h5')
print("✅ Model Loaded Successfully")

# Class names must match the model's training labels
classes = ["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]

# Function to classify a single image
def classify(img_path):
    try:
        img = image.load_img(img_path, target_size=(256, 256), color_mode='grayscale')
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array, verbose=0)
        prediction = classes[np.argmax(result)]

        print(f"{os.path.basename(img_path)} => {prediction} ({np.max(result):.2f})")
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")

# Path to the test directory
test_dir = r"D:\MS-Edunet Foundation Internship\SignLanguageDetectionDeepLearningProject-main-P4\HandGestureDataset\test\TWO"

# Loop through all test images and classify
for root, _, files in os.walk(test_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            classify(img_path)
