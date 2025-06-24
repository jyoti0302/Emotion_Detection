import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import cv2
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1️⃣ Load your model architecture and weights
# ----------------------------------------------------
# Load model structure
with open("model_emotion.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights("emotion_model.weights.h5")

# ----------------------------------------------------
# 2️⃣ Load and preprocess image
# ----------------------------------------------------
def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # shape (48, 48, 1)
    img = np.expand_dims(img, axis=0)   # shape (1, 48, 48, 1)
    return img

img_path = "train/surprise/Training_8796.jpg"  
img_tensor = preprocess(img_path)

# ----------------------------------------------------
# 3️⃣ Predict class
# ----------------------------------------------------
preds = model.predict(img_tensor)
predicted_class = np.argmax(preds[0])
print("Predicted class:", predicted_class)

# ----------------------------------------------------
# 4️⃣ Smooth Grad-CAM++ function
# ----------------------------------------------------
@tf.function
def get_gradients(image, model, class_index, last_conv_layer):
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            conv_outputs, predictions = tf.keras.models.Model(
                [model.inputs],
                [model.get_layer(last_conv_layer).output, model.output]
            )(image)
            
            loss = predictions[:, class_index]
        grads = tape2.gradient(loss, conv_outputs)
    grads2 = tape1.gradient(grads, conv_outputs)
    return conv_outputs, grads, grads2

def smooth_grad_cam_plus(model, image, class_index, layer_name):
    conv_outputs, grads, grads2 = get_gradients(image, model, class_index, layer_name)
    
    conv_outputs = conv_outputs[0].numpy()  # shape: (H, W, C)
    grads = grads[0].numpy()
    grads2 = grads2[0].numpy()

    weights = np.sum(grads ** 2, axis=(0, 1))
    weights /= (2 * weights + np.sum(conv_outputs * grads2, axis=(0, 1)) + 1e-10)
    
    cam = np.sum(weights * conv_outputs, axis=-1)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (48, 48))
    cam -= np.min(cam)
    cam /= np.max(cam) + 1e-10

    return cam

# ----------------------------------------------------
# 5️⃣ Automatically apply on all Conv2D layers
# ----------------------------------------------------

# Extract Conv2D layers only
conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
print("Conv layers found:", conv_layers)

# Load original image once
orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
orig_img = cv2.resize(orig_img, (48, 48))
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)

# Loop through each conv layer
for layer_name in conv_layers:
    print(f"\nProcessing layer: {layer_name}")
    
    heatmap = smooth_grad_cam_plus(model, img_tensor, predicted_class, layer_name)
    
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)

    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Smooth Grad-CAM++\nLayer: {layer_name}")
    plt.show()
