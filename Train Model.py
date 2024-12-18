import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random


TRAIN_DIR = r"D:\TUGAS KULIAHHH\SEM 5\12 2024\CNN train and test\train"
LOG_FILE = "training_status.log"

with open(LOG_FILE, "w") as f:
    f.write("Training started\n")

class TrainingLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open(LOG_FILE, "a") as f:
            f.write(f"Epoch {epoch + 1}/{self.params['epochs']}, "
                    f"Loss: {logs['loss']:.4f}, \n"
                    f"Accuracy: {logs['accuracy']:.4f}, "
                    f"Val Loss: {logs['val_loss']:.4f}, "
                    f"Val Accuracy: {logs['val_accuracy']:.4f}\n")
            
tsb = tf.keras.callbacks.TensorBoard(log_dir="logs")


classes = [i for i in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, i))]


train = []
for i in classes:
    current_path = os.path.join(TRAIN_DIR, i)
    current_class = classes.index(i)
    for j in os.listdir(current_path):
        try:
            img_path = os.path.join(current_path, j)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                train.append([img, current_class])
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue


random.shuffle(train)


x = []
y = []
for img, label in train:
    x.append(img)
    y.append(label)


x = np.array(x).reshape(-1, 128, 128, 1) / 255.0
y = np.array(y)
print(f"Total Train Images : {len(x)}")


model = tf.keras.models.Sequential([
    #kovolusi 1
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    tf.keras.layers.MaxPooling2D(),
    #kovolusi 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    #konvolusi 3
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    #flat to 1D vector
    tf.keras.layers.Flatten(),
    #Dense 1
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    #Dense 2
    tf.keras.layers.Dense(256, activation='relu'),
    #Output 
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.summary()


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


history = model.fit(x, y, epochs=30, validation_split=0.1, callbacks=[tsb, TrainingLogger()])


plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="best")
plt.show()


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="best")
plt.show()


model.save("64x3-cards.h5")
