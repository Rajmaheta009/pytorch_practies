# Keras full example with batch-wise loss printing
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

# Load and preprocess
data = load_iris()
X = StandardScaler().fit_transform(data.data)
y = to_categorical(data.target)  # One-hot for categorical_crossentropy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom Callback to print per-batch loss
class BatchLogger(Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(f"  Batch {batch} - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
print("🧠 Keras Training (Loss per batch):")
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=16,
    callbacks=[BatchLogger()],
    verbose=1  # Hides the default epoch summary
)
