"""
train_model.py
Entrena el modelo CNN-LSTM con el dataset preprocesado.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model_lstm import build_cnn_lstm

# === Configuración general ===
DATA_PATH = "data/processed/train_sequences.npz"
LABEL_PATH = "data/processed/labels.npy"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === Cargar datos ===
print("📂 Cargando dataset...")
data = np.load(DATA_PATH)
X = data["X"]
y = np.load(LABEL_PATH)

print("Forma de X:", X.shape)
print("Forma de y:", y.shape)

# === División train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Crear modelo ===
model = build_cnn_lstm(input_shape=X.shape[1:])
model.summary()

# === Callbacks ===
checkpoint_path = os.path.join(CHECKPOINT_DIR, "gait_dementia_model.h5")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
]

# === Entrenamiento ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

# === Evaluación final ===
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Precisión final: {acc*100:.2f}%  |  Pérdida: {loss:.4f}")

# === Graficar desempeño ===
plt.figure(figsize=(8,4))
plt.plot(history.history["accuracy"], label="Train acc")
plt.plot(history.history["val_accuracy"], label="Val acc")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, "training_accuracy.png"))
plt.show()
