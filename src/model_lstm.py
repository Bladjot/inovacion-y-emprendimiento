"""
model_lstm.py
-------------
Modelo híbrido CNN-LSTM para analizar secuencias de marcha
y detectar posibles signos de deterioro cognitivo.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_lstm(input_shape=(100, 3, 3)):
    model = models.Sequential([
        # --- CNN para patrones espaciales (por frame) ---
        layers.TimeDistributed(
            layers.Conv1D(32, kernel_size=2, activation='relu'),
            input_shape=input_shape
        ),
        layers.TimeDistributed(layers.MaxPooling1D(1)),
        layers.TimeDistributed(layers.Flatten()),

        # --- LSTM para patrones temporales ---
        layers.LSTM(128, return_sequences=False, dropout=0.3),

        # --- Capa densa final ---
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # salida binaria
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    model = build_cnn_lstm()
    model.summary()
