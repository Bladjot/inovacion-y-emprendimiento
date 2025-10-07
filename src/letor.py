import numpy as np

# Cargar los datos
data = np.load("data/processed/train_sequences.npz")
X = data["X"]
y = np.load("data/processed/labels.npy")

print("✅ Dataset cargado correctamente")
print("Forma de X:", X.shape)
print("Forma de y:", y.shape)
print("Ejemplo de una secuencia:", X[0].shape)
print("Primera etiqueta:", y[0])
