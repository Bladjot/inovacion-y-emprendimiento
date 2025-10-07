import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Cargar datos y modelo ===
print("📦 Cargando modelo y datos...")
model = load_model("../checkpoints/gait_dementia_model.h5")

# Cargar los datos preprocesados (ajusta las rutas si es necesario)
X = np.load("../data/X.npy")
y = np.load("../data/y.npy")

# === Generar predicciones ===
print("🧠 Generando predicciones...")
y_pred = (model.predict(X) > 0.5).astype(int)

# === Calcular métricas ===
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

# === Calcular métricas ===
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
auc = roc_auc_score(y, y_pred)

print("\n===== 📊 RESULTADOS =====")
print(f"Verdaderos Positivos (VP): {tp}")
print(f"Falsos Positivos (FP):     {fp}")
print(f"Falsos Negativos (FN):     {fn}")
print(f"Verdaderos Negativos (VN): {tn}")
print("-------------------------------")
print(f"Precisión:     {precision:.3f}")
print(f"Recall:        {recall:.3f}")
print(f"F1-Score:      {f1:.3f}")
print(f"Especificidad: {specificity:.3f}")
print(f"AUC:           {auc:.3f}")

# === Clasificación detallada ===
print("\n📋 Clasification report:")
print(classification_report(y, y_pred))

# === Graficar matriz de confusión ===
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Matriz de Confusión")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
