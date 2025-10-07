"""
preprocess_csv_dataset.py
-------------------------
Convierte los archivos dataSetSkel_composite.csv del dataset de Alzheimer
en tensores NumPy (T, J, 3) y genera etiquetas binarias.

Estructura esperada:
dataset-master/
 ├── 1/          -> Grupo 1 (p. ej. pacientes con Alzheimer)
 │   ├── mov1/
 │   │   ├── seq_1/dataSetSkel_composite.csv
 │   │   ├── seq_2/dataSetSkel_composite.csv
 │   │   └── ...
 │   ├── mov2/
 │   └── ...
 ├── 2/          -> Grupo 2 (controles sanos)
 └── ...

Salida:
data/processed/train_sequences.npz
data/processed/labels.npy
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET_ROOT = "dataset-master"
OUTPUT_DIR = "data/processed"
SEQ_LEN = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_csv_sequence(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df = df.select_dtypes(include=[np.number])

    # Eliminar primera columna (índice de frame o tiempo)
    if df.shape[1] > 1:
        df = df.iloc[:, 1:]

    arr = df.to_numpy()
    num_cols = arr.shape[1]

    # Verificar que sea múltiplo de 3
    if num_cols % 3 != 0:
        raise ValueError(f"{csv_path} tiene {num_cols} columnas (sin incluir frame), no múltiplo de 3.")

    num_joints = num_cols // 3
    arr = arr.reshape(len(df), num_joints, 3)  # (T, J, 3)
    return arr


def pad_or_truncate(seq, target_len):
    T = len(seq)
    if T >= target_len:
        return seq[:target_len]
    pad_len = target_len - T
    pad = np.repeat(seq[-1][np.newaxis, :, :], pad_len, axis=0)
    return np.concatenate([seq, pad], axis=0)


def build_dataset():
    X, y = [], []
    group_folders = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))])

    for group in group_folders:
        group_path = os.path.join(DATASET_ROOT, group)
        label = 1 if group == "1" else 0
        print(f"\nProcesando grupo {group_path} (etiqueta {label})")

        for root, dirs, files in os.walk(group_path):
            for file in files:
                if file == "dataSetSkel_composite.csv":
                    csv_path = os.path.join(root, file)
                    try:
                        seq = load_csv_sequence(csv_path)
                        seq = pad_or_truncate(seq, SEQ_LEN)
                        X.append(seq)
                        y.append(label)
                    except Exception as e:
                        print(f"[ADVERTENCIA] {csv_path}: {e}")

    if not X:
        print("\n⚠️ No se procesaron secuencias válidas.")
        return

    X = np.array(X)
    y = np.array(y)

    np.savez_compressed(os.path.join(OUTPUT_DIR, "train_sequences.npz"), X=X)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), y)

    print(f"\n✅ Dataset final: {X.shape[0]} secuencias, forma {X.shape[1:]} → {OUTPUT_DIR}")


if __name__ == "__main__":
    build_dataset()