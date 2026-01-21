from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def load_wdbc(path):
    """
    Load Breast Cancer Wisconsin (Diagnostic) from local wdbc.data. 
    Columns (per UCI description):
    0: ID
    1: Diagnosis (M/B)
    2-31: 30 real-valued features.
    """
    # wdbc.data is comma-separated. 
    raw = np.loadtxt(path, delimiter=",", dtype=str)

    # First column: ID, second: label
    ids = raw[:, 0]
    labels_str = raw[:, 1]
    features = raw[:, 2:].astype(float)

    # Map labels: M -> +1, B -> -1 (for AdaBoost math)
    label_map = {"M": 1, "B": -1}
    labels = np.vectorize(label_map.get)(labels_str)

    return features, labels, ids, labels_str


def to_pca_2d(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum()
    return X_2d, explained, scaler, pca

if __name__ == "__main__":
    X, y, ids, y_str = load_wdbc(path="../dataset/wdbc.data")
    print("X shape:", X.shape)      # Expect (569, 30) 
    print("y classes:", np.unique(y))

    X_2d, explained, _, _ = to_pca_2d(X)
    print("X_2d shape:", X_2d.shape)
    print(f"Variance explained by 2D PCA: {explained:.2%}")
