from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _load_train(path: Path) -> Tuple[pd.Series, np.ndarray, pd.DataFrame]:
    """Load CUP training set: return ids, inputs and targets."""
    df = pd.read_csv(path, comment="#", header=None)
    ids = df.iloc[:, 0].astype(int)
    targets = df.iloc[:, -4:].copy()
    inputs = df.iloc[:, 1:-4].values
    return ids, inputs, targets


def _load_test(path: Path) -> Tuple[pd.Series, np.ndarray]:
    """Load CUP test set: return ids and inputs."""
    df = pd.read_csv(path, comment="#", header=None)
    ids = df.iloc[:, 0].astype(int)
    inputs = df.iloc[:, 1:].values
    return ids, inputs


def transform_to_two_components(
    train_path: Path,
    test_path: Optional[Path] = None,
) -> Tuple[Path, Optional[Path]]:
    """Standardize inputs, fit PCA (2 comps) on train, transform train and test."""
    train_ids, X_train, y_train = _load_train(train_path)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=2, random_state=0)
    train_components = pca.fit_transform(X_train_scaled)

    train_out_path = train_path.with_name(f"{train_path.stem}-pca2.csv")
    train_columns = ["id", "pc1", "pc2"] + [f"t{i+1}" for i in range(y_train.shape[1])]
    train_payload = np.column_stack([train_ids.values, train_components, y_train.values])
    pd.DataFrame(train_payload, columns=train_columns).to_csv(train_out_path, index=False)

    test_out_path: Optional[Path] = None
    if test_path and test_path.exists():
        test_ids, X_test = _load_test(test_path)
        test_components = pca.transform(scaler.transform(X_test))
        test_out_path = test_path.with_name(f"{test_path.stem}-pca2.csv")
        pd.DataFrame(
            {
                "id": test_ids.values,
                "pc1": test_components[:, 0],
                "pc2": test_components[:, 1],
            }
        ).to_csv(test_out_path, index=False)

    explained = pca.explained_variance_ratio_
    print(
        f"Train shape: {X_train.shape} -> {train_components.shape} | "
        f"Explained variance ratio: {explained}"
    )

    return train_out_path, test_out_path


if __name__ == "__main__":
    base_dir = Path("data/CUP")
    train_file = base_dir / "ML-CUP25-TR.csv"
    test_file = base_dir / "ML-CUP25-TS.csv"
    transform_to_two_components(train_file, test_file)
