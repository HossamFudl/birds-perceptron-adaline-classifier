import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURES = ["gender", "body_mass", "beak_length", "beak_depth", "fin_length"]
CLASSES = ["A", "B", "C"]
GENDER_MAP = {"male": 1.0, "female": -1.0, "na": 0.0, "NA": 0.0}


@dataclass
class TrainingResult:
    weights: np.ndarray
    bias: float
    train_mse_history: List[float]
    class_to_target: Dict[str, int]
    mean: np.ndarray
    std: np.ndarray


def signum(v: float) -> int:
    return 1 if v >= 0 else -1


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = ["gender", "body_mass", "beak_length", "beak_depth", "fin_length", "bird category"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    clean_gender = df["gender"].astype(str).str.strip().str.lower()
    df["gender"] = clean_gender.map(GENDER_MAP).fillna(0.0).astype(float)

    numeric_cols = ["body_mass", "beak_length", "beak_depth", "fin_length"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isnull().sum().sum() > 0:
        raise ValueError("Dataset has non-numeric values in numeric feature columns.")

    return df


def split_train_test_by_class(
    df: pd.DataFrame,
    class_pair: Tuple[str, str],
    n_train_per_class: int = 30,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    c1, c2 = class_pair
    part1 = df[df["bird category"] == c1]
    part2 = df[df["bird category"] == c2]

    if len(part1) < 50 or len(part2) < 50:
        raise ValueError("Each selected class must have 50 samples.")

    rng = np.random.default_rng(seed)
    idx1 = np.array(part1.index)
    idx2 = np.array(part2.index)
    rng.shuffle(idx1)
    rng.shuffle(idx2)

    train_idx = list(idx1[:n_train_per_class]) + list(idx2[:n_train_per_class])
    test_idx = list(idx1[n_train_per_class:]) + list(idx2[n_train_per_class:])

    train_df = df.loc[train_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = df.loc[test_idx].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return train_df, test_df


def to_binary_targets(y_labels: np.ndarray, positive_class: str) -> np.ndarray:
    return np.where(y_labels == positive_class, 1, -1)


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std
    return X_train_s, X_test_s, mean, std


def train_perceptron(
    X: np.ndarray,
    d: np.ndarray,
    eta: float,
    epochs: int,
    mse_threshold: float,
    use_bias: bool,
    rng_seed: int = 7,
) -> Tuple[np.ndarray, float, List[float]]:
    rng = np.random.default_rng(rng_seed)
    w = rng.normal(0.0, 0.01, size=X.shape[1])
    b = float(rng.normal(0.0, 0.01)) if use_bias else 0.0
    mse_hist: List[float] = []

    for _ in range(epochs):
        sq_errors = []
        for xi, di in zip(X, d):
            v = float(np.dot(w, xi) + (b if use_bias else 0.0))
            yi = signum(v)
            err = di - yi
            w = w + eta * err * xi
            if use_bias:
                b = b + eta * err
            sq_errors.append(err ** 2)

        mse = float(np.mean(sq_errors))
        mse_hist.append(mse)
        if mse <= mse_threshold:
            break

    return w, b, mse_hist


def train_adaline(
    X: np.ndarray,
    d: np.ndarray,
    eta: float,
    epochs: int,
    mse_threshold: float,
    use_bias: bool,
    rng_seed: int = 11,
) -> Tuple[np.ndarray, float, List[float]]:
    rng = np.random.default_rng(rng_seed)
    w = rng.normal(0.0, 0.01, size=X.shape[1])
    b = float(rng.normal(0.0, 0.01)) if use_bias else 0.0
    mse_hist: List[float] = []

    for _ in range(epochs):
        sq_errors = []
        for xi, di in zip(X, d):
            v = float(np.dot(w, xi) + (b if use_bias else 0.0))
            yi = v
            err = di - yi
            w = w + eta * err * xi
            if use_bias:
                b = b + eta * err
            sq_errors.append(err ** 2)

        mse = float(np.mean(sq_errors))
        mse_hist.append(mse)
        if mse <= mse_threshold:
            break

    return w, b, mse_hist


def predict_binary(X: np.ndarray, w: np.ndarray, b: float, use_bias: bool) -> np.ndarray:
    v = X @ w + (b if use_bias else 0.0)
    return np.where(v >= 0.0, 1, -1)


def confusion_matrix_and_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    labels = [-1, 1]
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        i = labels.index(int(t))
        j = labels.index(int(p))
        cm[i, j] += 1
    acc = float(np.trace(cm) / np.sum(cm))
    return cm, acc


class BirdClassifierGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Birds Task 1 - Perceptron / Adaline")
        self.root.geometry("980x760")

        self.data = load_and_preprocess("birds(in).csv")
        self.result: TrainingResult | None = None
        self.selected_features: Tuple[str, str] | None = None
        self.selected_classes: Tuple[str, str] | None = None
        self.use_bias = tk.BooleanVar(value=True)
        self.algorithm = tk.StringVar(value="perceptron")

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Feature 1").grid(row=0, column=0, sticky="w", pady=4)
        self.feature1 = ttk.Combobox(frame, values=FEATURES, state="readonly")
        self.feature1.grid(row=0, column=1, sticky="ew", pady=4)
        self.feature1.set("beak_length")

        ttk.Label(frame, text="Feature 2").grid(row=1, column=0, sticky="w", pady=4)
        self.feature2 = ttk.Combobox(frame, values=FEATURES, state="readonly")
        self.feature2.grid(row=1, column=1, sticky="ew", pady=4)
        self.feature2.set("beak_depth")

        ttk.Label(frame, text="Class 1").grid(row=0, column=2, sticky="w", padx=(20, 0), pady=4)
        self.class1 = ttk.Combobox(frame, values=CLASSES, state="readonly")
        self.class1.grid(row=0, column=3, sticky="ew", pady=4)
        self.class1.set("A")

        ttk.Label(frame, text="Class 2").grid(row=1, column=2, sticky="w", padx=(20, 0), pady=4)
        self.class2 = ttk.Combobox(frame, values=CLASSES, state="readonly")
        self.class2.grid(row=1, column=3, sticky="ew", pady=4)
        self.class2.set("B")

        ttk.Label(frame, text="Learning rate (eta)").grid(row=2, column=0, sticky="w", pady=4)
        self.eta_entry = ttk.Entry(frame)
        self.eta_entry.grid(row=2, column=1, sticky="ew", pady=4)
        self.eta_entry.insert(0, "0.001")

        ttk.Label(frame, text="Epochs (m)").grid(row=2, column=2, sticky="w", padx=(20, 0), pady=4)
        self.epochs_entry = ttk.Entry(frame)
        self.epochs_entry.grid(row=2, column=3, sticky="ew", pady=4)
        self.epochs_entry.insert(0, "100")

        ttk.Label(frame, text="MSE threshold").grid(row=3, column=0, sticky="w", pady=4)
        self.mse_entry = ttk.Entry(frame)
        self.mse_entry.grid(row=3, column=1, sticky="ew", pady=4)
        self.mse_entry.insert(0, "0.01")

        ttk.Checkbutton(frame, text="Use bias", variable=self.use_bias).grid(row=3, column=2, sticky="w", padx=(20, 0), pady=4)

        algo_box = ttk.LabelFrame(frame, text="Algorithm", padding=8)
        algo_box.grid(row=4, column=0, columnspan=4, sticky="ew", pady=8)
        ttk.Radiobutton(algo_box, text="Perceptron", variable=self.algorithm, value="perceptron").pack(side="left", padx=8)
        ttk.Radiobutton(algo_box, text="Adaline", variable=self.algorithm, value="adaline").pack(side="left", padx=8)

        actions = ttk.Frame(frame)
        actions.grid(row=5, column=0, columnspan=4, sticky="ew", pady=6)
        ttk.Button(actions, text="Train + Test", command=self.train_and_test).pack(side="left", padx=4)
        ttk.Button(actions, text="Plot Decision Boundary", command=self.plot_decision_boundary).pack(side="left", padx=4)

        predict_box = ttk.LabelFrame(frame, text="Classify a single sample", padding=8)
        predict_box.grid(row=6, column=0, columnspan=4, sticky="ew", pady=8)

        ttk.Label(predict_box, text="x1").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.x1_entry = ttk.Entry(predict_box)
        self.x1_entry.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Label(predict_box, text="x2").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        self.x2_entry = ttk.Entry(predict_box)
        self.x2_entry.grid(row=0, column=3, sticky="ew", padx=4, pady=4)
        ttk.Button(predict_box, text="Predict Sample", command=self.predict_sample).grid(row=0, column=4, padx=6, pady=4)
        self.predict_label = ttk.Label(predict_box, text="Predicted class: -")
        self.predict_label.grid(row=0, column=5, padx=6, pady=4)

        self.output = tk.Text(frame, height=22, width=120)
        self.output.grid(row=7, column=0, columnspan=4, sticky="nsew", pady=8)

        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)
        frame.rowconfigure(7, weight=1)
        predict_box.columnconfigure(1, weight=1)
        predict_box.columnconfigure(3, weight=1)

    def _read_settings(self) -> Tuple[str, str, str, str, float, int, float, bool]:
        f1 = self.feature1.get().strip()
        f2 = self.feature2.get().strip()
        c1 = self.class1.get().strip()
        c2 = self.class2.get().strip()
        if f1 == f2:
            raise ValueError("Please select two different features.")
        if c1 == c2:
            raise ValueError("Please select two different classes.")

        eta = float(self.eta_entry.get().strip())
        epochs = int(self.epochs_entry.get().strip())
        mse_threshold = float(self.mse_entry.get().strip())
        if eta <= 0 or epochs <= 0:
            raise ValueError("Learning rate and epochs must be positive.")
        if mse_threshold < 0:
            raise ValueError("MSE threshold must be non-negative.")
        return f1, f2, c1, c2, eta, epochs, mse_threshold, self.use_bias.get()

    def _train_core(self) -> Tuple[TrainingResult, pd.DataFrame, pd.DataFrame]:
        f1, f2, c1, c2, eta, epochs, mse_threshold, use_bias = self._read_settings()
        train_df, test_df = split_train_test_by_class(self.data, (c1, c2), 30, seed=42)

        X_train = train_df[[f1, f2]].to_numpy(dtype=float)
        y_train_label = train_df["bird category"].to_numpy()
        d_train = to_binary_targets(y_train_label, positive_class=c1)
        X_test = test_df[[f1, f2]].to_numpy(dtype=float)
        X_train_s, _, mean, std = standardize_train_test(X_train, X_test)

        if self.algorithm.get() == "perceptron":
            w, b, mse_hist = train_perceptron(X_train_s, d_train, eta, epochs, mse_threshold, use_bias)
        else:
            w, b, mse_hist = train_adaline(X_train_s, d_train, eta, epochs, mse_threshold, use_bias)

        result = TrainingResult(
            weights=w,
            bias=b,
            train_mse_history=mse_hist,
            class_to_target={c1: 1, c2: -1},
            mean=mean,
            std=std,
        )
        self.selected_features = (f1, f2)
        self.selected_classes = (c1, c2)
        self.result = result
        return result, train_df, test_df

    def train_and_test(self) -> None:
        try:
            result, train_df, test_df = self._train_core()
            f1, f2 = self.selected_features
            c1, c2 = self.selected_classes
            use_bias = self.use_bias.get()

            X_test = test_df[[f1, f2]].to_numpy(dtype=float)
            y_test_label = test_df["bird category"].to_numpy()
            d_test = to_binary_targets(y_test_label, positive_class=c1)
            X_test_s = (X_test - result.mean) / result.std
            y_pred = predict_binary(X_test_s, result.weights, result.bias, use_bias)
            cm, acc = confusion_matrix_and_accuracy(d_test, y_pred)

            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, f"Algorithm: {self.algorithm.get().capitalize()}\n")
            self.output.insert(tk.END, f"Features: {f1}, {f2}\n")
            self.output.insert(tk.END, f"Classes: {c1}(+1), {c2}(-1)\n")
            self.output.insert(tk.END, f"Bias used: {use_bias}\n")
            self.output.insert(tk.END, f"Weights: {result.weights}\n")
            self.output.insert(tk.END, f"Bias: {result.bias:.6f}\n")
            self.output.insert(tk.END, f"Epochs run: {len(result.train_mse_history)}\n")
            self.output.insert(tk.END, f"Final train MSE: {result.train_mse_history[-1]:.6f}\n\n")
            self.output.insert(tk.END, "Confusion matrix (rows=true, cols=pred):\n")
            self.output.insert(tk.END, f"          pred {c2}(-1)   pred {c1}(+1)\n")
            self.output.insert(tk.END, f"true {c2}(-1)      {cm[0, 0]:>3}          {cm[0, 1]:>3}\n")
            self.output.insert(tk.END, f"true {c1}(+1)      {cm[1, 0]:>3}          {cm[1, 1]:>3}\n\n")
            self.output.insert(tk.END, f"Accuracy: {acc * 100:.2f}%\n")

            messagebox.showinfo("Done", "Training and testing completed.")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def plot_decision_boundary(self) -> None:
        if self.result is None:
            messagebox.showwarning("Not trained", "Run Train + Test first.")
            return

        f1, f2 = self.selected_features
        c1, c2 = self.selected_classes
        train_df, test_df = split_train_test_by_class(self.data, (c1, c2), 30, seed=42)
        merged = pd.concat([train_df, test_df], ignore_index=True)

        cls1 = merged[merged["bird category"] == c1]
        cls2 = merged[merged["bird category"] == c2]
        x1 = merged[f1].to_numpy()
        x_min, x_max = x1.min(), x1.max()
        xs = np.linspace(x_min - 1.0, x_max + 1.0, 200)

        w1, w2 = self.result.weights[0], self.result.weights[1]
        b = self.result.bias if self.use_bias.get() else 0.0

        plt.figure(figsize=(8, 6))
        plt.scatter(cls1[f1], cls1[f2], c="tab:blue", label=f"Class {c1}", alpha=0.8)
        plt.scatter(cls2[f1], cls2[f2], c="tab:orange", label=f"Class {c2}", alpha=0.8)

        if abs(w2) < 1e-12:
            if abs(w1) > 1e-12:
                x_line = self.result.mean[0] + self.result.std[0] * (-b / w1)
                plt.axvline(x_line, color="green", linestyle="--", label="Decision boundary")
        else:
            z1 = (xs - self.result.mean[0]) / self.result.std[0]
            z2 = -(w1 * z1 + b) / w2
            ys = self.result.mean[1] + self.result.std[1] * z2
            plt.plot(xs, ys, "g--", linewidth=2, label="Decision boundary")

        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title(f"{self.algorithm.get().capitalize()} decision boundary ({c1} vs {c2})")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    def predict_sample(self) -> None:
        if self.result is None or self.selected_classes is None:
            messagebox.showwarning("Not trained", "Run Train + Test first.")
            return
        try:
            x1 = float(self.x1_entry.get().strip())
            x2 = float(self.x2_entry.get().strip())
            x = np.array([[x1, x2]], dtype=float)
            x_s = (x - self.result.mean) / self.result.std
            pred = predict_binary(x_s, self.result.weights, self.result.bias, self.use_bias.get())[0]
            c1, c2 = self.selected_classes
            predicted_class = c1 if pred == 1 else c2
            self.predict_label.config(text=f"Predicted class: {predicted_class}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


def main() -> None:
    root = tk.Tk()
    BirdClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
