from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app import (
    FEATURES,
    CLASSES,
    load_and_preprocess,
    split_train_test_by_class,
    to_binary_targets,
    train_perceptron,
    train_adaline,
    predict_binary,
    confusion_matrix_and_accuracy,
    standardize_train_test,
)


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_one(
    df: pd.DataFrame,
    algo: str,
    feature_pair: tuple[str, str],
    class_pair: tuple[str, str],
    eta: float = 0.001,
    epochs: int = 100,
    mse_threshold: float = 0.01,
    use_bias: bool = True,
):
    c1, c2 = class_pair
    f1, f2 = feature_pair
    train_df, test_df = split_train_test_by_class(df, class_pair, n_train_per_class=30, seed=42)

    X_train = train_df[[f1, f2]].to_numpy(dtype=float)
    y_train = to_binary_targets(train_df["bird category"].to_numpy(), c1)
    X_test = test_df[[f1, f2]].to_numpy(dtype=float)
    y_test = to_binary_targets(test_df["bird category"].to_numpy(), c1)
    X_train_s, X_test_s, mean, std = standardize_train_test(X_train, X_test)

    if algo == "perceptron":
        w, b, mse_hist = train_perceptron(X_train_s, y_train, eta, epochs, mse_threshold, use_bias)
    else:
        w, b, mse_hist = train_adaline(X_train_s, y_train, eta, epochs, mse_threshold, use_bias)

    y_pred = predict_binary(X_test_s, w, b, use_bias)
    cm, acc = confusion_matrix_and_accuracy(y_test, y_pred)

    return {
        "algorithm": algo,
        "feature_1": f1,
        "feature_2": f2,
        "class_1_pos": c1,
        "class_2_neg": c2,
        "accuracy": acc,
        "final_train_mse": mse_hist[-1],
        "epochs_run": len(mse_hist),
        "w1": w[0],
        "w2": w[1],
        "b": b,
        "mean_1": mean[0],
        "mean_2": mean[1],
        "std_1": std[0],
        "std_2": std[1],
        "cm_tn": int(cm[0, 0]),
        "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]),
        "cm_tp": int(cm[1, 1]),
    }


def plot_result(df: pd.DataFrame, row: pd.Series, save_path: Path):
    c1 = row["class_1_pos"]
    c2 = row["class_2_neg"]
    f1 = row["feature_1"]
    f2 = row["feature_2"]
    w1 = row["w1"]
    w2 = row["w2"]
    b = row["b"]
    mean_1 = row["mean_1"]
    mean_2 = row["mean_2"]
    std_1 = row["std_1"]
    std_2 = row["std_2"]

    pair_df = df[df["bird category"].isin([c1, c2])]
    cls1 = pair_df[pair_df["bird category"] == c1]
    cls2 = pair_df[pair_df["bird category"] == c2]
    xs = np.linspace(pair_df[f1].min() - 1, pair_df[f1].max() + 1, 200)

    plt.figure(figsize=(8, 6))
    plt.scatter(cls1[f1], cls1[f2], c="tab:blue", alpha=0.8, label=f"Class {c1}")
    plt.scatter(cls2[f1], cls2[f2], c="tab:orange", alpha=0.8, label=f"Class {c2}")

    if abs(w2) < 1e-12:
        if abs(w1) > 1e-12:
            x_line = mean_1 + std_1 * (-b / w1)
            plt.axvline(x_line, color="green", linestyle="--", label="Decision boundary")
    else:
        z1 = (xs - mean_1) / std_1
        z2 = -(w1 * z1 + b) / w2
        ys = mean_2 + std_2 * z2
        plt.plot(xs, ys, "g--", linewidth=2, label="Decision boundary")

    plt.title(f'{row["algorithm"].capitalize()} | {c1} vs {c2} | {f1} & {f2}')
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    df = load_and_preprocess("birds(in).csv")

    class_pairs = list(combinations(CLASSES, 2))
    feature_pairs = list(combinations(FEATURES, 2))
    records = []

    for algo in ["perceptron", "adaline"]:
        for cp in class_pairs:
            for fp in feature_pairs:
                rec = run_one(
                    df=df,
                    algo=algo,
                    feature_pair=fp,
                    class_pair=cp,
                    eta=0.001,
                    epochs=100,
                    mse_threshold=0.01,
                    use_bias=True,
                )
                records.append(rec)

    results = pd.DataFrame(records).sort_values("accuracy", ascending=False).reset_index(drop=True)
    results.to_csv(OUTPUT_DIR / "all_results.csv", index=False)

    top5 = results.head(5)
    top5.to_csv(OUTPUT_DIR / "top5_results.csv", index=False)

    for i, row in top5.iterrows():
        out_name = f'top{i + 1}_{row["algorithm"]}_{row["class_1_pos"]}{row["class_2_neg"]}_{row["feature_1"]}_{row["feature_2"]}.png'
        plot_result(df, row, OUTPUT_DIR / out_name)

    print("Saved:")
    print("- outputs/all_results.csv")
    print("- outputs/top5_results.csv")
    print("- top 5 plots in outputs/")


if __name__ == "__main__":
    main()
