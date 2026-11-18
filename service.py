"""Train, evaluate, or serve one 8×8 digit through a JSON command line."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

MODEL_VERSION = "digits-logreg-v1"
SEED = 42


def split():
    x, y = load_digits(return_X_y=True)
    return train_test_split(
        x, y, np.arange(len(y)), test_size=0.25, random_state=SEED, stratify=y
    )


def train(path):
    train_x, _, train_y, _, train_ids, _ = split()
    model = LogisticRegression(C=1.0, max_iter=2000).fit(train_x / 16, train_y)
    baseline = DummyClassifier(strategy="most_frequent").fit(train_x, train_y)
    joblib.dump({"version": MODEL_VERSION, "model": model, "baseline": baseline}, path)
    return {"model_version": MODEL_VERSION, "train_count": len(train_ids), "seed": SEED}


def evaluate(bundle):
    _, x, _, y, _, ids = split()
    predicted = bundle["model"].predict(x / 16)
    baseline = bundle["baseline"].predict(x)

    def scores(values):
        return {
            "correct": int(np.sum(y == values)),
            "accuracy": float(accuracy_score(y, values)),
            "macro_f1": float(f1_score(y, values, average="macro")),
        }

    mistakes = np.flatnonzero(y != predicted)
    first_error = None
    if len(mistakes):
        i = mistakes[0]
        first_error = {
            "dataset_index": int(ids[i]),
            "actual": int(y[i]),
            "predicted": int(predicted[i]),
            "pixels": x[i].astype(int).reshape(8, 8).tolist(),
        }
    return {
        "test_count": len(y), "labels": list(range(10)),
        "baseline_digit": int(baseline[0]),
        "baseline": scores(baseline), "model": scores(predicted),
        "confusion_matrix": confusion_matrix(y, predicted, labels=range(10)).tolist(),
        "first_error": first_error,
    }


def read_pixels(path):
    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict) or "pixels" not in request:
        raise ValueError("필수 필드 pixels가 없습니다.")
    pixels = request["pixels"]
    if not isinstance(pixels, list) or len(pixels) != 8:
        raise ValueError("pixels에는 행 8개가 필요합니다.")
    for row in pixels:
        if not isinstance(row, list) or len(row) != 8:
            raise ValueError("각 행에는 픽셀 8개가 필요합니다.")
        if any(type(pixel) is not int or not 0 <= pixel <= 16 for pixel in row):
            raise ValueError("픽셀은 0~16 범위의 정수여야 합니다.")
    return np.asarray(pixels).reshape(1, 64) / 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "evaluate", "predict"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.command == "predict" and args.input is None:
        parser.error("predict requires --input")
    try:
        if args.command == "train":
            result = train(args.model)
        else:
            pixels = read_pixels(args.input) if args.command == "predict" else None
            bundle = joblib.load(args.model)
            if args.command == "evaluate":
                result = evaluate(bundle)
            else:
                result = {
                    "digit": int(bundle["model"].predict(pixels)[0]),
                    "model_version": bundle["version"],
                }
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
