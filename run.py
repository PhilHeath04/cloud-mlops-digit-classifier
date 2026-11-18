"""Run the pipeline and retain inputs, outputs, and lifecycle evidence."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from service import MODEL_VERSION


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    runner = os.environ.get("GITHUB_RUN_ID")
    record = {
        "started_at_utc": now(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_version": MODEL_VERSION,
        "platform": platform.system() + " " + platform.machine(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in
                     ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")},
        "service": "GitHub Actions hosted runner" if runner else "직접 실행 환경",
        "region": "GitHub 호스팅 러너에서 실행 리전을 지정하지 않으며 실행 위치는 확인되지 않음" if runner else None,
        "runner_image": os.environ.get("ImageOS"),
        "runner_image_version": os.environ.get("ImageVersion"),
        "run_url": f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{runner}" if runner else None,
        "configuration": {"test_fraction": 0.25, "stratified": True, "seed": 42,
                          "C": 1.0, "max_iter": 2000, "pixel_divisor": 16},
        "steps": [],
    }

    def step(name, args, expected=0, input_path=None):
        command = ["service.py", *map(str, args)]
        start = now()
        process = subprocess.run(
            [sys.executable, *command], capture_output=True, text=True
        )
        entry = {"name": name, "started_at_utc": start, "completed_at_utc": now(),
                 "command": ["python", *command], "exit_code": process.returncode,
                 "expected_exit_code": expected,
                 "status": "success" if process.returncode == 0 else "failure",
                 "input_snapshot": input_path.name if input_path else None,
                 "input_sha256": digest(input_path) if input_path else None}
        record["steps"].append(entry)
        if process.stderr:
            filename = f"{name}.stderr.txt"
            (output / filename).write_text(process.stderr, encoding="utf-8")
            entry["stderr_file"] = filename
        if process.returncode != expected:
            raise RuntimeError(f"{name}: exit {process.returncode}, expected {expected}")
        if expected:
            if process.stdout or "필수 필드 pixels가 없습니다." not in process.stderr:
                raise RuntimeError("Expected validation error with no prediction")
            return None
        data = json.loads(process.stdout)
        filename = f"{name}.json"
        save(output / filename, data)
        entry["output_file"] = filename
        return data

    record["status"] = "failure"
    temp_path = None
    try:
        with TemporaryDirectory(prefix="digits-model-") as folder:
            temp_path = Path(folder)
            model = temp_path / "digits.joblib"
            request = temp_path / "request.json"
            step("train", ["train", "--model", model])
            record["model_sha256"] = digest(model)
            scores = step("evaluation", ["evaluate", "--model", model])
            if scores["model"]["accuracy"] < 0.90 or scores["model"]["accuracy"] <= scores["baseline"]["accuracy"]:
                raise RuntimeError("Model did not meet accuracy criteria")
            bad = output / "input_before.json"
            save(bad, {})
            shutil.copyfile(bad, request)
            command = ["predict", "--model", model, "--input", request]
            step("failed_request", command, expected=2, input_path=bad)
            good = output / "input_after.json"
            shutil.copyfile("examples/digit.json", good)
            shutil.copyfile(good, request)
            recovered = step("recovered_request", command, input_path=good)
            normal = step("normal_request", ["predict", "--model", model, "--input", good], input_path=good)
            if recovered["digit"] != 0 or normal["digit"] != 0:
                raise RuntimeError("The example digit must be classified as 0")
        record["status"] = "success"
    finally:
        record["cleanup"] = {"temporary_directory": str(temp_path) if temp_path else None,
                             "still_exists": temp_path.exists() if temp_path else None}
        record["completed_at_utc"] = now()
        save(output / "run.json", record)


if __name__ == "__main__":
    main()
