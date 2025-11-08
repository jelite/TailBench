import json
import logging
import subprocess
import sys
from pathlib import Path

JSON_PATH = Path("models_base.json")
LOG_PATH = Path("model_launch_runs.log")


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(sys.stdout),
        ],
    )

def load_model_names(json_path: Path) -> list[str]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    models = data.get("models", [])
    return [model["name"] for model in models if "name" in model]

def run_model(model_name: str, batch_size: int) -> None:
    logging.info("Launching model: %s (batch=%d)", model_name, batch_size)
    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx",
        "--capture-range=cudaProfilerApi",
        "--sample=none",
        "-o", f"./profile/{model_name.replace('/', '_')}_b{batch_size}",
        sys.executable,
        "model_launch.py",
        "--model_name", model_name,
        "--batch", str(batch_size),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        logging.info("stdout for %s:\n%s", model_name, result.stdout.strip())
    if result.stderr:
        log_level = logging.ERROR if result.returncode != 0 else logging.INFO
        logging.log(log_level, "stderr for %s:\n%s", model_name, result.stderr.strip())

    if result.returncode != 0:
        logging.error("Execution failed for model %s (exit code %d)", model_name, result.returncode)


def main() -> None:
    batches = [1,2,4,8,16,32]
    
    if not JSON_PATH.exists():
        logging.error("JSON file not found: %s", JSON_PATH)
        sys.exit(1)

    model_names = load_model_names(JSON_PATH)
    if not model_names:
        logging.warning("No models found in %s", JSON_PATH)
        return

    for model_name in model_names:
        for batch in batches:
            run_model(model_name, batch)


if __name__ == "__main__":
    setup_logging()
    logging.info("Starting sequential model launch.")
    main()
    logging.info("Completed sequential model launch.")
