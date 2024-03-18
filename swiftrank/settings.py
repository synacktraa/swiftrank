import os
import zipfile
import requests
from pathlib import Path

from tqdm import tqdm

DEFAULT_CACHE_DIR = Path(os.getenv(
    "SWIFTRANK_CACHE", 
    default=Path("~").expanduser() / ".cache" / "swiftrank"
))
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MAP = {
    "ms-marco-TinyBERT-L-2-v2": "Point it to actual onnx format file",
    "ms-marco-MiniLM-L-12-v2": "Point it to actual onnx format file",
    "ms-marco-MultiBERT-L-12": "Point it to actual onnx format file",
    "rank-T5-flan": "Point it to actual onnx format file"
}

DEFAULT_MODEL = os.getenv("SWIFTRANK_MODEL", "ms-marco-TinyBERT-L-2-v2")
"""Default Model to use"""

def get_model_path(model_id: str) -> Path:
    model_dir = DEFAULT_CACHE_DIR / model_id
    if model_dir.exists():
        return model_dir 

    local_zip_file = str(DEFAULT_CACHE_DIR / f"{model_id}.zip")
    model_url = f"https://some-url-to-model/{model_id}.zip"

    with requests.get(model_url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        progress_bar = tqdm(
            desc=model_id, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024,
        )
        with open(local_zip_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                progress_bar.update(f.write(chunk))
        
        progress_bar.desc = local_zip_file
        progress_bar.close()

    with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
        zip_ref.extractall(DEFAULT_CACHE_DIR)

    os.remove(local_zip_file)
    return model_dir
