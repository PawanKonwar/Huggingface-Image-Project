import re
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

import sys

# Allow running this file directly:
#   python src/utils/download_images_loremflickr.py
# by adding the project root to `sys.path`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import DATA_DIR  # noqa: E402

classes = {
    "my_cat": "cat",
    "my_dog": "dog",
    "my_car": "car",
    "my_house": "house",
    "my_phone": "smartphone"
}
#
# Balancing config
#
# Only top-up classes that have fewer than MIN_IMAGES.
# Top-up targets ~TARGET_IMAGES to keep some headroom on small datasets.
MIN_IMAGES = 50
TARGET_IMAGES = 60

#
# Network / retry config
#
MAX_RETRIES_PER_IMAGE = 5
RETRY_PAUSE_SECONDS = 2.0
PAUSE_BETWEEN_SUCCESSFUL_DOWNLOADS = 0.5  # tiny delay to reduce rate-limit risk

# Supported extensions for counting/verifying images.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def _count_images(class_dir: Path) -> int:
    if not class_dir.exists():
        return 0
    return sum(1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _next_index_for_class(class_name: str, class_dir: Path) -> int:
    """
    Determine the next integer suffix for filenames like: <class_name>_123.jpg
    so we don't overwrite existing images.
    """
    if not class_dir.exists():
        return 1
    pattern = re.compile(rf"^{re.escape(class_name)}_(\d+)\.(?:jpg|jpeg|png|bmp|gif)$", re.IGNORECASE)
    max_idx = 0
    for p in class_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def download_image(url: str) -> Image.Image | None:
    """
    Download and validate an image. Retries are handled by the caller.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 503:
            raise requests.HTTPError("503 Service Unavailable", response=r)
        r.raise_for_status()

        # Validate with PIL (verify() does not fully load the image).
        img_bytes = r.content
        img = Image.open(BytesIO(img_bytes))
        img.verify()

        # Re-open and convert for saving.
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception as e:
        print(f"Failed download/verify: {e}")
        return None


def top_up_class(class_name: str, query: str) -> None:
    class_dir = DATA_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    current = _count_images(class_dir)
    if current >= MIN_IMAGES:
        print(f"\n{class_name}: already has {current} images (>= {MIN_IMAGES}), skipping.")
        return

    # Aim for TARGET_IMAGES (user requested 50-60 range).
    target = TARGET_IMAGES
    need = max(0, target - current)
    next_idx = _next_index_for_class(class_name, class_dir)

    print(f"\n{class_name}: has {current} images, topping up to {target} (+{need}).")

    # Use a monotonically increasing lock parameter to reduce repeated fetches.
    # We keep the URL parameter tied to next_idx for deterministic naming.
    downloaded = 0
    lock = next_idx

    while downloaded < need:
        success = False
        for attempt in range(1, MAX_RETRIES_PER_IMAGE + 1):
            url = f"https://loremflickr.com/400/400/{query}?lock={lock}"
            img = download_image(url)
            if img is None:
                # Retry for network/server issues or invalid images.
                print(
                    f"Attempt {attempt}/{MAX_RETRIES_PER_IMAGE} failed for {class_name} (lock={lock})."
                )
                time.sleep(RETRY_PAUSE_SECONDS)
                continue

            img_path = class_dir / f"{class_name}_{next_idx}.jpg"
            img.save(img_path)
            success = True
            downloaded += 1
            next_idx += 1
            lock += 1
            time.sleep(PAUSE_BETWEEN_SUCCESSFUL_DOWNLOADS)
            print(f"Saved {img_path}")
            break

        if not success:
            # Skip this particular index after all retries fail.
            print(f"Skipped index after {MAX_RETRIES_PER_IMAGE} failed attempts: {class_name}_{next_idx}")
            next_idx += 1
            lock += 1


def main() -> None:
    # User-requested classes to balance (but we also safely check any class < MIN_IMAGES).
    prioritize = {"my_house", "my_car", "my_phone"}

    # First do user-specified classes.
    for class_name in sorted(prioritize):
        query = classes.get(class_name)
        if not query:
            print(f"Unknown class '{class_name}' in prioritization set, skipping.")
            continue
        top_up_class(class_name, query)

    # Then (optional) do other classes if they are also under MIN_IMAGES.
    for class_name, query in classes.items():
        if class_name in prioritize:
            continue
        top_up_class(class_name, query)


if __name__ == "__main__":
    main()
