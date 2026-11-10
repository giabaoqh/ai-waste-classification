"""Prepare a reproducible, author-annotated TACO detection pilot, not material labels."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import random
import urllib.request

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "70479ad9adc3da971a8ae75ac97eb1026910eec7"
ANNOTATIONS_URL = f"https://raw.githubusercontent.com/pedropro/TACO/{SOURCE_COMMIT}/data/annotations.json"
ANNOTATIONS_SHA256 = "812c2e3f0d336b437f276597715de2abb08bb8a641599c45f593d233ec66457b"
CLASSES = ["bottle_jar", "can", "cup", "carton", "cap_lid", "bag_film_wrapper", "paper", "cigarette", "other_litter"]
GROUPS = {
    "bottle_jar": ["Other plastic bottle", "Clear plastic bottle", "Glass bottle", "Glass jar"],
    "can": ["Food Can", "Drink can"],
    "cup": ["Paper cup", "Disposable plastic cup", "Foam cup", "Glass cup", "Other plastic cup"],
    "carton": ["Other carton", "Egg carton", "Drink carton", "Corrugated carton", "Meal carton", "Pizza box"],
    "cap_lid": ["Plastic bottle cap", "Metal bottle cap", "Plastic lid", "Metal lid"],
    "bag_film_wrapper": ["Paper bag", "Plastified paper bag", "Plastic film", "Garbage bag", "Other plastic wrapper", "Single-use carrier bag", "Polypropylene bag", "Crisp packet"],
    "paper": ["Magazine paper", "Tissues", "Wrapping paper", "Normal paper"],
    "cigarette": ["Cigarette"],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def category_mapping(categories):
    by_name = {name: group for group, names in GROUPS.items() for name in names}
    return {category["id"]: CLASSES.index(by_name.get(category["name"], "other_litter")) for category in categories}


def normalized_box(bbox, width, height):
    """COCO xywh -> clipped normalized YOLO xywh; reject malformed geometry."""
    if width <= 0 or height <= 0 or len(bbox) != 4:
        raise ValueError("Invalid image dimensions or bbox length")
    x, y, w, h = map(float, bbox)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError(f"Invalid bbox: {bbox}")
    x1, y1 = max(0., x), max(0., y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Bbox outside image: {bbox}")
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)


def assign_groups(images, seed):
    """Hold out entire source batches; these are a proxy, not verified scene IDs."""
    groups = sorted({Path(image["file_name"]).parts[0] for image in images})
    if len(groups) < 5:
        raise ValueError("At least five source groups required")
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * .15))
    n_test = max(1, round(len(groups) * .20))
    return {group: ("val" if i < n_val else "test" if i < n_val + n_test else "train") for i, group in enumerate(groups)}


def balanced_order(images, annotations_by_image, mapping, seed):
    remaining = list(images)
    random.Random(seed).shuffle(remaining)
    seen = Counter()
    while remaining:
        def score(image):
            labels = {mapping[a["category_id"]] for a in annotations_by_image[image["id"]]}
            return sum(1. / (1 + seen[label]) for label in labels)
        index = max(range(len(remaining)), key=lambda i: score(remaining[i]))
        image = remaining.pop(index)
        seen.update({mapping[a["category_id"]] for a in annotations_by_image[image["id"]]})
        yield image


def download_image(image, cache):
    destination = cache / f"{image['id']:05d}.jpg"
    url = image.get("flickr_640_url") or image.get("flickr_url")
    try:
        if not url:
            raise ValueError("Source image has no available download URL")
        if not destination.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "AI-Waste-Research/1.0"})
            with urllib.request.urlopen(request, timeout=25) as response:
                content = response.read(15 * 1024 * 1024 + 1)
            if len(content) > 15 * 1024 * 1024:
                raise ValueError("Image exceeds 15 MB download limit")
            partial = destination.with_suffix(".part")
            partial.write_bytes(content)
            partial.replace(destination)
        with Image.open(destination) as original:
            original.load()
            pil = ImageOps.exif_transpose(original)
            width, height = pil.size
            if abs((width / height) / (image["width"] / image["height"]) - 1) > .02:
                raise ValueError("Thumbnail aspect ratio differs from annotation dimensions")
            pixels = list(pil.convert("L").resize((9, 8)).get_flattened_data())
            dhash = sum((pixels[row * 9 + col] > pixels[row * 9 + col + 1]) << (row * 8 + col) for row in range(8) for col in range(8))
        return {"image": image, "path": destination, "url": url, "width": width, "height": height,
                "sha256": sha256(destination), "dhash": dhash}
    except Exception as exc:
        return {"image_id": image["id"], "error": f"{type(exc).__name__}: {exc}", "url": url}


def write_training_image(source, destination):
    """Canonicalize a derived JPEG; preserve raw bytes and avoid trainer repairs."""
    with Image.open(source) as original:
        ImageOps.exif_transpose(original).convert("RGB").save(destination, format="JPEG", quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/taco_pilot_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if args.count < 20:
        raise SystemExit("At least 20 images required for this pilot")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output is not empty; use a new version: {output}")
    raw = ROOT / "data/raw/taco"
    cache = raw / "thumbnails"
    cache.mkdir(parents=True, exist_ok=True)
    annotation_file = raw / "annotations.json"
    if not annotation_file.exists():
        urllib.request.urlretrieve(ANNOTATIONS_URL, annotation_file)
    if sha256(annotation_file) != ANNOTATIONS_SHA256:
        raise SystemExit("Annotation hash differs from pinned source; inspect before proceeding")
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    mapping = category_mapping(data["categories"])
    annotations_by_image = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
    group_splits = assign_groups(data["images"], args.seed)
    val_count = round(args.count * .15)
    quotas = {"train": args.count - 2 * val_count, "val": val_count, "test": val_count}
    selected, failures, skipped = [], [], []
    for split, quota in quotas.items():
        candidates = [image for image in data["images"] if group_splits[Path(image["file_name"]).parts[0]] == split]
        ordered = iter(balanced_order(candidates, annotations_by_image, mapping, args.seed))
        accepted = 0
        while accepted < quota:
            batch = [image for _, image in zip(range(min(8, quota - accepted)), ordered)]
            if not batch:
                raise RuntimeError(f"Insufficient available images for {split}")
            with ThreadPoolExecutor(max_workers=8) as pool:
                pending = {pool.submit(download_image, image, cache): image["id"] for image in batch}
                downloaded = [future.result() for future in as_completed(pending)]
            downloaded.sort(key=lambda item: item.get("image", {}).get("id", item.get("image_id")))
            for item in downloaded:
                if "error" in item:
                    failures.append(item)
                    continue
                duplicate = next((old for old in selected if old["sha256"] == item["sha256"] or (old["dhash"] ^ item["dhash"]).bit_count() <= 4), None)
                if duplicate is not None:
                    skipped.append({"image_id": item["image"]["id"], "similar_to": duplicate["image"]["id"], "reason": "sha256_or_dhash"})
                    continue
                item["split"] = split
                selected.append(item)
                accepted += 1
            print(f"{split}: {accepted}/{quota}; failed URLs: {len(failures)}; excluded duplicates: {len(skipped)}", flush=True)
    manifest = []
    support = {split: Counter() for split in quotas}
    for item in selected:
        image, split = item["image"], item["split"]
        name = f"taco_{image['id']:05d}"
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        derived_image = image_dir / (name + ".jpg")
        write_training_image(item["path"], derived_image)
        labels = []
        for annotation in annotations_by_image[image["id"]]:
            class_id = mapping[annotation["category_id"]]
            box = normalized_box(annotation["bbox"], image["width"], image["height"])
            labels.append(f"{class_id} " + " ".join(f"{number:.8f}" for number in box))
            support[split][CLASSES[class_id]] += 1
        (label_dir / (name + ".txt")).write_text("\n".join(labels) + "\n", encoding="utf-8")
        manifest.append({"image_id": image["id"], "source_file": image["file_name"], "source_batch": Path(image["file_name"]).parts[0],
                         "split": split, "url": item["url"], "source_license": image.get("license"),
                         "source_sha256": item["sha256"], "sha256": sha256(derived_image), "dhash": f"{item['dhash']:016x}",
                         "image_transform": "EXIF transpose, RGB, JPEG quality=95; original preserved",
                         "image_size": [item["width"], item["height"]], "annotation_size": [image["width"], image["height"]],
                         "annotation_ids": [a["id"] for a in annotations_by_image[image["id"]]], "label_file": f"labels/{split}/{name}.txt",
                         "image_file": f"images/{split}/{name}.jpg", "human_review": "pending"})
    config = f"path: {json.dumps(output.as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    config += "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    (output / "dataset.yaml").write_text(config, encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    taxonomy = [{"source_id": c["id"], "source_name": c["name"], "target_id": mapping[c["id"]], "target_name": CLASSES[mapping[c["id"]]]} for c in data["categories"]]
    (output / "taxonomy.json").write_text(json.dumps({"version": "taco-object-groups-v1-pilot", "classes": CLASSES, "mapping": taxonomy}, indent=2), encoding="utf-8")
    audit = {"status": "technical_checks_passed_human_review_pending", "source": ANNOTATIONS_URL, "source_commit": SOURCE_COMMIT,
             "annotations_sha256": ANNOTATIONS_SHA256, "seed": args.seed, "image_counts": dict(Counter(item["split"] for item in manifest)),
             "instance_support": {split: {name: support[split][name] for name in CLASSES} for split in quotas},
             "source_batch_splits": group_splits, "failed_downloads": failures, "excluded_similar_images": skipped,
             "limitations": ["100-image pilot is not a representative benchmark", "source batch is only a proxy for scene groups",
                             "dHash is a heuristic, not proof of no near-duplicate scenes", "all annotations require project-specific human review",
                             "other_litter includes rare and potentially hazardous items; not a disposal recommendation",
                             "thumbnail when available, otherwise original; training letterboxes images to configured size",
                             "per-image source license is recorded when provided; missing license needs review before redistribution",
                             "no medical or polymer capability established"]}
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": str(output / "dataset.yaml"), "audit": audit}, indent=2), flush=True)


if __name__ == "__main__":
    main()
