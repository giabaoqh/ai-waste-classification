"""
OOD and Non-Waste Evaluation Script for AI Waste Classification Project.
Tests the current 6-class model behavior on:
1. Actual waste items
2. People / Faces
3. Non-waste everyday objects (furniture, electronics, pets, vehicles)
4. Natural scenery
5. People holding waste
"""

import sys
import math
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torchvision import transforms, models

# Configure UTF-8 for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "mobilenetv3_best.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def load_classifier():
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    classes = checkpoint.get("target_classes", ["plastic", "paper", "metal", "glass", "organic", "other"])
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(classes))
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.to(DEVICE)
    model.eval()
    return model, classes

def evaluate_image(model, classes, pil_img):
    t = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(t)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        max_idx = int(torch.argmax(logits, dim=1).item())
        max_prob = float(probs[max_idx])
        pred_class = classes[max_idx]
        
        # Calculate Shannon Entropy: H = - sum(p * log(p))
        # Uniform across 6 classes: H_max = log(6) = 1.7918
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        
        # Energy score (Liu et al. 2020: Energy-based OOD): -T * logsumexp(logits / T)
        energy = -torch.logsumexp(logits, dim=1).item()

    return {
        "pred_class": pred_class,
        "confidence": max_prob,
        "entropy": entropy,
        "energy": energy,
        "all_probs": {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))}
    }

def main():
    print(f"Loading model on {DEVICE}...")
    model, classes = load_classifier()
    print(f"Model classes: {classes}")
    print("=" * 80)
    print(f"{'Category':<20} | {'Description':<25} | {'Predicted':<10} | {'Conf':<7} | {'Entropy':<7} | {'Energy':<7}")
    print("-" * 80)

    # 1. Real Waste Samples (from data/raw/realwaste if present)
    real_waste_dir = PROJECT_ROOT / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste"
    test_cases = []

    if real_waste_dir.exists():
        for cat in ["Plastic", "Paper", "Metal", "Glass", "Food Organics"]:
            cat_dir = real_waste_dir / cat
            if cat_dir.exists():
                imgs = list(cat_dir.glob("*.jpg"))
                if imgs:
                    test_cases.append(("Actual Waste", f"{cat} Sample", str(imgs[0])))

    # 2. Synthetic & Downloaded Non-Waste / OOD test images
    # Synthesize clean representative patterns for controlled baseline
    img_face = Image.new("RGB", (300, 300), color=(255, 224, 189)) # Skin tone
    d = ImageDraw.Draw(img_face)
    d.ellipse((50, 50, 250, 250), fill=(240, 200, 160)) # Head
    d.ellipse((90, 110, 120, 130), fill=(40, 40, 40)) # Eye
    d.ellipse((180, 110, 210, 130), fill=(40, 40, 40)) # Eye
    d.arc((110, 170, 190, 210), start=0, end=180, fill=(200, 50, 50), width=4) # Smile
    
    img_landscape = Image.new("RGB", (300, 300), color=(135, 206, 235)) # Sky
    d2 = ImageDraw.Draw(img_landscape)
    d2.polygon([(0, 300), (150, 120), (300, 300)], fill=(34, 139, 34)) # Mountain
    d2.ellipse((200, 30, 270, 100), fill=(255, 223, 0)) # Sun

    img_furniture = Image.new("RGB", (300, 300), color=(240, 240, 240))
    d3 = ImageDraw.Draw(img_furniture)
    d3.rectangle((50, 120, 250, 220), fill=(139, 69, 19)) # Wood table
    d3.rectangle((70, 220, 90, 280), fill=(100, 50, 10))
    d3.rectangle((210, 220, 230, 280), fill=(100, 50, 10))

    # Evaluate synthetic controls
    for name, desc, img in [
        ("People (Sim)", "Human Face Drawing", img_face),
        ("Scenery (Sim)", "Blue Sky & Mountain", img_landscape),
        ("Furniture (Sim)", "Wooden Table", img_furniture),
    ]:
        res = evaluate_image(model, classes, img)
        print(f"{name:<20} | {desc:<25} | {res['pred_class']:<10} | {res['confidence']*100:>5.1f}% | {res['entropy']:>6.3f} | {res['energy']:>6.2f}")
        print(f"   -> Probs: {res['all_probs']}")

    # 3. Public Web Test Images for People, Pets, Furniture, and People Holding Waste
    public_urls = [
        ("People Only", "Human Portrait Face", "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=300"),
        ("People Only", "Person in office", "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=300"),
        ("Pet / Animal", "Golden Retriever Dog", "https://images.unsplash.com/photo-1552053831-71594a27632d?w=300"),
        ("Furniture", "Living Room Sofa", "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=300"),
        ("Scenery", "Mountain Scenery", "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=300"),
        ("Person + Waste", "Hand holding coffee cup", "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=300"),
        ("Person + Waste", "Person throwing trash", "https://images.unsplash.com/photo-1618477461853-cf6ed80faba5?w=300"),
        ("Actual Waste", "Crumpled plastic bottle", "https://images.unsplash.com/photo-1563245372-f21724e3856d?w=300"),
        ("Actual Waste", "Crushed soda can", "https://images.unsplash.com/photo-1528190336454-13cd56b45b5a?w=300")
    ]

    print("\nEvaluating Public Photographic Test Cases...")
    print("-" * 80)
    for category, desc, url in public_urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                from io import BytesIO
                img = Image.open(BytesIO(resp.read())).convert("RGB")
                res = evaluate_image(model, classes, img)
                print(f"{category:<20} | {desc:<25} | {res['pred_class']:<10} | {res['confidence']*100:>5.1f}% | {res['entropy']:>6.3f} | {res['energy']:>6.2f}")
                print(f"   -> Top Probs: {res['all_probs']}")
        except Exception as e:
            print(f"{category:<20} | {desc:<25} | Download error: {e}")

    # Evaluate Local RealWaste Images if available
    if test_cases:
        print("\nEvaluating Local RealWaste Ground Truth Images...")
        print("-" * 80)
        for category, desc, path in test_cases:
            try:
                img = Image.open(path).convert("RGB")
                res = evaluate_image(model, classes, img)
                print(f"{category:<20} | {desc:<25} | {res['pred_class']:<10} | {res['confidence']*100:>5.1f}% | {res['entropy']:>6.3f} | {res['energy']:>6.2f}")
            except Exception as e:
                print(f"{category:<20} | {desc:<25} | Error: {e}")

if __name__ == "__main__":
    main()

