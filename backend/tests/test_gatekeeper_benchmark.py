"""
Evaluate Gatekeeper on Real Waste vs Non-Waste images.
"""

import sys, os, urllib.request
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from PIL import Image
from io import BytesIO
import torch
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cached_path = Path(os.environ['USERPROFILE']) / '.cache' / 'torch' / 'hub' / 'checkpoints' / 'mobilenet_v3_large-5c1a4163.pth'
state = torch.load(cached_path, map_location=DEVICE)
imagenet_model = models.mobilenet_v3_large(weights=None)
imagenet_model.load_state_dict(state)
imagenet_model.to(DEVICE).eval()

weights = models.MobileNet_V3_Large_Weights.DEFAULT
categories = weights.meta["categories"]

PERSON_CLASSES = {
    "suit", "suit of clothes", "trench coat", "academic gown", "jersey", "t-shirt", 
    "jean", "blue jean", "denim", "cardigan", "sweatshirt", "brassiere",
    "wig", "hair slide", "sunglasses", "sunglass", "mortarboard", "groom", "scuba diver"
}
PET_ANIMAL_CLASSES = {
    cat for cat in categories if any(animal in cat.lower() for animal in [
        "retriever", "terrier", "hound", "dog", "cat", "shepherd", "spaniel", "poodle",
        "pug", "bulldog", "chihuahua", "tabby", "siamese", "persian cat", "horse", "zebra"
    ])
}
FURNITURE_CLASSES = {
    "studio couch", "day bed", "sofa", "couch", "dining table", "board", "desk",
    "bookcase", "wardrobe", "closet", "armoire", "four-poster bed", "cradle", "crib"
}
SCENERY_CLASSES = {
    "valley", "alp", "lakeside", "seashore", "coast", "sandbar", "promontory",
    "volcano", "cliff", "drop", "drop-off", "geyser"
}

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def check_image(pil_img):
    t = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = imagenet_model(t)
        probs = torch.softmax(out, dim=1)[0]
        topk = torch.topk(probs, 5)
        top_preds = [(categories[idx], float(prob)*100) for idx, prob in zip(topk.indices, topk.values)]
        
    for class_name, prob in top_preds[:3]:
        class_clean = class_name.lower().strip()
        if class_clean in PERSON_CLASSES and prob >= 5.0:
            return False, "person_detected", f"Phát hiện người hoặc trang phục ({class_name}: {prob:.1f}%)", top_preds
        if any(a == class_clean or a in class_clean.split() for a in PET_ANIMAL_CLASSES) and prob >= 15.0:
            return False, "pet_detected", f"Phát hiện thú cưng hoặc động vật ({class_name}: {prob:.1f}%)", top_preds
        if (class_clean in FURNITURE_CLASSES or any(f in class_clean for f in ["couch", "sofa", "bed", "wardrobe"])) and prob >= 20.0:
            return False, "furniture_detected", f"Phát hiện nội thất / đồ dùng ({class_name}: {prob:.1f}%)", top_preds
        if (class_clean in SCENERY_CLASSES or any(s in class_clean for s in ["valley", "alp", "seashore"])) and prob >= 20.0:
            return False, "scenery_detected", f"Phát hiện phong cảnh thiên nhiên ({class_name}: {prob:.1f}%)", top_preds
            
    return True, "valid_waste", "Hợp lệ (Không phát hiện đối tượng ngoại lai)", top_preds

# Test on Local RealWaste images
real_waste_dir = PROJECT_ROOT / "data" / "raw" / "realwaste" / "extracted" / "realwaste-main" / "RealWaste"
print("=" * 80)
print("KIỂM THỬ TRÊN ẢNH RÁC THỰC TẾ (REALWASTE):")
print("-" * 80)
for cat in ["Plastic", "Paper", "Metal", "Glass", "Food Organics"]:
    cat_dir = real_waste_dir / cat
    if cat_dir.exists():
        for img_path in list(cat_dir.glob("*.jpg"))[:3]:
            img = Image.open(img_path).convert("RGB")
            is_valid, status, reason, preds = check_image(img)
            mark = "✅ PASS" if is_valid else "❌ FALSE POSITIVE"
            print(f"{mark} [{cat:<14}] {img_path.name:<15} -> {status}: {reason}")
            if not is_valid:
                print(f"   Top-3: {preds[:3]}")

# Test on Non-Waste Public URLs
public_urls = [
    ("Human Portrait Face", "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=300"),
    ("Person in office", "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=300"),
    ("Golden Retriever Dog", "https://images.unsplash.com/photo-1552053831-71594a27632d?w=300"),
    ("Living Room Sofa", "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=300"),
    ("Mountain Scenery", "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=300"),
]

print("\n" + "=" * 80)
print("KIỂM THỬ TRÊN ẢNH KHÔNG PHẢI RÁC (NON-WASTE / OOD):")
print("-" * 80)
for desc, url in public_urls:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            img = Image.open(BytesIO(resp.read())).convert("RGB")
            is_valid, status, reason, preds = check_image(img)
            mark = "🛡️ CHẶN THÀNH CÔNG" if not is_valid else "⚠️ LỌT QUA (FALSE NEGATIVE)"
            print(f"{mark} | {desc:<25} -> Status: {status} | {reason}")
    except Exception as e:
        print(f"Error: {e}")
