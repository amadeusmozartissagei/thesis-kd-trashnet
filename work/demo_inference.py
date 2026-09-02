"""
Demo visual + latency benchmark untuk WasteNet-256K (R8, TA-KD, seed 42, K6).

- Prediksi gambar dari luar TrashNet (work/demo_images/<class>/*.jpg), disusun jadi
  satu figure dengan gambar + label prediksi + confidence, buat ditunjukin ke penguji.
- Latency: rata-rata waktu inference per gambar (CPU, laptop ini), dilaporkan dalam
  ms/gambar dan FPS.

Jalankan dari root repo:
    python work/demo_inference.py
"""

import argparse
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = (
    REPO_ROOT
    / "final_research_kd/runs/final/R8/seed_42/ta_t4_a0p5_lr1em4_av380"
    / "wastenet_256k_r8_ta_kd_final_seed_42_best.pth"
)
OUTPUT_DIR = REPO_ROOT / "outputs"

SOURCES = {
    "external": {
        "dir": REPO_ROOT / "work/demo_images",
        "title": "Demo prediksi (dataset luar TrashNet, sumber Wikimedia Commons)",
        "output": "demo_predictions_external.png",
    },
    "trashnet-test": {
        "dir": REPO_ROOT / "work/demo_images_trashnet_test",
        "title": "Demo prediksi (TrashNet held-out test set, seed 42 — belum pernah dilihat model)",
        "output": "demo_predictions_trashnet_test.png",
    },
    "combined": {
        "dir": REPO_ROOT / "work/demo_images_combined",
        "title": "Demo prediksi — foto luar TrashNet (diutamakan) + pelengkap TrashNet held-out test",
        "output": "demo_predictions_combined.png",
    },
    "final": {
        "dir": REPO_ROOT / "work/demo_images_final",
        "title": "Demo prediksi",
        "output": "demo_predictions_final.png",
    },
}
IMG_SIZE = 160
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
N_LATENCY_RUNS = 100
N_WARMUP = 10


# ---------------------------------------------------------------------------
# Model definition (persis dari final_research_kd/notebook6_r6_wastenet_256k_ce_baseline.ipynb)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WasteNetVariant:
    variant_id: str
    display_name: str
    channels: tuple
    repeats: tuple
    expand_ratio: float
    classifier_hidden: int
    expected_params: int
    checkpoint_prefix: str


WASTENET_256K = WasteNetVariant(
    variant_id="wastenet_256k",
    display_name="WasteNet-256K",
    channels=(16, 32, 72, 128, 224),
    repeats=(1, 1, 2, 1),
    expand_ratio=2.5,
    classifier_hidden=0,
    expected_params=256_002,
    checkpoint_prefix="wastenet_256k",
)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=kernel_size // 2, groups=groups, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=1.0):
        super().__init__()
        hidden_channels = int(round(in_channels * expand_ratio))
        layers = []
        if hidden_channels != in_channels:
            layers.append(ConvBNAct(in_channels, hidden_channels, kernel_size=1, stride=1))
        layers.extend([
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, stride=stride, groups=hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.block = nn.Sequential(*layers)
        self.activation = nn.SiLU(inplace=True)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        y = self.block(x)
        if self.use_residual:
            y = y + x
        return self.activation(y)


class WasteNet(nn.Module):
    def __init__(self, variant: WasteNetVariant, num_classes: int = 6, dropout: float = 0.2):
        super().__init__()
        layers = [ConvBNAct(3, variant.channels[0], kernel_size=3, stride=2)]
        in_channels = variant.channels[0]
        for out_channels, repeat in zip(variant.channels[1:], variant.repeats):
            for block_idx in range(repeat):
                stride = 2 if block_idx == 0 else 1
                layers.append(
                    DepthwiseSeparableBlock(in_channels, out_channels, stride=stride, expand_ratio=variant.expand_ratio)
                )
                in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_mapping"]["class_names"]
    model = WasteNet(WASTENET_256K, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def load_demo_images(demo_images_dir):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    items = []  # (true_class_folder, path, tensor, pil_image)
    for class_dir in sorted(demo_images_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            pil_img = Image.open(img_path).convert("RGB")
            tensor = transform(pil_img)
            items.append((class_dir.name, img_path, tensor, pil_img))
    return items


def run_latency_benchmark(model, sample_tensor):
    torch.set_num_threads(torch.get_num_threads())  # gunakan default thread CPU laptop
    x = sample_tensor.unsqueeze(0)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(x)
        times_ms = []
        for _ in range(N_LATENCY_RUNS):
            t0 = time.perf_counter()
            model(x)
            times_ms.append((time.perf_counter() - t0) * 1000)
    times_ms = np.array(times_ms)
    return {
        "mean_ms": times_ms.mean(),
        "std_ms": times_ms.std(),
        "p50_ms": np.percentile(times_ms, 50),
        "p95_ms": np.percentile(times_ms, 95),
        "fps": 1000.0 / times_ms.mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES.keys(), default="external")
    args = parser.parse_args()
    source = SOURCES[args.source]

    print("Loading model (WasteNet-256K, R8 TA-KD, seed 42)...")
    model, class_names = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Class names: {class_names}")
    print(f"Total params: {n_params:,}")

    print(f"Loading demo images ({args.source}) from {source['dir']}...")
    items = load_demo_images(source["dir"])
    print(f"Found {len(items)} demo images across {len(set(i[0] for i in items))} class folders")
    if not items:
        raise SystemExit(f"No images found in {source['dir']}")

    print(f"\nRunning latency benchmark ({N_WARMUP} warmup + {N_LATENCY_RUNS} timed runs, CPU)...")
    latency = run_latency_benchmark(model, items[0][2])
    print(f"  mean : {latency['mean_ms']:.2f} ms  (std {latency['std_ms']:.2f})")
    print(f"  p50  : {latency['p50_ms']:.2f} ms")
    print(f"  p95  : {latency['p95_ms']:.2f} ms")
    print(f"  FPS  : {latency['fps']:.1f}")

    print("\nRunning predictions on demo images (grouped per kelas)...")
    by_class = {}
    for true_class, img_path, tensor, pil_img in items:
        by_class.setdefault(true_class, []).append((img_path, tensor, pil_img))
    classes = sorted(by_class)
    cols = max(len(v) for v in by_class.values())
    rows = len(classes)

    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 4.0 * rows))
    axes = np.array(axes).reshape(rows, cols)

    total_correct = 0
    total_n = 0
    per_class_results = []

    with torch.no_grad():
        for r, true_class in enumerate(classes):
            samples = by_class[true_class]
            n_correct = 0
            for c in range(cols):
                ax = axes[r, c]
                ax.axis("off")
                if c >= len(samples):
                    continue
                img_path, tensor, pil_img = samples[c]
                logits = model(tensor.unsqueeze(0))
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(probs.argmax())
                pred_class = class_names[pred_idx]
                conf = float(probs[pred_idx]) * 100

                correct = pred_class == true_class
                n_correct += int(correct)
                color = "#2a9d34" if correct else "#e34948"

                if "_external" in img_path.stem:
                    tag = "[luar TrashNet]"
                elif "_trashnet" in img_path.stem:
                    tag = "[TrashNet test]"
                else:
                    tag = ""

                ax.imshow(pil_img)
                ax.set_title(f"pred: {pred_class} ({conf:.1f}%)", fontsize=10, color=color)
                if tag:
                    ax.text(
                        0.5, -0.06, tag, transform=ax.transAxes,
                        fontsize=9, color="#666666", ha="center", va="top",
                    )
                if c == 0:
                    ax.text(
                        -0.08, 0.5, true_class, transform=ax.transAxes,
                        fontsize=13, fontweight="bold", rotation=90,
                        va="center", ha="center",
                    )
                print(f"  {img_path.relative_to(REPO_ROOT)}: true={true_class} pred={pred_class} conf={conf:.1f}%")

            total_correct += n_correct
            total_n += len(samples)
            per_class_results.append((true_class, n_correct, len(samples)))

    print(f"\nPer-class accuracy (demo, source={args.source}):")
    for cls, k, n_cls in per_class_results:
        print(f"  {cls:10s}: {k}/{n_cls}")
    print(f"  {'TOTAL':10s}: {total_correct}/{total_n} ({100*total_correct/total_n:.1f}%)")

    fig_width_in = fig.get_size_inches()[0]
    wrap_chars = max(30, int(fig_width_in * 8))
    title_line1 = textwrap.fill(f"{source['title']} — WasteNet-256K R8 TA-KD, seed 42", wrap_chars)
    title_line2 = (
        f"Akurasi demo: {total_correct}/{total_n} ({100*total_correct/total_n:.0f}%) — "
        f"Latency CPU: {latency['mean_ms']:.2f} ms/gambar (~{latency['fps']:.1f} FPS)"
    )
    fig.suptitle(f"{title_line1}\n{title_line2}", fontsize=10, x=0.52)
    fig.tight_layout(rect=[0.02, 0, 1, 0.93])

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / source["output"]
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
