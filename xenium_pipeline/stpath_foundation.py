"""Pathology encoders and ImageNet ResNet50 baseline.

Pathology recipes are shared unchanged with external prediction pilots.
ResNet50 is a separate comparator and is excluded from Combined.
"""
from pathlib import Path
from datetime import datetime
import os
import torch
import timm
import torchvision.transforms as transforms
from huggingface_hub import HfApi
from PIL import Image
from timm.layers import SwiGLUPacked
from torch.utils.data import Dataset

RECIPE_VERSION = "stpath_brca_training_compatible_v1"
MODEL_NAMES = ["UNI2h", "Virchow", "Virchow2", "ProvGigapath", "Conch", "ResNet50"]
EXPECTED_DIMS = {
    "UNI2h": 1536,
    "Virchow": 2560,
    "Virchow2": 2560,
    "ProvGigapath": 1536,
    "Conch": 512,
    "ResNet50": 2048,
}
MODEL_SOURCES = {
    "UNI2h": "MahmoodLab/UNI2-h",
    "Virchow": "paige-ai/Virchow",
    "Virchow2": "paige-ai/Virchow2",
    "ProvGigapath": "prov-gigapath/prov-gigapath",
    "Conch": "MahmoodLab/conch",
    "ResNet50": "torchvision.models.resnet50:IMAGENET1K_V1",
}

HF_TOKEN = os.environ.get("HF_TOKEN")


def log(message):
    print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}] {message}", flush=True)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRANSFORM_224 = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
TRANSFORM_PROV = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
TRANSFORM_DESCRIPTIONS = {
    "UNI2h": "Resize(224), CenterCrop(224), ImageNet normalization",
    "Virchow": "Resize(224), CenterCrop(224), ImageNet normalization",
    "Virchow2": "Resize(224), CenterCrop(224), ImageNet normalization",
    "ProvGigapath": "Resize(256, bicubic), CenterCrop(224), ImageNet normalization",
    "Conch": "Transform returned by create_model_from_pretrained",
    "ResNet50": "IMAGENET1K_V1.transforms(): Resize(256, bilinear), CenterCrop(224), ImageNet normalization",
}
POOLING_DESCRIPTIONS = {
    "UNI2h": "direct model output",
    "Virchow": "CLS concatenated with mean(tokens[5:]); STpath-compatible",
    "Virchow2": "CLS concatenated with mean(tokens[5:])",
    "ProvGigapath": "direct tile-encoder output",
    "Conch": "encode_image(proj_contrast=False, normalize=False)",
    "ResNet50": "global average pooling, flatten; fc replaced with Identity (2048 dimensions)",
}


class ManifestTileDataset(Dataset):
    def __init__(self, table, transform):
        self.table = table.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        path = Path(row.materialized_tile_path)
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Could not load tile {row.tile_id} from {path}: {exc}") from exc
        return tensor, str(row.tile_id)


def require_hf_token():
    global HF_TOKEN
    from .hf_auth import ensure_hf_token
    ensure_hf_token()
    HF_TOKEN = os.environ['HF_TOKEN']


def build_model(model_name, device):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")
    if model_name != "ResNet50":
        require_hf_token()
    log(f"Loading {model_name} from {MODEL_SOURCES[model_name]}")

    if model_name == "ResNet50":
        from torchvision.models import ResNet50_Weights, resnet50
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        transform = weights.transforms()
    elif model_name == "UNI2h":
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
        )
        transform = TRANSFORM_224
    elif model_name == "Virchow":
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        transform = TRANSFORM_224
    elif model_name == "Virchow2":
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        transform = TRANSFORM_224
    elif model_name == "ProvGigapath":
        model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True
        )
        transform = TRANSFORM_PROV
    else:
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "CONCH is not installed in this kernel. Install MahmoodLab/CONCH first."
            ) from exc
        model, transform = create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/conch",
            hf_auth_token=HF_TOKEN,
        )

    model = model.to(device).eval()
    log(f"Loaded {model_name}")
    return model, transform


def encode_batch(model_name, model, images):
    if model_name == "Conch":
        return model.encode_image(images, proj_contrast=False, normalize=False)
    embeddings = model(images)
    if model_name in {"Virchow", "Virchow2"}:
        embeddings = torch.cat(
            [embeddings[:, 0], embeddings[:, 5:].mean(dim=1)], dim=-1
        )
    return embeddings


def resolve_hf_revision(model_name):
    if model_name == "ResNet50":
        return None  # Torchvision weights are pinned in MODEL_SOURCES, not hosted on HF.
    try:
        return HfApi(token=HF_TOKEN).model_info(MODEL_SOURCES[model_name]).sha
    except Exception as exc:
        log(f"Could not resolve Hugging Face revision for {model_name}: {exc}")
        return None
