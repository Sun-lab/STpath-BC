"""Reuse Hugging Face's saved login without recording credentials in run outputs."""
import os
from getpass import getpass
from pathlib import Path


def ensure_hf_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        from huggingface_hub import get_token
        token = get_token()
    if not token:
        raise ValueError(
            "HF_TOKEN and saved Hugging Face login are missing. Run: "
            "python Workflow/08_predict_tcga.py cohort -- login"
        )
    # The existing encoder loader and Slurm --export=ALL consume this variable.
    os.environ["HF_TOKEN"] = token


def login_once():
    from huggingface_hub import constants, login
    token = getpass("Hugging Face token (hidden): ").strip()
    if not token:
        raise ValueError("Token is empty; no login saved")
    previous_umask = os.umask(0o077)
    try:
        login(token=token, add_to_git_credential=False)
        for name in ("HF_TOKEN_PATH", "HF_STORED_TOKENS_PATH"):
            location = getattr(constants, name, None)
            if location and Path(location).is_file():
                Path(location).chmod(0o600)
    finally:
        os.umask(previous_umask)
    print("Hugging Face login saved. Future BRCA batch runs load it automatically.", flush=True)
