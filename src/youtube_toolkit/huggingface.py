"""Model plumbing shared by local speech-to-text and local embeddings: picking the device, checking the Hugging Face download cache, and quieting library output.

torch and the Hugging Face libraries are imported inside each function, so commands that never run a local model do not load them; callers check that their extra is installed first.
"""

from __future__ import annotations

from pathlib import Path


def model_downloaded(model: str) -> bool:
    """A local directory counts as downloaded; a model ID counts once its config.json is in the Hugging Face cache."""
    if Path(model).is_dir():
        return True
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model, "config.json"), str)


def torch_device() -> str:
    """Prefer an Nvidia GPU, then the Apple GPU, then the CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def quiet_transformers(*, downloaded: bool) -> None:
    """The CLI prints JSON on stdout and errors on stderr; library chatter belongs in neither, except progress for a download that can take minutes."""
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    if downloaded:
        transformers_logging.disable_progress_bar()
