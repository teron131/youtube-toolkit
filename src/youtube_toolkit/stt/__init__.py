"""Speech-to-text for videos without captions: cloud providers and local Hugging Face ASR models."""

from .cloud import (
    PROVIDERS,
    check_cloud_stt,
    cloud_provider,
    cloud_stt_configured,
    cloud_stt_status,
    transcribe_in_cloud,
)
from .local import (
    check_local_stt,
    local_stt_downloaded,
    local_stt_installed,
    transcribe_locally,
)

__all__ = [
    "PROVIDERS",
    "check_cloud_stt",
    "check_local_stt",
    "cloud_provider",
    "cloud_stt_configured",
    "cloud_stt_status",
    "local_stt_downloaded",
    "local_stt_installed",
    "transcribe_in_cloud",
    "transcribe_locally",
]
