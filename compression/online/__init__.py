"""Online / static LoRA-adaptive compression layer.

Built on top of the team's BaseCompressor entropy kernel; one modality-agnostic
scheduler drives text / audio / image through a single ``OnlineBackend`` seam.
"""
from compression.online.config import OnlineLearningConfig
from compression.online.static_compressor import StaticCompressor
from compression.online.online_compressor import OnlineCompressor
from compression.online.trainer import OnlineTrainer, build_optimizer

__all__ = [
    "OnlineLearningConfig",
    "StaticCompressor",
    "OnlineCompressor",
    "OnlineTrainer",
    "build_optimizer",
]
