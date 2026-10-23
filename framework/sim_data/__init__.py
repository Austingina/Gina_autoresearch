"""Native, G-independent cross-simulator data collection contracts."""

from .contracts import EpisodeRequest, NativeCollectorAdapter, NativeEpisodeResult, OracleEvidence
from .gates import calvin_oracle_gate, libero_oracle_gate, robotwin_oracle_gate
from .runner import NativeCollectionRunner

__all__ = [
    "EpisodeRequest",
    "NativeCollectorAdapter",
    "NativeEpisodeResult",
    "OracleEvidence",
    "NativeCollectionRunner",
    "calvin_oracle_gate",
    "libero_oracle_gate",
    "robotwin_oracle_gate",
]
