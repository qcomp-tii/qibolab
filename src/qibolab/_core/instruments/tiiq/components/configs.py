from dataclasses import dataclass, field, asdict
from pydantic import Field
from typing import Generic, Literal, TypeVar, Union, Optional, Annotated

from qibolab._core.components import AcquisitionConfig, IqConfig, DcConfig
from qibolab._core.serialize import NdArray

__all__ = [
    "TIIqDriveConfig",
    "TIIqFluxConfig",
    "TIIqProbeConfig",
    "TIIqAcquisitionConfig",
    "TIIqConfigs",
]


class TIIqDriveConfig(IqConfig):
    """Drive channel config for TIIq."""

    kind: Literal["tiiq-drive"] = "tiiq-drive"

    frequency: float
    digital_mixer_frequency: float


class TIIqFluxConfig(DcConfig):
    """Flux channel config for TIIq."""

    kind: Literal["tiiq-flux"] = "tiiq-flux"

    offset: float = 0.0
    frequency: Optional[float] = 0.0


class TIIqProbeConfig(IqConfig):
    """Probe channel config for TIIq."""

    kind: Literal["tiiq-probe"] = "tiiq-probe"

    frequency: float
    digital_mixer_frequency: float


class TIIqAcquisitionConfig(AcquisitionConfig):
    """Acquisition channel config for TIIq."""

    kind: Literal["tiiq-acquisition"] = "tiiq-acquisition"
    frequency: Optional[float|None] = None

    delay: Optional[float] = 0.0
    """Delay between readout pulse start and acquisition start."""
    smearing: Optional[float|None] = None

    threshold: Optional[float|None] = None
    """Signal threshold for discriminating ground and excited states."""
    iq_angle: Optional[float|None] = None
    """Signal angle in the IQ-plane for disciminating ground and excited
    states."""
    kernel: Annotated[Optional[NdArray], Field(repr=False)] = None
    """Integration weights to be used when post-processing the acquired
    signal."""


TIIqConfigs = Union[TIIqDriveConfig, TIIqFluxConfig, TIIqProbeConfig, TIIqAcquisitionConfig]
