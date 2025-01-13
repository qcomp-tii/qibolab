from dataclasses import dataclass, field
from pydantic import Field
from typing import Generic, Literal, TypeVar, Union, Optional, Annotated

from qibolab._core.instruments.tiiq.common import PortId
from qibolab._core.instruments.tiiq.components.configs import (
    TIIqDriveConfig,
    TIIqFluxConfig,
    TIIqProbeConfig,
    TIIqAcquisitionConfig,
)
from qibolab._core.serialize import Model, NdArray

__all__ = [
    "DriveComponent",
    "FluxComponent",
    "ProbeComponent",
    "DriveComponent",
    "TIIqComponents",
]


@dataclass(frozen=True)
class Component:
    port_id: PortId

@dataclass(frozen=True)
class DriveComponent(Component):
    frequency: float
    digital_mixer_frequency: float

    @classmethod
    def from_config(cls, config: TIIqDriveConfig, port_id: PortId):
        return cls(**config.model_dump(exclude={"kind"}), port_id=port_id)


@dataclass(frozen=True)
class FluxComponent(Component):
    offset: float = 0.0
    frequency: Optional[float] = 0.0

    @classmethod
    def from_config(cls, config: TIIqFluxConfig, port_id: PortId):
        return cls(**config.model_dump(exclude={"kind"}), port_id=port_id)


@dataclass(frozen=True)
class ProbeComponent(Component):
    frequency: float
    digital_mixer_frequency: float

    @classmethod
    def from_config(cls, config: TIIqProbeConfig, port_id: PortId):
        return cls(**config.model_dump(exclude={"kind"}), port_id=port_id)

@dataclass(frozen=True)
class AcquisitionComponent(Component):
    frequency: Optional[float] = None
    delay: Optional[float] = 0.0
    smearing: Optional[float|None] = None
    threshold: Optional[float|None] = None
    iq_angle: Optional[float|None] = None
    kernel: Annotated[Optional[NdArray], Field(repr=False)] = None

    @classmethod
    def from_config(cls, config: TIIqAcquisitionConfig, port_id: PortId):
        return cls(**config.model_dump(exclude={"kind"}), port_id=port_id)
    

TIIqComponents = Union[DriveComponent, FluxComponent, ProbeComponent, AcquisitionComponent]
