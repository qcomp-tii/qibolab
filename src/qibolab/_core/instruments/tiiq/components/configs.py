from typing import Literal, Union

from qibolab._core.components import AcquisitionConfig, DcConfig

__all__ = [
    "TIIqOutputConfig",
    "TIIqAcquisitionConfig",
    "TIIqConfigs",
]

class TIIqOutputConfig(DcConfig):
    """DC channel config using TIIq."""

    kind: Literal["tiiq-output"] = "tiiq-output"

    offset: float = 0.0
    """DC offset to be applied in V.

    Possible values are -0.5V to 0.5V.
    """


class TIIqAcquisitionConfig(AcquisitionConfig):
    """Acquisition config for TIIq."""

    kind: Literal["tiiq-acquisition"] = "tiiq-acquisition"

    gain: int = 0
    """Input gain in dB.

    Possible values are -12dB to 20dB in steps of 1dB.
    """
    offset: float = 0.0
    """Constant voltage to be applied on the input."""


TIIqConfigs = Union[TIIqAcquisitionConfig]
