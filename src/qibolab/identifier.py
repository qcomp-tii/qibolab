from enum import Enum
from typing import Annotated, Union

from pydantic import BeforeValidator, Field, PlainSerializer

QubitId = Annotated[Union[int, str], Field(union_mode="left_to_right")]
"""Type for qubit names."""

QubitPairId = Annotated[
    tuple[QubitId, QubitId],
    BeforeValidator(lambda p: tuple(p.split("-")) if isinstance(p, str) else p),
    PlainSerializer(lambda p: f"{p[0]}-{p[1]}"),
]
"""Type for holding ``QubitPair``s in the ``platform.pairs`` dictionary."""


# TODO: replace with StrEnum, once py3.10 will be abandoned
# at which point, it will also be possible to replace values with auto()
class ChannelType(str, Enum):
    """Names of channels that belong to a qubit.

    Not all channels are required to operate a qubit.
    """

    PROBE = "probe"
    ACQUISITION = "acquisition"
    DRIVE = "drive"
    FLUX = "flux"

    def __str__(self) -> str:
        return str(self.value)


ChannelId = str
"""Unique identifier for a channel."""
