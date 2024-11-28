import numpy as np
from pydantic import ConfigDict, Field
from typing import Optional, Union

from qibo.config import log
from qibolab._core.components import (
    AcquisitionChannel,
    Config,
    Channel,
    DcChannel,
    IqChannel,
    IqConfig,
    OscillatorConfig,
)
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.instruments.abstract import Controller, InstrumentSettings
from qibolab._core.pulses import Acquisition, Align, Delay, Pulse, Readout
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers, Parameter, Sweeper
from qibolab._core.unrolling import unroll_sequences, Bounds

SAMPLING_RATE = 1
BOUNDS = Bounds(waveforms=1, readout=1, instructions=1)


__all__ = ["TIIqController"]


class TIIqController(Controller):
    """:class:`qibolab.instruments.abstract.Controller` object for controlling
    a TIIq intrument.
    """
    bounds: str = "tiiq/bounds"
    # address: str
    # channels: dict[ChannelId, Channel] = Field(default_factory=dict)
    # settings: Optional[InstrumentSettings] = None

    # @property
    # def signature(self):
    #     return f"{type(self).__name__}@{self.address}"

    def connect(self):
        """Establish connection to the physical instrument."""
        log.info(f"Connecting to TIIqController instrument.")

    def disconnect(self):
        """Close connection to the physical instrument."""
        log.info(f"Disconnecting from TIIqController instrument.")

    def setup(self, *args, **kwargs):
        """Set instrument settings.

        Used primarily by non-controller instruments, to upload settings
        (like LO frequency and power) to the instrument after
        connecting.
        """

    @property
    def sampling_rate(self) -> int:
        """Sampling rate of control electronics in giga samples per second
        (GSps)."""
        return SAMPLING_RATE


    def _generate_values(self, options: ExecutionParameters, shape: tuple[int, ...]):
        if options.acquisition_type is AcquisitionType.DISCRIMINATION:
            if options.averaging_mode is AveragingMode.SINGLESHOT:
                return np.random.randint(2, size=shape)
            return np.random.rand(*shape)
        return np.random.rand(*shape) * 100



    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:
        """Play a pulse sequence and retrieve feedback.

        If :class:`qibolab.Sweeper` objects are passed as arguments, they are
        executed in real-time. If not possible, an error is raised.

        Returns a mapping with the id of the probe pulses used to acquired data.
        """
        def _values(acq: Acquisition):
            samples = int(acq.duration * self.sampling_rate)
            return np.array(
                self._generate_values(options, options.results_shape(sweepers, samples))
            )

        return {
            acq.id: _values(acq) for seq in sequences for (_, acq) in seq.acquisitions
        }