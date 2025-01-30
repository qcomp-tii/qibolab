import numpy as np
import threading
from pydantic import ConfigDict, Field

from qibo.config import log
# TODO: use qibolab logger or create a new one
from qibolab._core.components import Channel, Config
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.instruments.abstract import Controller, InstrumentSettings
from qibolab._core.instruments.tiiq.common import modes, DEMO, DEBUG, SIMULATION
from qibolab._core.instruments.tiiq.devices.module import TIIqModule 
from qibolab._core.pulses import Acquisition, Align, Delay, Pulse, Readout
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers, Parameter, Sweeper
from qibolab._core.unrolling import unroll_sequences, Bounds

__all__ = [
    "TIIqController",
    ]

SAMPLING_RATE = 1
BOUNDS = Bounds(waveforms=1, readout=1, instructions=1)

modes.enable(DEMO) # generates random values for the results
# TODO: remove once the code to process the results is implemented
modes.enable(DEBUG) # enables debug messages
# modes.enable(SIMULATION) # generates module program and configuration without executing


# pydantic
class TIIqController(Controller):
    """:class:`qibolab.instruments.abstract.Controller` object for controlling
    a TIIq intrument.

    TIIqController is comprised of multiple :class:`TIIqModule` objects, each
    representing a module of the TIIq instrument. The controller is responsible
    for preparing each module for execution and consolidating the results.
    """
    modules: dict[str, TIIqModule]
    bounds: str = "tiiq/bounds"

    def model_post_init(self, __context):
        self.channels: dict[ChannelId, Channel]  = {}
        # confirm modules are present
        if len(self.modules) == 0:
            raise ValueError('No modules found in TIIqController.')
        # register channels
        for module_id, module in self.modules.items():
            for channel_id, channel in module.channels.items():
                if not channel_id in self.channels:
                    self.channels[channel_id] = channel
                else:
                    log.warning(
                        f"""Channel {channel_id} pulses registered with module {self.channels[channel_id].device}
                        are being replicated at module {module}."""
                    )
                    # In principle, each channel_id should only be registered in one module, 
                    # but it is possible to register the same channel in other modules
                    # to generate replicas of the signals and debug them with an oscilloscope

        return super().model_post_init(__context)

    def connect(self):
        """Establish connection to the physical instruments."""
        log.info(f"Connecting to TIIqController modules.")
        module: TIIqModule
        for module in self.modules.values():
            module.connect()

    def disconnect(self):
        """Close connection to the physical instruments."""
        log.info(f"Disconnecting from TIIqController modules.")
        module: TIIqModule
        for module in self.modules.values():
            module.disconnect()

    @property
    def sampling_rate(self) -> int:
        """Sampling rate of control electronics in giga samples per second
        (GSps)."""
        return SAMPLING_RATE
        # raise NotImplementedError("TIIq sampling rates are different for different channels")

    def _generate_random_values(self, options: ExecutionParameters, shape: tuple[int, ...]) -> np.ndarray:
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
        """Perform an experiment and retrieve the results.

        An experiment can be the execution of a single pulse sequence, or the execution
        of multiple sequences while sweeping one or several parameters.
        The parameters being swept are passed as :class:`qibolab.Sweeper` objects.
        Sweepers are executed in real time if possible.

        Returns a mapping with the id of the probe pulses used to acquired data.
        """
        log.info(f"{self.signature} playing sequences.")
        # TODO if play was called before connecting, then issue wanring and try to connect or default to simulation.
        # unroll sequences
        if len(sequences) == 0:
            return {}
        elif len(sequences) == 1:
            sequence = sequences[0]
        else:
            sequence, readout_map = unroll_sequences(sequences, options.relaxation_time)

        if len(sequence) == 0:
            return {}

        # prepare modules for execution
        module_id: str
        module: TIIqModule
        for module_id, module in self.modules.items():
            module.prepare_execution(configs, sequence, options, sweepers)

        # trigger execution
        module_id: str
        module: TIIqModule
        module_results: dict[str, dict[int, Result]] = {}
        if modes.is_enabled(SIMULATION):
            for module_id, module in self.modules.items():
                module_results[module_id] = module.execute()
        else:
            threads: dict[str, threading.Thread] = {}
            # Create and start threads
            for module_id, module in self.modules.items():
                    def thread_target(mod=module):
                        log.debug(f"Starting thread for module {module_id}.")
                        module_results[module_id] = mod.execute()
                    threads[module_id] = threading.Thread(target=thread_target)
                    threads[module_id].start()

            # Join threads
            for module_id in threads:
                threads[module_id].join()

        log.debug(f"all modules completed execution")

        # return results
        results: dict[int, Result] = {}
        if modes.is_enabled(DEMO) or modes.is_enabled(SIMULATION):
            def _values(acq: Acquisition):
                samples = int(acq.duration * self.sampling_rate)
                return np.array(
                    self._generate_random_values(options, options.results_shape(sweepers, samples))
                )
            results = {
                acq.id: _values(acq) for seq in sequences for (channel_id, acq) in seq.acquisitions
            }
        else:
            # merge results
            for module_id in module_results:
                results.update(module_results[module_id])

        return results

    
