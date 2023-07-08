"""Platform for controlling quantum devices."""

import math
import re
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import yaml
from qibo.config import log, raise_error

from qibolab.execution_parameters import ExecutionParameters
from qibolab.instruments.abstract import Controller, Instrument, InstrumentId
from qibolab.native import NativeType
from qibolab.pulses import PulseSequence
from qibolab.qubits import Qubit, QubitId, QubitPair, QubitPairId
from qibolab.sweeper import Sweeper

InstrumentMapType = Dict[InstrumentId, Instrument]
QubitMapType = Dict[QubitId, Qubit]
QubitPairMapType = Dict[QubitPairId, QubitPair]


class ResonatorType(Enum):
    """Available resonator types."""

    dim2 = "2D"
    dim3 = "3D"


@dataclass
class PlatformSettings:
    """Default execution settings read from the runcard."""

    nshots: int = 1024
    """Default number of repetitions when executing a pulse sequence."""
    sampling_rate: int = int(1e9)
    """Number of waveform samples supported by the instruments per second."""
    relaxation_time: int = int(1e5)
    """Time in ns to wait for the qubit to relax to its ground state between shots."""
    time_of_flight: int = 280
    """Time in ns for the signal to reach the qubit from the instruments."""
    smearing: int = 0
    """Readout pulse window to be excluded during the signal integration."""


@dataclass
class Platform:
    """Platform for controlling quantum devices.

    Args:

        runcard (str): path to the yaml file containing the platform setup.
        instruments:
    """

    name: str
    """Name of the platform."""
    qubits: QubitMapType
    pairs: QubitPairMapType
    instruments: InstrumentMapType

    settings: PlatformSettings = field(default_factory=PlatformSettings)
    resonator_type: Optional[ResonatorType] = None

    nqubits: int = 0
    is_connected: bool = False
    two_qubit_native_types: NativeType = field(default_factory=lambda: NativeType(0))
    topology: nx.Graph = field(default_factory=nx.Graph)

    def __post_init__(self):
        log.info("Loading platform %s", self.name)
        self.nqubits = len(self.qubits)
        if isinstance(self.resonator_type, str):
            self.resonator_type = ResonatorType(self.resonator_type)
        elif self.resonator_type is None:
            self.resonator_type = ResonatorType("3D") if self.nqubits == 1 else ResonatorType("2D")

        for pair in self.pairs.values():
            self.two_qubit_native_types |= pair.native_gates.types
        if self.two_qubit_native_types is NativeType(0):
            # dummy value to avoid transpiler failure for single qubit devices
            self.two_qubit_native_types = NativeType.CZ

        self.topology.add_nodes_from(self.qubits.keys())
        self.topology.add_edges_from([(pair.qubit1.name, pair.qubit2.name) for pair in self.pairs.values()])

    def dump(self, path: Path):
        from qibolab.utils import dump_qubits

        settings = {
            "nqubits": self.nqubits,
            "qubits": list(self.qubits),
            "settings": asdict(self.settings),
        }
        settings.update(dump_qubits(self.qubits, self.pairs))
        path.write_text(yaml.dump(settings, sort_keys=False, indent=4, default_flow_style=None))

    def update(self, updates: dict):
        r"""Updates platform common runcard parameters after calibration actions.

        Args:

            updates (dict): Dictionary containing the parameters to update the runcard. A typical dictionary should be of the following form
                            {`parameter_to_update_in_runcard`:{`qubit0`:`par_value_qubit0`, ..., `qubit_i`:`par_value_qubit_i`, ...}}.
                            The parameters that can be updated by this method are:
                                - readout_frequency (GHz)
                                - readout_attenuation (dimensionless)
                                - bare_resonator_frequency (GHz)
                                - sweetspot(V)
                                - drive_frequency (GHz)
                                - readout_amplitude (dimensionless)
                                - drive_amplitude (dimensionless)
                                - drive_length
                                - t2 (ns)
                                - t2_spin_echo (ns)
                                - t1 (ns)
                                - thresold(V)
                                - iq_angle(deg)
                                - mean_gnd_states(V)
                                - mean_exc_states(V)
                                - beta(dimensionless)
        """

        for par, values in updates.items():
            for qubit, value in values.items():
                # resonator_spectroscopy / resonator_spectroscopy_flux / resonator_punchout_attenuation
                if par == "readout_frequency":
                    freq = int(value * 1e9)
                    mz = self.qubits[qubit].native_gates.MZ
                    mz.frequency = freq
                    if mz.if_frequency is not None:
                        mz.if_frequency = freq - self.get_lo_readout_frequency(qubit)
                    self.qubits[qubit].readout_frequency = freq

                # resonator_punchout_attenuation
                elif par == "readout_attenuation":
                    self.qubits[qubit].readout.attenuation = value

                # resonator_punchout_attenuation
                elif par == "bare_resonator_frequency":
                    freq = int(value * 1e9)
                    self.qubits[qubit].bare_resonator_frequency = freq

                # resonator_spectroscopy_flux / qubit_spectroscopy_flux
                elif par == "sweetspot":
                    sweetspot = float(value)
                    self.qubits[qubit].sweetspot = sweetspot
                    # set sweetspot as the flux offset (IS THIS NEEDED?)
                    self.qubits[qubit].flux.offset = sweetspot

                # qubit_spectroscopy / qubit_spectroscopy_flux / ramsey
                elif par == "drive_frequency":
                    freq = int(value * 1e9)
                    self.qubits[qubit].native_gates.RX.frequency = freq
                    self.qubits[qubit].drive_frequency = freq

                elif "amplitude" in par:
                    amplitude = float(value)
                    # resonator_spectroscopy
                    if par == "readout_amplitude" and not math.isnan(amplitude):
                        self.qubits[qubit].native_gates.MZ.amplitude = amplitude

                    # rabi_amplitude / flipping
                    if par == "drive_amplitude" or par == "amplitudes":
                        self.qubits[qubit].native_gates.RX.amplitude = amplitude

                # rabi_duration
                elif par == "drive_length":
                    self.qubits[qubit].native_gates.RX.duration = int(value)

                # ramsey
                elif par == "t2":
                    self.qubits[qubit].T2 = float(value)

                # spin_echo
                elif par == "t2_spin_echo":
                    self.qubits[qubit].T2_spin_echo = float(value)

                # t1
                elif par == "t1":
                    self.qubits[qubit].T1 = float(value)

                # classification
                elif par == "threshold":
                    self.qubits[qubit].threshold = float(value)

                # classification
                elif par == "iq_angle":
                    self.qubits[qubit].iq_angle = float(value)

                # classification
                elif par == "mean_gnd_states":
                    self.qubits[qubit].mean_gnd_states = [float(voltage) for voltage in value]

                # classification
                elif par == "mean_exc_states":
                    self.qubits[qubit].mean_exc_states = [float(voltage) for voltage in value]

                # drag pulse tunning
                elif "beta" in par:
                    rx = self.qubits[qubit].native_gates.RX
                    shape = rx.shape
                    rel_sigma = re.findall(r"[\d]+[.\d]+|[\d]*[.][\d]+|[\d]+", shape)[0]
                    rx.shape = f"Drag({rel_sigma}, {float(value)})"

                elif "length" in par:  # assume only drive length
                    self.qubits[qubit].native_gates.RX.duration = int(value)

                elif par == "classifiers_hpars":
                    self.qubits[qubit].classifiers_hpars = value

                else:
                    raise_error(ValueError, f"Unknown parameter {par} for qubit {qubit}")

    def connect(self):
        """Connect to all instruments."""
        if not self.is_connected:
            for instrument in self.instruments.values():
                try:
                    log.info(f"Connecting to instrument {instrument}.")
                    instrument.connect()
                except Exception as exception:
                    raise_error(
                        RuntimeError,
                        f"Cannot establish connection to {instrument} instruments. Error captured: '{exception}'",
                    )
        self.is_connected = True

    def setup(self):
        """Prepares instruments to execute experiments.

        Sets flux port offsets to the qubit sweetspots.
        """
        for instrument in self.instruments.values():
            instrument.setup()
        for qubit in self.qubits.values():
            if qubit.flux is not None and qubit.sweetspot != 0:
                qubit.flux.offset = qubit.sweetspot

    def start(self):
        """Starts all the instruments."""
        if self.is_connected:
            for instrument in self.instruments.values():
                instrument.start()

    def stop(self):
        """Starts all the instruments."""
        if self.is_connected:
            for instrument in self.instruments.values():
                instrument.stop()

    def disconnect(self):
        """Disconnects from instruments."""
        if self.is_connected:
            for instrument in self.instruments.values():
                instrument.disconnect()
        self.is_connected = False

    def _execute(self, method, sequences, options, **kwargs):
        """Executes the sequences on the controllers"""
        if options.nshots is None:
            options = replace(options, nshots=self.settings.nshots)

        if options.relaxation_time is None:
            options = replace(options, relaxation_time=self.settings.relaxation_time)

        result = {}
        for instrument in self.instruments.values():
            if isinstance(instrument, Controller):
                new_result = getattr(instrument, method)(self.qubits, sequences, options)
                if isinstance(new_result, dict):
                    result.update(new_result)
                elif new_result is not None:
                    # currently the result of QMSim is not a dict
                    result = new_result
        return result

    def execute_pulse_sequence(self, sequences: PulseSequence, options: ExecutionParameters, **kwargs):
        """
        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequences to execute.
            options (:class:`qibolab.platforms.platform.ExecutionParameters`): Object holding the execution options.
            **kwargs: May need them for something
        Returns:
            Readout results acquired by after execution.

        """
        return self._execute("play", sequences, options, **kwargs)

    def execute_pulse_sequences(self, sequences: List[PulseSequence], options: ExecutionParameters, **kwargs):
        """
        Args:
            sequence (List[:class:`qibolab.pulses.PulseSequence`]): Pulse sequences to execute.
            options (:class:`qibolab.platforms.platform.ExecutionParameters`): Object holding the execution options.
            **kwargs: May need them for something
        Returns:
            Readout results acquired by after execution.

        """
        return self._execute("play_sequences", sequences, options, **kwargs)

    def sweep(self, sequence: PulseSequence, options: ExecutionParameters, *sweepers: Sweeper):
        """Executes a pulse sequence for different values of sweeped parameters.

        Useful for performing chip characterization.

        Example:
            .. testcode::

                import numpy as np
                from qibolab.dummy import create_dummy
                from qibolab.sweeper import Sweeper, Parameter
                from qibolab.pulses import PulseSequence
                from qibolab.execution_parameters import ExecutionParameters


                platform = create_dummy()
                sequence = PulseSequence()
                parameter = Parameter.frequency
                pulse = platform.create_qubit_readout_pulse(qubit=0, start=0)
                sequence.add(pulse)
                parameter_range = np.random.randint(10, size=10)
                sweeper = Sweeper(parameter, parameter_range, [pulse])
                platform.sweep(sequence, ExecutionParameters(), sweeper)

        Returns:
            Readout results acquired by after execution.
        """
        if options.nshots is None:
            options = replace(options, nshots=self.settings.nshots)

        if options.relaxation_time is None:
            options = replace(options, relaxation_time=self.settings.relaxation_time)

        result = {}
        for instrument in self.instruments.values():
            if isinstance(instrument, Controller):
                new_result = instrument.sweep(self.qubits, sequence, options, *sweepers)
                if isinstance(new_result, dict):
                    result.update(new_result)
                elif new_result is not None:
                    # currently the result of QMSim is not a dict
                    result = new_result
        return result

    def __call__(self, sequence, options):
        return self.execute_pulse_sequence(sequence, options)

    def create_RX90_pulse(self, qubit, start=0, relative_phase=0):
        return self.qubits[qubit].native_gates.RX90.pulse(start, relative_phase)

    def create_RX_pulse(self, qubit, start=0, relative_phase=0):
        return self.qubits[qubit].native_gates.RX.pulse(start, relative_phase)

    def create_CZ_pulse_sequence(self, qubits, start=0):
        # Check in the settings if qubits[0]-qubits[1] is a key
        pair = tuple(sorted(qubits))
        if pair not in self.pairs or self.pairs[pair].native_gates.CZ is None:
            raise_error(
                ValueError,
                f"Calibration for CZ gate between qubits {qubits[0]} and {qubits[1]} not found.",
            )
        return self.pairs[pair].native_gates.CZ.sequence(start)

    def create_MZ_pulse(self, qubit, start):
        return self.qubits[qubit].native_gates.MZ.pulse(start)

    def create_qubit_drive_pulse(self, qubit, start, duration, relative_phase=0):
        pulse = self.qubits[qubit].native_gates.RX.pulse(start, relative_phase)
        pulse.duration = duration
        return pulse

    def create_qubit_readout_pulse(self, qubit, start):
        return self.create_MZ_pulse(qubit, start)

    # TODO Remove RX90_drag_pulse and RX_drag_pulse, replace them with create_qubit_drive_pulse
    # TODO Add RY90 and RY pulses

    def create_RX90_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        pulse = self.qubits[qubit].native_gates.RX90.pulse(start, relative_phase)
        if beta is not None:
            pulse.shape = "Drag(5," + str(beta) + ")"
        return pulse

    def create_RX_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        pulse = self.qubits[qubit].native_gates.RX.pulse(start, relative_phase)
        if beta is not None:
            pulse.shape = "Drag(5," + str(beta) + ")"
        return pulse

    def set_lo_drive_frequency(self, qubit, freq):
        """Set frequency of the qubit drive local oscillator.

        Args:
            qubit (int): qubit whose local oscillator will be modified.
            freq (int): new value of the frequency in Hz.
        """
        self.qubits[qubit].drive.local_oscillator.frequency = freq

    def get_lo_drive_frequency(self, qubit):
        """Get frequency of the qubit drive local oscillator in Hz."""
        return self.qubits[qubit].drive.local_oscillator.frequency

    def set_lo_readout_frequency(self, qubit, freq):
        """Set frequency of the qubit drive local oscillator.

        Args:
            qubit (int): qubit whose local oscillator will be modified.
            freq (int): new value of the frequency in Hz.
        """
        self.qubits[qubit].readout.local_oscillator.frequency = freq

    def get_lo_readout_frequency(self, qubit):
        """Get frequency of the qubit readout local oscillator in Hz."""
        return self.qubits[qubit].readout.local_oscillator.frequency

    def set_lo_twpa_frequency(self, qubit, freq):
        """Set frequency of the local oscillator of the TWPA to which the qubit's feedline is connected to.

        Args:
            qubit (int): qubit whose local oscillator will be modified.
            freq (int): new value of the frequency in Hz.
        """
        self.qubits[qubit].twpa.local_oscillator.frequency = freq

    def get_lo_twpa_frequency(self, qubit):
        """Get frequency of the local oscillator of the TWPA to which the qubit's feedline is connected to in Hz."""
        return self.qubits[qubit].twpa.local_oscillator.frequency

    def set_lo_twpa_power(self, qubit, power):
        """Set power of the local oscillator of the TWPA to which the qubit's feedline is connected to.

        Args:
            qubit (int): qubit whose local oscillator will be modified.
            power (int): new value of the power in dBm.
        self.qubits[qubit].twpa.local_oscillator.power = power
        """

    def get_lo_twpa_power(self, qubit):
        """Get power of the local oscillator of the TWPA to which the qubit's feedline is connected to in dBm."""
        return self.qubits[qubit].twpa.local_oscillator.power

    def set_attenuation(self, qubit, att):
        """Set attenuation value. Usefeul for calibration routines such as punchout.

        Args:
            qubit (int): qubit whose attenuation will be modified.
            att (int): new value of the attenuation (dB).
        Returns:
            None
        """
        self.qubits[qubit].readout.attenuation = att

    def get_attenuation(self, qubit):
        """Get attenuation value. Usefeul for calibration routines such as punchout."""
        return self.qubits[qubit].readout.attenuation

    def set_gain(self, qubit, gain):
        """Set gain value. Usefeul for calibration routines such as Rabi oscillations.

        Args:
            qubit (int): qubit whose attenuation will be modified.
            gain (int): new value of the gain (dimensionless).
        Returns:
            None
        """
        raise_error(NotImplementedError, f"{self.name} does not support gain.")

    def get_gain(self, qubit):
        """Get gain value. Usefeul for calibration routines such as Rabi oscillations."""
        raise_error(NotImplementedError, f"{self.name} does not support gain.")

    def set_bias(self, qubit, bias):
        """Set bias value. Usefeul for calibration routines involving flux.

        Args:
            qubit (int): qubit whose attenuation will be modified.
            bias (int): new value of the bias (V).
        Returns:
            None
        """
        if self.qubits[qubit].flux is None:
            raise_error(NotImplementedError, f"{self.name} does not have flux.")
        self.qubits[qubit].flux.bias = bias

    def get_bias(self, qubit):
        """Get bias value. Usefeul for calibration routines involving flux."""
        return self.qubits[qubit].flux.bias
