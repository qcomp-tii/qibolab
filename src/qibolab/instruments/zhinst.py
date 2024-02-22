"""Instrument for using the Zurich Instruments (Zhinst) devices."""

import copy
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Union

import laboneq.simple as lo
import numpy as np
from laboneq.dsl.experiment.pulse_library import (
    sampled_pulse_complex,
    sampled_pulse_real,
)
from qibo.config import log

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.couplers import Coupler

from qibolab.instruments.unrolling import batch_max_sequences
from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.qubits import Qubit
from qibolab.sweeper import Parameter, Sweeper
from qibolab.unrolling import Bounds

from .abstract import Controller
from .port import Port

SAMPLING_RATE = 2
NANO_TO_SECONDS = 1e-9
COMPILER_SETTINGS = {
    "SHFSG_MIN_PLAYWAVE_HINT": 32,
    "SHFSG_MIN_PLAYZERO_HINT": 32,
    "HDAWG_MIN_PLAYWAVE_HINT": 64,
    "HDAWG_MIN_PLAYZERO_HINT": 64,
}
"""Translating to Zurich ExecutionParameters."""
ACQUISITION_TYPE = {
    AcquisitionType.INTEGRATION: lo.AcquisitionType.INTEGRATION,
    AcquisitionType.RAW: lo.AcquisitionType.RAW,
    AcquisitionType.DISCRIMINATION: lo.AcquisitionType.DISCRIMINATION,
}

AVERAGING_MODE = {
    AveragingMode.CYCLIC: lo.AveragingMode.CYCLIC,
    AveragingMode.SINGLESHOT: lo.AveragingMode.SINGLE_SHOT,
}

SWEEPER_SET = {"amplitude", "frequency", "duration", "relative_phase"}
SWEEPER_BIAS = {"bias"}
SWEEPER_START = {"start"}


def measure_channel_name(qubit: Qubit) -> str:
    """Construct and return a name for qubit's measure channel.

    FIXME: We cannot use channel name directly, because currently channels are named after wires, and due to multiplexed readout
    multiple qubits have the same channel name for their readout. Should be fixed once channels are refactored.
    """
    return f"{qubit.readout.name}_{qubit.name}"


def acquire_channel_name(qubit: Qubit) -> str:
    """Construct and return a name for qubit's acquire channel.

    FIXME: We cannot use acquire channel name, because qibolab does not have a concept of acquire channel. This function shall be removed
    once all channel refactoring is done.
    """
    return f"acquire{qubit.name}"


def select_pulse(pulse, pulse_type):
    """Pulse translation."""

    if "IIR" not in str(pulse.shape):
        if str(pulse.shape) == "Rectangular()":
            can_compress = pulse.type is not PulseType.READOUT
            return lo.pulse_library.const(
                uid=f"{pulse_type}_{pulse.qubit}_",
                length=round(pulse.duration * NANO_TO_SECONDS, 9),
                amplitude=pulse.amplitude,
                can_compress=can_compress,
            )
        if "Gaussian" in str(pulse.shape):
            sigma = pulse.shape.rel_sigma
            return lo.pulse_library.gaussian(
                uid=f"{pulse_type}_{pulse.qubit}_",
                length=round(pulse.duration * NANO_TO_SECONDS, 9),
                amplitude=pulse.amplitude,
                sigma=2 / sigma,
                zero_boundaries=False,
            )

        if "GaussianSquare" in str(pulse.shape):
            sigma = pulse.shape.rel_sigma
            width = pulse.shape.width
            can_compress = pulse.type is not PulseType.READOUT
            return lo.pulse_library.gaussian_square(
                uid=f"{pulse_type}_{pulse.qubit}_",
                length=round(pulse.duration * NANO_TO_SECONDS, 9),
                width=round(pulse.duration * NANO_TO_SECONDS, 9) * width,
                amplitude=pulse.amplitude,
                can_compress=can_compress,
                sigma=2 / sigma,
                zero_boundaries=False,
            )

        if "Drag" in str(pulse.shape):
            sigma = pulse.shape.rel_sigma
            beta = pulse.shape.beta
            return lo.pulse_library.drag(
                uid=f"{pulse_type}_{pulse.qubit}_",
                length=round(pulse.duration * NANO_TO_SECONDS, 9),
                amplitude=pulse.amplitude,
                sigma=2 / sigma,
                beta=beta,
                zero_boundaries=False,
            )

    if np.all(pulse.envelope_waveform_q(SAMPLING_RATE).data == 0):
        return sampled_pulse_real(
            uid=f"{pulse_type}_{pulse.qubit}_",
            samples=pulse.envelope_waveform_i(SAMPLING_RATE).data,
            can_compress=True,
        )
    else:
        # Test this when we have pulses that use it
        return sampled_pulse_complex(
            uid=f"{pulse_type}_{pulse.qubit}_",
            samples=pulse.envelope_waveform_i(SAMPLING_RATE).data
            + (1j * pulse.envelope_waveform_q(SAMPLING_RATE).data),
            can_compress=True,
        )


def select_sweeper(sweeper, ptype=None, qubit=None):
    """Sweeper translation."""
    if sweeper.parameter in (Parameter.duration, Parameter.start):
        return lo.SweepParameter(
            uid=sweeper.parameter.name,
            values=sweeper.values * NANO_TO_SECONDS,
        )
    if sweeper.parameter is Parameter.frequency:
        if ptype is None or qubit is None:
            raise ValueError(
                "For frequency sweep ptype and qubit arguments are mandatory"
            )  # FIXME
        if ptype is PulseType.READOUT:
            intermediate_frequency = (
                qubit.readout_frequency - qubit.readout.local_oscillator.frequency
            )
        elif ptype is PulseType.DRIVE:
            intermediate_frequency = (
                qubit.drive_frequency - qubit.drive.local_oscillator.frequency
            )
        else:
            raise ValueError(
                f"Cannot sweep frequency of pulse of type {ptype}, because it does not have associated frequency"
            )
        return lo.LinearSweepParameter(
            uid=sweeper.parameter.name,
            start=sweeper.values[0] + intermediate_frequency,
            stop=sweeper.values[-1] + intermediate_frequency,
            count=len(sweeper.values),
        )

    return lo.SweepParameter(uid=sweeper.parameter.name, values=sweeper.values)


def classify_sweepers(sweepers):
    """"""
    nt_sweepers = []
    rt_sweepers = []
    for sweeper in sweepers:
        if (
            sweeper.parameter is Parameter.amplitude
            and sweeper.pulses[0].type is PulseType.READOUT
        ):
            nt_sweepers.append(sweeper)
        elif sweeper.parameter.name in SWEEPER_BIAS:
            nt_sweepers.append(sweeper)
        else:
            rt_sweepers.append(sweeper)
    return nt_sweepers, rt_sweepers


@dataclass
class ZhPort(Port):
    name: Tuple[str, str]
    offset: float = 0.0
    power_range: int = 0


class ZhPulse:
    """Zurich pulse from qibolab pulse translation."""

    def __init__(self, pulse):
        """Zurich pulse from qibolab pulse."""
        self.pulse = pulse
        """Qibolab pulse."""
        self.zhpulse = select_pulse(pulse, pulse.type.name.lower())
        """Zurich pulse."""
        self.zhsweepers = []

        self.delay_sweeper = None

    # pylint: disable=R0903
    def add_sweeper(self, sweeper, qubit):
        """Add sweeper to list of sweepers associated with this pulse."""
        self.zhsweepers.append(
            (sweeper.parameter, select_sweeper(sweeper, self.pulse.type, qubit))
        )

    def add_delay_sweeper(self, sweeper):
        self.delay_sweeper = select_sweeper(sweeper)


@dataclass
class SubSequence:
    measurements: list[str, Pulse]
    control_sequence: dict[str, list[Pulse]]


class Zurich(Controller):
    """Zurich driver main class."""

    PortType = ZhPort

    def __init__(self, name, device_setup, time_of_flight=0.0, smearing=0.0):
        super().__init__(name, None)

        self.signal_map = {}
        "Signals to lines mapping"
        self.calibration = lo.Calibration()
        "Zurich calibration object)"

        self.device_setup = device_setup
        self.session = None
        "Zurich device parameters for connection"

        self.time_of_flight = time_of_flight
        self.smearing = smearing
        "Parameters read from the runcard not part of ExecutionParameters"

        self.exp = None
        self.experiment = None
        self.results = None
        "Zurich experiment definitions"

        self.bounds = Bounds(
            waveforms=int(4e4),
            readout=250,
            instructions=int(1e6),
        )

        self.acquisition_type = None
        "To store if the AcquisitionType.SPECTROSCOPY needs to be enabled by parsing the sequence"

        self.sequence = defaultdict(list)
        "Zurich pulse sequence"
        self.sequence_qibo = None
        self.sub_sequences = {}
        "Sub sequences between each measurement"

        self.sweepers = []
        self.nt_sweeps = []
        self.rt_sweeps = []
        "Storing sweepers"
        # Improve the storing of multiple sweeps

    @property
    def sampling_rate(self):
        return SAMPLING_RATE

    def connect(self):
        if self.is_connected is False:
            # To fully remove logging #configure_logging=False
            # I strongly advise to set it to 20 to have time estimates of the experiment duration!
            self.session = lo.Session(self.device_setup, log_level=20)
            _ = self.session.connect()
            self.is_connected = True

    def disconnect(self):
        if self.is_connected:
            _ = self.session.disconnect()
            self.is_connected = False

    def calibration_step(self, qubits, couplers, options):
        """Zurich general pre experiment calibration definitions.

        Change to get frequencies from sequence
        """

        for coupler in couplers.values():
            self.register_couplerflux_line(coupler)

        for qubit in qubits.values():
            if qubit.flux is not None:
                self.register_flux_line(qubit)
            if len(self.sequence[qubit.drive.name]) != 0:
                self.register_drive_line(
                    qubit=qubit,
                    intermediate_frequency=qubit.drive_frequency
                    - qubit.drive.local_oscillator.frequency,
                )
            if len(self.sequence[measure_channel_name(qubit)]) != 0:
                self.register_readout_line(
                    qubit=qubit,
                    intermediate_frequency=qubit.readout_frequency
                    - qubit.readout.local_oscillator.frequency,
                    options=options,
                )
        self.device_setup.set_calibration(self.calibration)

    def register_readout_line(self, qubit, intermediate_frequency, options):
        """Registers qubit measure and acquire lines to calibration and signal
        map.

        Note
        ----
        To allow debugging with and oscilloscope, just set the following::

            self.calibration[f"/logical_signal_groups/q{q}/measure_line"] = lo.SignalCalibration(
                ...,
                local_oscillator=lo.Oscillator(
                    ...
                    frequency=0.0,
                ),
                ...,
                port_mode=lo.PortMode.LF,
                ...,
            )
        """

        q = qubit.name  # pylint: disable=C0103
        self.signal_map[measure_channel_name(qubit)] = (
            self.device_setup.logical_signal_groups[f"q{q}"].logical_signals[
                "measure_line"
            ]
        )
        self.calibration[f"/logical_signal_groups/q{q}/measure_line"] = (
            lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=intermediate_frequency,
                    modulation_type=lo.ModulationType.SOFTWARE,
                ),
                local_oscillator=lo.Oscillator(
                    uid="lo_shfqa_m" + str(q),
                    frequency=int(qubit.readout.local_oscillator.frequency),
                ),
                range=qubit.readout.power_range,
                port_delay=None,
                delay_signal=0,
            )
        )

        self.signal_map[acquire_channel_name(qubit)] = (
            self.device_setup.logical_signal_groups[f"q{q}"].logical_signals[
                "acquire_line"
            ]
        )

        oscillator = lo.Oscillator(
            frequency=intermediate_frequency,
            modulation_type=lo.ModulationType.SOFTWARE,
        )
        threshold = None

        if options.acquisition_type == AcquisitionType.DISCRIMINATION:
            if qubit.kernel is not None:
                # Kernels don't work with the software modulation on the acquire signal
                oscillator = None
            else:
                # To keep compatibility with angle and threshold discrimination (Remove when possible)
                threshold = qubit.threshold

        self.calibration[f"/logical_signal_groups/q{q}/acquire_line"] = (
            lo.SignalCalibration(
                oscillator=oscillator,
                range=qubit.feedback.power_range,
                port_delay=self.time_of_flight * NANO_TO_SECONDS,
                threshold=threshold,
            )
        )

    def register_drive_line(self, qubit, intermediate_frequency):
        """Registers qubit drive line to calibration and signal map."""
        q = qubit.name  # pylint: disable=C0103
        self.signal_map[qubit.drive.name] = self.device_setup.logical_signal_groups[
            f"q{q}"
        ].logical_signals["drive_line"]
        self.calibration[f"/logical_signal_groups/q{q}/drive_line"] = (
            lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=intermediate_frequency,
                    modulation_type=lo.ModulationType.HARDWARE,
                ),
                local_oscillator=lo.Oscillator(
                    uid="lo_shfqc" + str(q),
                    frequency=int(qubit.drive.local_oscillator.frequency),
                ),
                range=qubit.drive.power_range,
                port_delay=None,
                delay_signal=0,
            )
        )

    def register_flux_line(self, qubit):
        """Registers qubit flux line to calibration and signal map."""
        q = qubit.name  # pylint: disable=C0103
        self.signal_map[qubit.flux.name] = self.device_setup.logical_signal_groups[
            f"q{q}"
        ].logical_signals["flux_line"]
        self.calibration[f"/logical_signal_groups/q{q}/flux_line"] = (
            lo.SignalCalibration(
                range=qubit.flux.power_range,
                port_delay=None,
                delay_signal=0,
                voltage_offset=qubit.flux.offset,
            )
        )

    def register_couplerflux_line(self, coupler):
        """Registers qubit flux line to calibration and signal map."""
        c = coupler.name  # pylint: disable=C0103
        self.signal_map[coupler.flux.name] = self.device_setup.logical_signal_groups[
            f"qc{c}"
        ].logical_signals["flux_line"]
        self.calibration[f"/logical_signal_groups/qc{c}/flux_line"] = (
            lo.SignalCalibration(
                range=coupler.flux.power_range,
                port_delay=None,
                delay_signal=0,
                voltage_offset=coupler.flux.offset,
            )
        )

    def run_exp(self):
        """
        Compilation settings, compilation step, execution step and data retrival
        - Save a experiment Python object:
        self.experiment.save("saved_exp")
        - Save a experiment compiled experiment ():
        self.exp.save("saved_exp")  # saving compiled experiment
        """
        self.exp = self.session.compile(
            self.experiment, compiler_settings=COMPILER_SETTINGS
        )
        self.results = self.session.run(self.exp)

    @staticmethod
    def frequency_from_pulses(qubits, sequence):
        """Gets the frequencies from the pulses to the qubits."""
        for pulse in sequence:
            qubit = qubits[pulse.qubit]
            if pulse.type is PulseType.READOUT:
                qubit.readout_frequency = pulse.frequency
            if pulse.type is PulseType.DRIVE:
                qubit.drive_frequency = pulse.frequency

    def create_sub_sequences(self, qubits: list[Qubit]) -> list[SubSequence]:
        """Create subsequences for different lines (drive, flux, coupler flux).

        Args:
            qubits (dict[str, Qubit]): qubits for the platform.
            couplers (dict[str, Coupler]): couplers for the platform.
        """
        # collect and group all measurements. In one sequence, qubits may be measured multiple times (e.g. when the sequence was unrolled
        # at platform level), however it is assumed that 1. all qubits are measured the same number of times, 2. each time all qubits are
        # measured simultaneously.
        measure_channels = {measure_channel_name(qb) for qb in qubits}
        other_channels = set(self.sequence.keys()) - measure_channels

        measurement_groups = defaultdict(list)
        for ch in measure_channels:
            for i, pulse in enumerate(self.sequence[ch]):
                measurement_groups[i].append((ch, pulse))

        measurement_starts = {}
        for i, group in measurement_groups.items():
            starts = np.array([meas.pulse.start for _, meas in group])
            # TODO: validate that all start times are equal
            measurement_starts[i] = max(starts)

        # split all non-measurement channels according to the locations of the measurements
        sub_sequences = defaultdict(lambda: defaultdict(list))
        for ch in other_channels:
            measurement_index = 0
            for pulse in self.sequence[ch]:
                if pulse.pulse.finish > measurement_starts[measurement_index]:
                    measurement_index += 1
                sub_sequences[measurement_index][ch].append(pulse)
        if len(sub_sequences) > len(measurement_groups):
            log.warning("There are control pulses after the last measurement start.")

        return [
            SubSequence(measurement_groups[i], sub_sequences[i])
            for i in range(len(measurement_groups))
        ]

    def experiment_flow(
        self,
        qubits: Dict[str, Qubit],
        couplers: Dict[str, Coupler],
        sequence: PulseSequence,
        options: ExecutionParameters,
    ):
        """Create the experiment object for the devices, following the steps
        separated one on each method:

        Translation, Calibration, Experiment Definition.

        Args:
            qubits (dict[str, Qubit]): qubits for the platform.
            couplers (dict[str, Coupler]): couplers for the platform.
            sequence (PulseSequence): sequence of pulses to be played in the experiment.
        """
        self.sequence_zh(sequence, qubits, couplers)
        self.sub_sequences = self.create_sub_sequences(list(qubits.values()))
        self.calibration_step(qubits, couplers, options)
        self.create_exp(qubits, couplers, options)

    # pylint: disable=W0221
    def play(self, qubits, couplers, sequence, options):
        """Play pulse sequence."""
        self.signal_map = {}

        self.frequency_from_pulses(qubits, sequence)

        self.experiment_flow(qubits, couplers, sequence, options)

        self.run_exp()

        # Get the results back
        results = {}
        for qubit in qubits.values():
            q = qubit.name  # pylint: disable=C0103
            for i, ropulse in enumerate(self.sequence[measure_channel_name(qubit)]):
                data = np.array(self.results.get_data(f"sequence{q}_{i}"))
                if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                    data = (
                        np.ones(data.shape) - data.real
                    )  # Probability inversion patch
                serial = ropulse.pulse.serial
                qubit = ropulse.pulse.qubit
                results[serial] = results[qubit] = options.results_type(data)

        return results

    def sequence_zh(self, sequence, qubits, couplers):
        """Qibo sequence to Zurich sequence."""
        # Define and assign the sequence
        zhsequence = defaultdict(list)
        self.sequence_qibo = sequence

        # Fill the sequences with pulses according to their lines in temporal order
        for pulse in sequence:
            if pulse.type == PulseType.READOUT:
                ch = measure_channel_name(qubits[pulse.qubit])
            else:
                ch = pulse.channel
            zhsequence[ch].append(ZhPulse(pulse))

        def get_sweeps(pulse: Pulse):
            sweepers_for_pulse = []
            for sweeper in self.sweepers:
                for p in sweeper.pulses or []:
                    if p == pulse:
                        sweepers_for_pulse.append(sweeper)
                        break
            return sweepers_for_pulse

        for ch, zhpulses in zhsequence.items():
            for i, zhpulse in enumerate(zhpulses):
                for s in get_sweeps(zhpulse.pulse):
                    if s.parameter.name in SWEEPER_SET:
                        zhpulse.add_sweeper(s, qubits[zhpulse.pulse.qubit])
                    if s.parameter.name in SWEEPER_START:
                        zhpulse.add_delay_sweeper(s)

        self.sequence = zhsequence

        def set_acquisition_type():
            for sweeper in self.sweepers:
                if sweeper.parameter.name in {Parameter.frequency, Parameter.amplitude}:
                    for pulse in sweeper.pulses:
                        if pulse.type is PulseType.READOUT:
                            # FIXME: case of multiple pulses
                            self.acquisition_type = lo.AcquisitionType.SPECTROSCOPY

        set_acquisition_type()

    def create_exp(self, qubits, couplers, options):
        """Zurich experiment initialization using their Experiment class."""

        # Setting experiment signal lines
        signals = [lo.ExperimentSignal(name) for name in self.signal_map.keys()]

        exp = lo.Experiment(
            uid="Sequence",
            signals=signals,
        )

        if self.acquisition_type:
            acquisition_type = self.acquisition_type
            self.acquisition_type = None
        else:
            acquisition_type = ACQUISITION_TYPE[options.acquisition_type]
        averaging_mode = AVERAGING_MODE[options.averaging_mode]
        exp_options = replace(
            options, acquisition_type=acquisition_type, averaging_mode=averaging_mode
        )

        exp_calib = lo.Calibration()
        # Near Time recursion loop or directly to Real Time recursion loop
        if self.nt_sweeps:
            self.sweep_recursion_nt(qubits, couplers, exp_options, exp, exp_calib)
        else:
            self.define_exp(qubits, couplers, exp_options, exp, exp_calib)

    def define_exp(self, qubits, couplers, exp_options, exp, exp_calib):
        """Real time definition."""
        with exp.acquire_loop_rt(
            uid="shots",
            count=exp_options.nshots,
            acquisition_type=exp_options.acquisition_type,
            averaging_mode=exp_options.averaging_mode,
        ):
            # Recursion loop for sweepers or just play a sequence
            if len(self.rt_sweeps) > 0:
                self.sweep_recursion(qubits, couplers, exp, exp_calib, exp_options)
            else:
                self.select_exp(exp, qubits, couplers, exp_options)
            exp.set_calibration(exp_calib)
            exp.set_signal_map(self.signal_map)
            self.experiment = exp

    def select_exp(self, exp, qubits, couplers, exp_options):
        """Build Zurich Experiment selecting the relevant sections."""
        weights = {}
        previous_section = None
        for i, seq in enumerate(self.sub_sequences):
            section_uid = f"control_{i}"
            with exp.section(uid=section_uid, play_after=previous_section):
                for ch, pulses in seq.control_sequence.items():
                    time = 0
                    for j, pulse in enumerate(pulses):
                        if pulse.delay_sweeper:
                            exp.delay(signal=ch, time=pulse.delay_sweeper)
                        exp.delay(
                            signal=ch,
                            time=round(pulse.pulse.start * NANO_TO_SECONDS, 9) - time,
                        )
                        time = round(pulse.pulse.duration * NANO_TO_SECONDS, 9) + round(
                            pulse.pulse.start * NANO_TO_SECONDS, 9
                        )
                        pulse.zhpulse.uid += (
                            f"{i}_{j}"  # FIXME: is this actually needed?
                        )
                        if pulse.zhsweepers:
                            self.play_sweep(exp, ch, pulse)
                        else:
                            exp.play(
                                signal=ch,
                                pulse=pulse.zhpulse,
                                phase=pulse.pulse.relative_phase,
                            )
            previous_section = section_uid

            if any(m.delay_sweeper is not None for _, m in seq.measurements):
                section_uid = f"measurement_delay_{i}"
                with exp.section(uid=section_uid, play_after=previous_section):
                    for ch, m in seq.measurements:
                        if m.delay_sweeper:
                            exp.delay(signal=ch, time=m.delay_sweeper)
                previous_section = section_uid

            section_uid = f"measure_{i}"
            with exp.section(uid=section_uid, play_after=previous_section):
                for ch, pulse in seq.measurements:
                    qubit = qubits[pulse.pulse.qubit]
                    q = qubit.name
                    pulse.zhpulse.uid += str(i)

                    exp.delay(
                        signal=acquire_channel_name(qubit),
                        time=self.smearing * NANO_TO_SECONDS,
                    )

                    if (
                        qubit.kernel is not None
                        and exp_options.acquisition_type
                        == lo.AcquisitionType.DISCRIMINATION
                    ):
                        weight = lo.pulse_library.sampled_pulse_complex(
                            uid="weight" + str(q),
                            samples=qubit.kernel * np.exp(1j * qubit.iq_angle),
                        )

                    else:
                        if i == 0:
                            if (
                                exp_options.acquisition_type
                                == lo.AcquisitionType.DISCRIMINATION
                            ):
                                weight = lo.pulse_library.sampled_pulse_complex(
                                    samples=np.ones(
                                        [
                                            int(
                                                pulse.pulse.duration * 2
                                                - 3 * self.smearing * NANO_TO_SECONDS
                                            )
                                        ]
                                    )
                                    * np.exp(1j * qubit.iq_angle),
                                    uid="weights" + str(q),
                                )
                                weights[q] = weight
                            else:
                                # TODO: Patch for multiple readouts: Remove different uids
                                weight = lo.pulse_library.const(
                                    uid="weight" + str(q),
                                    length=round(
                                        pulse.pulse.duration * NANO_TO_SECONDS, 9
                                    )
                                    - 1.5 * self.smearing * NANO_TO_SECONDS,
                                    amplitude=1,
                                )

                                weights[q] = weight
                        elif i != 0:
                            weight = weights[q]

                    measure_pulse_parameters = {"phase": 0}

                    if i == len(self.sequence[measure_channel_name(qubit)]) - 1:
                        reset_delay = exp_options.relaxation_time * NANO_TO_SECONDS
                    else:
                        reset_delay = 0

                    exp.measure(
                        acquire_signal=acquire_channel_name(qubit),
                        handle=f"sequence{q}_{i}",
                        integration_kernel=weight,
                        integration_kernel_parameters=None,
                        integration_length=None,
                        measure_signal=measure_channel_name(qubit),
                        measure_pulse=pulse.zhpulse,
                        measure_pulse_length=round(
                            pulse.pulse.duration * NANO_TO_SECONDS, 9
                        ),
                        measure_pulse_parameters=measure_pulse_parameters,
                        measure_pulse_amplitude=None,
                        acquire_delay=self.time_of_flight * NANO_TO_SECONDS,
                        reset_delay=reset_delay,
                    )
            previous_section = section_uid

    @staticmethod
    def play_sweep(exp, channel_name, pulse):
        """Play Zurich pulse when a single sweeper is involved."""
        play_parameters = {}
        for p, zhs in pulse.zhsweepers:
            if p is Parameter.amplitude:
                pulse.zhpulse.amplitude *= max(zhs.values)
                zhs.values /= max(zhs.values)
                play_parameters["amplitude"] = zhs
            if p is Parameter.duration:
                play_parameters["length"] = zhs
            if p is Parameter.relative_phase:
                play_parameters["phase"] = zhs
        if "phase" not in play_parameters:
            play_parameters["phase"] = pulse.pulse.relative_phase

        exp.play(signal=channel_name, pulse=pulse.zhpulse, **play_parameters)

    @staticmethod
    def rearrange_sweepers(sweepers: List[Sweeper]) -> Tuple[np.ndarray, List[Sweeper]]:
        """Rearranges sweepers from qibocal based on device hardware
        limitations.

        Frequency sweepers must be applied before (on the outer loop) bias or amplitude sweepers.

        Args:
            sweepers (list[Sweeper]): list of sweepers used in the experiment.

        Returns:
            rearranging_axes (np.ndarray): array of shape (2,) and dtype=int containing
                the indexes of the sweepers to be swapped. Defaults to np.array([0, 0])
                if no swap is needed.
            sweepers (list[Sweeper]): updated list of sweepers used in the experiment. If
                sweepers must be swapped, the list is updated accordingly.
        """
        swapped_axis_pair = np.zeros(2, dtype=int)
        freq_sweeper = next(
            iter(s for s in sweepers if s.parameter is Parameter.frequency), None
        )
        if freq_sweeper:
            freq_sweeper_idx = sweepers.index(freq_sweeper)
            sweepers[freq_sweeper_idx] = sweepers[0]
            sweepers[0] = freq_sweeper
            swapped_axis_pair = np.array([0, freq_sweeper_idx])
            log.warning("Sweepers were reordered")
        return swapped_axis_pair, sweepers

    def sweep(self, qubits, couplers, sequence: PulseSequence, options, *sweepers):
        """Play pulse and sweepers sequence."""

        self.signal_map = {}
        self.nt_sweeps, self.rt_sweeps = classify_sweepers(self.sweepers)
        swapped_axis_pair, self.rt_sweeps = self.rearrange_sweepers(self.rt_sweeps)
        self.sweepers = list(sweepers)
        swapped_axis_pair += len(self.nt_sweeps)
        # if using singleshot, the first axis contains shots,
        # i.e.: (nshots, sweeper_1, sweeper_2)
        # if using integration: (sweeper_1, sweeper_2)
        if options.averaging_mode is AveragingMode.SINGLESHOT:
            swapped_axis_pair += 1

        self.frequency_from_pulses(qubits, sequence)

        self.experiment_flow(qubits, couplers, sequence, options)
        self.run_exp()

        #  Get the results back
        results = {}
        for qubit in qubits.values():
            q = qubit.name  # pylint: disable=C0103
            for i, ropulse in enumerate(self.sequence[measure_channel_name(qubit)]):
                exp_res = self.results.get_data(f"sequence{q}_{i}")

                # Reorder dimensions
                data = np.moveaxis(exp_res, swapped_axis_pair[0], swapped_axis_pair[1])
                if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                    data = (
                        np.ones(data.shape) - data.real
                    )  # Probability inversion patch

                serial = ropulse.pulse.serial
                qubit = ropulse.pulse.qubit
                results[serial] = results[qubit] = options.results_type(data)

        return results

    def sweep_recursion(self, qubits, couplers, exp, exp_calib, exp_options):
        """Sweepers recursion for multiple nested Real Time sweepers."""

        sweeper = self.rt_sweeps[0]

        i = len(self.rt_sweeps) - 1
        self.rt_sweeps.remove(sweeper)
        parameter = None

        if sweeper.parameter is Parameter.frequency:
            for pulse in sweeper.pulses:
                line = "drive" if pulse.type is PulseType.DRIVE else "readout"
                zhsweeper = select_sweeper(
                    sweeper, pulse.type, qubits[sweeper.pulses[0].qubit]
                )
                if line == "readout":
                    channel_name = measure_channel_name(qubits[pulse.qubit])
                else:
                    channel_name = getattr(qubits[pulse.qubit], line).name
                exp_calib[channel_name] = lo.SignalCalibration(
                    oscillator=lo.Oscillator(
                        frequency=zhsweeper,
                        modulation_type=lo.ModulationType.HARDWARE,
                    )
                )
        if sweeper.parameter is Parameter.amplitude:
            for pulse in sweeper.pulses:
                pulse = pulse.copy()
                pulse.amplitude *= max(abs(sweeper.values))

                aux_max = max(abs(sweeper.values))

                sweeper.values /= aux_max
                parameter = select_sweeper(sweeper)
                sweeper.values *= aux_max

        if sweeper.parameter is Parameter.bias:
            parameter = select_sweeper(sweeper)

        elif sweeper.parameter is Parameter.start:
            parameter = select_sweeper(sweeper)

        elif parameter is None:
            parameter = select_sweeper(
                sweeper, sweeper.pulses[0].type, qubits[sweeper.pulses[0].qubit]
            )

        with exp.sweep(
            uid=f"sweep_{sweeper.parameter.name.lower()}_{i}",
            parameter=parameter,
            reset_oscillator_phase=True,
        ):
            if len(self.rt_sweeps) > 0:
                self.sweep_recursion(qubits, couplers, exp, exp_calib, exp_options)
            else:
                self.select_exp(exp, qubits, couplers, exp_options)

    def find_instrument_address(
        self, quantum_element: Union[Qubit, Coupler], parameter: str
    ) -> str:
        """Find path of the instrument connected to a specified line and
        qubit/coupler.

        Args:
            quantum_element (Qubit | Coupler): qubits or couplers on which perform the near time sweep.
            parameter (str): parameter on which perform the near time sweep.
        """
        line_names = {
            "bias": "flux",
            "amplitude": "drive",
        }
        line_name = line_names[parameter]
        channel_uid = (
            self.device_setup.logical_signal_groups[f"q{quantum_element.name}"]
            .logical_signals[f"{line_name}_line"]
            .physical_channel.uid
        )
        channel_name = channel_uid.split("/")[0]
        instruments = self.device_setup.instruments
        for instrument in instruments:
            if instrument.uid == channel_name:
                return instrument.address
        raise RuntimeError(
            f"Could not find instrument for {quantum_element} {line_name}"
        )

    def sweep_recursion_nt(
        self,
        qubits: Dict[str, Qubit],
        couplers: Dict[str, Coupler],
        options: ExecutionParameters,
        exp: lo.Experiment,
        exp_calib: lo.Calibration,
    ):
        """Sweepers recursion for Near Time sweepers. Faster than regular
        software sweepers as they are executed on the actual device by
        (software ? or slower hardware ones)

        You want to avoid them so for now they are implement for a
        specific sweep.
        """

        log.info("nt Loop")

        sweeper = self.nt_sweeps[0]

        i = len(self.nt_sweeps) - 1
        self.nt_sweeps.remove(sweeper)

        parameter = None

        if sweeper.parameter is Parameter.bias:
            if sweeper.qubits:
                for qubit in sweeper.qubits:
                    zhsweeper = select_sweeper(sweeper)
                    zhsweeper.uid = "bias"
                    path = self.find_instrument_address(qubit, "bias")

                    parameter = copy.deepcopy(zhsweeper)
                    parameter.values += qubit.flux.offset
                    device_path = f"{path}/sigouts/0/offset"

        elif sweeper.parameter is Parameter.amplitude:
            for pulse in sweeper.pulses:
                pulse = pulse.copy()
                pulse.amplitude *= max(abs(sweeper.values))

                aux_max = max(abs(sweeper.values))

                sweeper.values /= aux_max
                zhsweeper = select_sweeper(sweeper)
                sweeper.values *= aux_max

                zhsweeper.uid = "amplitude"
                path = self.find_instrument_address(
                    qubits[sweeper.pulses[0].qubit], "amplitude"
                )
                parameter = zhsweeper
                device_path = (
                    f"/{path}/qachannels/*/oscs/0/gain"  # Hardcoded SHFQA device
                )

        elif parameter is None:
            parameter = select_sweeper(
                sweeper, sweeper.pulses[0].type, qubits[sweeper.pulses[0].qubit]
            )
            device_path = f"/{path}/qachannels/*/oscs/0/gain"  # Hardcoded SHFQA device

        with exp.sweep(
            uid=f"sweep_{sweeper.parameter.name.lower()}_{i}",
            parameter=parameter,
        ):
            exp.set_node(
                path=device_path,
                value=parameter,
            )

            if len(self.nt_sweeps) > 0:
                self.sweep_recursion_nt(qubits, couplers, options, exp, exp_calib)
            else:
                self.define_exp(qubits, couplers, options, exp, exp_calib)

    def split_batches(self, sequences):
        return batch_max_sequences(sequences, MAX_SEQUENCES)
