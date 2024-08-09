from collections import defaultdict
from dataclasses import dataclass, field

from qibo import Circuit, gates
from qibo.config import raise_error

from qibolab.compilers.default import (
    cnot_rule,
    cz_rule,
    gpi2_rule,
    gpi_rule,
    identity_rule,
    measurement_rule,
    rz_rule,
    z_rule,
)
from qibolab.platform import Platform
from qibolab.pulses import Delay, PulseSequence
from qibolab.qubits import QubitId


@dataclass
class Compiler:
    """Compiler that transforms a :class:`qibo.models.Circuit` to a
    :class:`qibolab.pulses.PulseSequence`.

    The transformation is done using a dictionary of rules which map each Qibo gate to a
    pulse sequence and some virtual Z-phases.

    A rule is a function that takes two argumens:
        - gate (:class:`qibo.gates.abstract.Gate`): Gate object to be compiled.
        - platform (:class:`qibolab.platforms.abstract.AbstractPlatform`): Platform object to read
            native gate pulses from.

    and returns:
        - sequence (:class:`qibolab.pulses.PulseSequence`): Sequence of pulses that implement
            the given gate.
        - virtual_z_phases (dict): Dictionary mapping qubits to virtual Z-phases induced by the gate.

    See :class:`qibolab.compilers.default` for an example of a compiler implementation.
    """

    rules: dict = field(default_factory=dict)
    """Map from gates to compilation rules."""

    @classmethod
    def default(cls):
        return cls(
            {
                gates.I: identity_rule,
                gates.Z: z_rule,
                gates.RZ: rz_rule,
                gates.CZ: cz_rule,
                gates.CNOT: cnot_rule,
                gates.GPI2: gpi2_rule,
                gates.GPI: gpi_rule,
                gates.M: measurement_rule,
            }
        )

    def __setitem__(self, key, rule):
        """Sets a new rule to the compiler.

        If a rule already exists for the gate, it will be overwritten.
        """
        self.rules[key] = rule

    def __getitem__(self, item):
        """Get an existing rule for a given gate."""
        try:
            return self.rules[item]
        except KeyError:
            raise_error(KeyError, f"Compiler rule not available for {item}.")

    def __delitem__(self, item):
        """Remove rule for the given gate."""
        try:
            del self.rules[item]
        except KeyError:
            raise_error(
                KeyError,
                f"Cannot remove {item} from compiler because it does not exist.",
            )

    def register(self, gate_cls):
        """Decorator for registering a function as a rule in the compiler.

        Using this decorator is optional. Alternatively the user can set the rules directly
        via ``__setitem__``.

        Args:
            gate_cls: Qibo gate object that the rule will be assigned to.
        """

        def inner(func):
            self[gate_cls] = func
            return func

        return inner

    def get_sequence(self, gate, platform):
        """Get pulse sequence implementing the given gate using the registered
        rules.

        Args:
            gate (:class:`qibo.gates.Gate`): Qibo gate to convert to pulses.
            platform (:class:`qibolab.platform.Platform`): Qibolab platform to read the native gates from.
        """
        # get local sequence for the current gate
        rule = self[type(gate)]
        if isinstance(gate, gates.M):
            qubits = [platform.get_qubit(q) for q in gate.qubits]
            gate_sequence = rule(gate, qubits)
        elif len(gate.qubits) == 1:
            qubit = platform.get_qubit(gate.target_qubits[0])
            gate_sequence = rule(gate, qubit)
        elif len(gate.qubits) == 2:
            pair = platform.pairs[
                tuple(platform.get_qubit(q).name for q in gate.qubits)
            ]
            gate_sequence = rule(gate, pair)
        else:
            raise NotImplementedError(f"{type(gate)} is not a native gate.")
        return gate_sequence

    # FIXME: pulse.qubit and pulse.channel do not exist anymore
    def compile(self, circuit: Circuit, platform: Platform):
        """Transforms a circuit to pulse sequence.

        Args:
            circuit (qibo.models.Circuit): Qibo circuit that respects the platform's
                                           connectivity and native gates.
            platform (qibolab.platforms.abstract.AbstractPlatform): Platform used
                to load the native pulse representations.

        Returns:
            sequence (qibolab.pulses.PulseSequence): Pulse sequence that implements the circuit.
            measurement_map (dict): Map from each measurement gate to the sequence of  readout pulse implementing it.
        """
        ch_to_qb = platform.channels_map

        sequence = PulseSequence()
        # FIXME: This will not work with qubits that have string names
        # TODO: Implement a mapping between circuit qubit ids and platform ``Qubit``s

        measurement_map = {}
        channel_clock = defaultdict(int)

        def qubit_clock(qubit: QubitId):
            return max(channel_clock[ch] for ch in platform.qubits[qubit].channels)

        # process circuit gates
        for moment in circuit.queue.moments:
            for gate in set(filter(lambda x: x is not None, moment)):
                if isinstance(gate, gates.Align):
                    for qubit in gate.qubits:
                        clock = qubit_clock(qubit)
                        for ch in platform.qubits[qubit].channels:
                            channel_clock[qubit] = clock + gate.delay
                    continue

                delay_sequence = PulseSequence()
                gate_sequence = self.get_sequence(gate, platform)
                for ch in gate_sequence.channels:
                    qubit = ch_to_qb[ch]
                    delay = qubit_clock(qubit) - channel_clock[ch]
                    if delay > 0:
                        delay_sequence.append((ch, Delay(duration=delay)))
                        channel_clock[ch] += delay
                    channel_duration = gate_sequence.channel_duration(ch)
                    channel_clock[ch] += channel_duration
                sequence.concatenate(delay_sequence)
                sequence.concatenate(gate_sequence)

                # register readout sequences to ``measurement_map`` so that we can
                # properly map acquisition results to measurement gates
                if isinstance(gate, gates.M):
                    measurement_map[gate] = gate_sequence

        return sequence, measurement_map
