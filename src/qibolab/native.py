from typing import Annotated, Optional

import numpy as np

from .pulses import Drag, Gaussian, Pulse, PulseSequence
from .serialize_ import Model, replace


def _normalize_angles(theta, phi):
    """Normalize theta to (-pi, pi], and phi to [0, 2*pi)."""
    theta = theta % (2 * np.pi)
    theta = theta - 2 * np.pi * (theta > np.pi)
    phi = phi % (2 * np.pi)
    return theta, phi


class RxyFactory(PulseSequence):
    """Factory for pulse sequences that generate single-qubit rotations around
    an axis in xy plane.

    It is assumed that the underlying sequence contains only a single pulse.
    It is assumed that the base sequence corresponds to a calibrated pi rotation around X axis.
    Other rotation angles are achieved by scaling the amplitude, assuming a linear transfer function.

    Args:
        sequence: The base sequence for the factory.
    """

    @classmethod
    def validate(cls, value):
        sequence = PulseSequence(value)
        if len(sequence.channels) != 1:
            raise ValueError(
                f"Incompatible number of channels: {len(sequence.channels)}. "
                f"{cls} expects a sequence on exactly one channel."
            )

        if len(sequence) != 1:
            raise ValueError(
                f"Incompatible number of pulses: {len(sequence)}. "
                f"{cls} expects a sequence with exactly one pulse."
            )

        pulse = sequence[0][1]
        assert isinstance(pulse, Pulse)
        expected_envelopes = (Gaussian, Drag)
        if not isinstance(pulse.envelope, expected_envelopes):
            raise ValueError(
                f"Incompatible pulse envelope: {pulse.envelope.__class__}. "
                f"{cls} expects {expected_envelopes} envelope."
            )

        return cls(value)

    def create_sequence(self, theta: float = np.pi, phi: float = 0.0) -> PulseSequence:
        """Create a sequence for single-qubit rotation.

        Args:
            theta: the angle of rotation.
            phi: the angle that rotation axis forms with x axis.
        """
        theta, phi = _normalize_angles(theta, phi)
        ch, pulse = self[0]
        assert isinstance(pulse, Pulse)
        new_amplitude = pulse.amplitude * theta / np.pi
        return PulseSequence(
            [(ch, replace(pulse, amplitude=new_amplitude, relative_phase=phi))]
        )


class FixedSequenceFactory(PulseSequence):
    """Simple factory for a fixed arbitrary sequence."""

    def create_sequence(self) -> PulseSequence:
        return self.copy()


class SingleQubitNatives(Model):
    """Container with the native single-qubit gates acting on a specific
    qubit."""

    RX: Optional[RxyFactory] = None
    """Pulse to drive the qubit from state 0 to state 1."""
    RX12: Optional[FixedSequenceFactory] = None
    """Pulse to drive to qubit from state 1 to state 2."""
    MZ: Optional[FixedSequenceFactory] = None
    """Measurement pulse."""
    CP: Optional[FixedSequenceFactory] = None
    """Pulse to activate coupler."""


class TwoQubitNatives(Model):
    """Container with the native two-qubit gates acting on a specific pair of
    qubits."""

    CZ: Annotated[Optional[FixedSequenceFactory], {"symmetric": True}] = None
    CNOT: Annotated[Optional[FixedSequenceFactory], {"symmetric": True}] = None
    iSWAP: Annotated[Optional[FixedSequenceFactory], {"symmetric": True}] = None

    @property
    def symmetric(self):
        """Check if the defined two-qubit gates are symmetric between target
        and control qubits."""
        return all(
            info.metadata[0]["symmetric"] or getattr(self, fld) is None
            for fld, info in self.model_fields.items()
        )
