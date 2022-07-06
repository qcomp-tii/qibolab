import os
from qibo.backends.numpy import NumpyBackend
from qibo.config import raise_error


class QibolabBackend(NumpyBackend):

    def __init__(self, platform, runcard=None):
        from qibolab.platform import Platform
        super().__init__()
        self.name = "qibolab"
        self.platform = Platform(platform, runcard)

    def asmatrix(self, gate): # pragma: no cover
        raise_error(NotImplementedError, "Matrices not available for qibolab backend.")

    def asmatrix_parametrized(self, gate): # pragma: no cover
        raise_error(NotImplementedError, "Matrices not available for qibolab backend.")

    def asmatrix_fused(self, gate): # pragma: no cover
        raise_error(NotImplementedError, "Matrices not available for qibolab backend.")

    def apply_gate(self, gate, state, nqubits): # pragma: no cover
        raise_error(NotImplementedError)

    def apply_gate_density_matrix(self, gate, state, nqubits): # pragma: no cover
        raise_error(NotImplementedError, "Density matrices")

    def execute_circuit(self, circuit, initial_state=None, nshots=None): # pragma: no cover
        """Executes a quantum circuit.

        Args:
            circuit (:class:`qibo.core.circuit.Circuit`): Circuit to execute.
            nshots (int): Number of shots to sample from the experiment.
                If ``None`` the default value provided as hardware_avg in the
                calibration yml will be used.

        Returns:
            Readout results acquired by after execution.
        """
        from qibolab.pulses import PulseSequence
        if initial_state is not None:
            raise_error(ValueError, "Hardware backend does not support "
                                    "initial state in circuits.")

        # Translate gates to pulses and create a ``PulseSequence``
        if circuit.measurement_gate is None:
            raise_error(RuntimeError, "No measurement register assigned.")

        sequence = PulseSequence()
        for gate in circuit.queue:
            self.platform.to_sequence(sequence, gate)
        self.platform.to_sequence(sequence, circuit.measurement_gate)

        # Execute the pulse sequence on the platform
        self.platform.connect()
        self.platform.setup()
        self.platform.start()
        readout = self.platform(sequence, nshots)
        self.platform.stop()

        return CircuitResult(self, circuit, readout, nshots)

    def get_state_tensor(self):
        raise_error(NotImplementedError, "Qibolab cannot return state vector.")

    def get_state_repr(self, result): # pragma: no cover
        return result.execution_result
