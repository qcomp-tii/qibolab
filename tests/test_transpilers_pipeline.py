import itertools

import numpy as np
import pytest
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibolab.native import NativeType
from qibolab.transpilers import Pipeline

from .test_transpilers_star_connectivity import transpose_qubits


def generate_random_circuit(nqubits, ngates, seed=None):
    """Generate random circuits one-qubit rotations and CZ gates."""
    pairs = list(itertools.combinations(range(nqubits), 2))
    if seed is not None:  # pragma: no cover
        np.random.seed(seed)

    one_qubit_gates = [gates.RX, gates.RY, gates.RZ, gates.X, gates.Y, gates.Z, gates.H]
    two_qubit_gates = [
        gates.CNOT,
        gates.CZ,
        gates.SWAP,
        gates.iSWAP,
        gates.CRX,
        gates.CRY,
        gates.CRZ,
    ]
    n1, n2 = len(one_qubit_gates), len(two_qubit_gates)
    n = n1 + n2 if nqubits > 1 else n1
    circuit = Circuit(nqubits)
    for _ in range(ngates):
        igate = int(np.random.randint(0, n))
        if igate >= n1:
            q = tuple(np.random.randint(0, nqubits, 2))
            while q[0] == q[1]:
                q = tuple(np.random.randint(0, nqubits, 2))
            gate = two_qubit_gates[igate - n1]
        else:
            q = (np.random.randint(0, nqubits),)
            gate = one_qubit_gates[igate]
        if issubclass(gate, gates.ParametrizedGate):
            theta = 2 * np.pi * np.random.random()
            circuit.add(gate(*q, theta=theta))
        else:
            circuit.add(gate(*q))
    return circuit


@pytest.mark.parametrize(
    "two_qubit_natives",
    [NativeType.CZ, NativeType.iSWAP, NativeType.CZ | NativeType.iSWAP],
)
@pytest.mark.parametrize("middle_qubit", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("nqubits", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("ngates", [10, 40])
@pytest.mark.parametrize("fuse_one_qubit", [False, True])
def test_transpile(middle_qubit, nqubits, ngates, fuse_one_qubit, two_qubit_natives):
    backend = NumpyBackend()
    # find the number of qubits for hardware circuit
    if nqubits == 1:
        hardware_qubits = 1
    else:
        hardware_qubits = max(nqubits, middle_qubit + 1)

    circuit = generate_random_circuit(hardware_qubits, ngates)
    transpiler = Pipeline.default(
        two_qubit_natives=two_qubit_natives,
        middle_qubit=middle_qubit,
        fuse_one_qubit=fuse_one_qubit,
    )
    transpiled_circuit, hardware_qubits = transpiler.transpile(circuit)
    assert transpiler.is_satisfied(transpiled_circuit)

    final_state = backend.execute_circuit(transpiled_circuit).state()
    target_state = backend.execute_circuit(circuit).state()
    target_state = transpose_qubits(target_state, hardware_qubits)
    fidelity = np.abs(np.conj(target_state).dot(final_state))
    np.testing.assert_allclose(fidelity, 1.0)


def test_can_execute_false():
    transpiler = Pipeline.default(two_qubit_natives=NativeType.CZ | NativeType.iSWAP)
    circuit1 = Circuit(1)
    circuit1.add(gates.H(0))
    assert not transpiler.is_satisfied(circuit1)
    circuit2 = Circuit(2)
    circuit2.add(gates.CNOT(0, 1))
    assert not transpiler.is_satisfied(circuit2)
    circuit3 = Circuit(3)
    circuit3.add(gates.TOFFOLI(0, 1, 2))
    assert not transpiler.is_satisfied(circuit3)