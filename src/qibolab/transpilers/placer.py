import random

import networkx as nx
from qibo import gates
from qibo.config import raise_error
from qibo.models import Circuit

from qibolab.transpilers.abstract import Placer, Router, find_gates_qubits_pairs


class PlacementError(Exception):
    """Raise for an error in the qubit placement"""


def assert_placement(circuit: Circuit, layout: dict) -> bool:
    """Check if layout is correct and matches the number of qubits of the circuit.

    Args:
        circuit (qibo.models.Circuit): Circuit model to check.
        layout (dict): physical to logical qubit mapping.

    Raise PlacementError if the following conditions are not satisfied:
        - layout is written in the correct form.
        - layout matches the number of qubits in the circuit.
    """
    assert_mapping_consistency(layout)
    if circuit.nqubits > len(layout):
        raise PlacementError("Layout can't be used on circuit. The circuit requires more qubits.")
    if circuit.nqubits < len(layout):
        raise PlacementError("Layout can't be used on circuit. Ancillary extra qubits need to be added to the circuit.")


def assert_mapping_consistency(layout):
    """Check if layout is correct.

    Args:
        layout (dict): physical to logical qubit mapping.

    Raise PlacementError if layout is not written in the correct form.
    """
    values = sorted(layout.values())
    keys = list(layout)
    ref_keys = ["q" + str(i) for i in range(len(keys))]
    if keys != ref_keys:
        raise PlacementError("Some physical qubits in the layout may be missing or duplicated.")
    if values != list(range(len(values))):
        raise PlacementError("Some logical qubits in the layout may be missing or duplicated.")


class Trivial(Placer):
    """Place qubits according to the following simple notation: {'q0' : 0, 'q1' : 1, ..., 'qn' : n}.

    Attributes:
        connectivity (networkx.Graph): chip connectivity.
    """

    def __init__(self, connectivity: nx.Graph = None):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit):
        """Find the trivial placement for the circuit.

        Args:
            circuit (qibo.models.Circuit): circuit to be transpiled.

        Returns:
            (dict): physical to logical qubit mapping.
        """
        if self.connectivity is not None:
            if self.connectivity.number_of_nodes() != circuit.nqubits:
                raise PlacementError(
                    "The number of nodes of the connectivity graph must match the number of qubits in the circuit"
                )
        return dict(zip(list("q" + str(i) for i in range(circuit.nqubits)), range(circuit.nqubits)))


class Custom(Placer):
    """Define a custom initial qubit mapping.

    Attributes:
        map (list or dict): physical to logical qubit mapping,
        example [1,2,0] or {"q0":1, "q1":2, "q2":0} to assign the
        physical qubits 0;1;2 to the logical qubits 1;2;0 respectively.
        connectivity (networkx.Graph): chip connectivity.
    """

    def __init__(self, map, connectivity=None):
        self.connectivity = connectivity
        self.map = map

    def __call__(self, circuit=None):
        """Return the custom placement if it can be applied to the given circuit (if given).

        Args:
            circuit (qibo.models.Circuit): circuit to be transpiled.

        Returns:
            (dict): physical to logical qubit mapping.
        """
        if isinstance(self.map, dict):
            pass
        elif isinstance(self.map, list):
            self.map = dict(zip(list("q" + str(i) for i in range(len(self.map))), self.map))
        else:
            raise_error(TypeError, "Use dict or list to define mapping.")
        if circuit is not None:
            assert_placement(circuit, self.map)
        else:
            assert_mapping_consistency(self.map)
        return self.map


class Subgraph(Placer):
    """
    Subgraph isomorphism qubit placer, NP-complete it can take a long time
    for large circuits. This initialization method may fail for very short circuits.

    Attributes:
        connectivity (networkx.Graph): chip connectivity.
    """

    def __init__(self, connectivity):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit):
        """Find the initial layout of the given circuit using subgraph isomorphism.

        Args:
            circuit (qibo.models.Circuit): circuit to be transpiled.

        Returns:
            (dict): physical to logical qubit mapping.
        """
        gates_qubits_pairs = find_gates_qubits_pairs(circuit)
        if len(gates_qubits_pairs) < 3:
            raise_error(
                ValueError, "Circuit must contain at least two two qubit gates to implement subgraph placement."
            )
        circuit_subgraph = nx.Graph()
        circuit_subgraph.add_nodes_from(range(self.connectivity.number_of_nodes()))
        matcher = nx.algorithms.isomorphism.GraphMatcher(self.connectivity, circuit_subgraph)
        i = 0
        circuit_subgraph.add_edge(gates_qubits_pairs[i][0], gates_qubits_pairs[i][1])
        while matcher.subgraph_is_monomorphic() == True:
            result = matcher
            i += 1
            circuit_subgraph.add_edge(gates_qubits_pairs[i][0], gates_qubits_pairs[i][1])
            matcher = nx.algorithms.isomorphism.GraphMatcher(self.connectivity, circuit_subgraph)
            if (
                self.connectivity.number_of_edges() == circuit_subgraph.number_of_edges()
                or i == len(gates_qubits_pairs) - 1
            ):
                keys = list(result.mapping.keys())
                keys.sort()
                return {i: result.mapping[i] for i in keys}
        return dict(sorted(result.mapping.items()))


class Random(Placer):
    """
    Random initialization with greedy policy, let a maximum number of 2-qubit
    gates can be applied without introducing any SWAP gate.

    Attributes:
        connectivity (networkx.Graph): chip connectivity.
        samples (int): number of initial random layouts tested.
    """

    def __init__(self, connectivity, samples=100):
        self.connectivity = connectivity
        self.samples = samples

    def __call__(self, circuit):
        """Find an initial layout of the given circuit using random greedy algorithm.

        Args:
            circuit (qibo.models.Circuit): Circuit to be transpiled.

        Returns:
            (dict): physical to logical qubit mapping.
        """
        gates_qubits_pairs = find_gates_qubits_pairs(circuit)
        nodes = self.connectivity.number_of_nodes()
        keys = list(self.connectivity.nodes())
        final_mapping = dict(zip(keys, range(nodes)))
        final_graph = nx.relabel_nodes(self.connectivity, final_mapping)
        final_cost = self.cost(final_graph, gates_qubits_pairs)
        for _ in range(self.samples):
            mapping = dict(zip(keys, random.sample(range(nodes), nodes)))
            graph = nx.relabel_nodes(self.connectivity, mapping)
            cost = self.cost(graph, gates_qubits_pairs)
            if cost == 0:
                return mapping
            if cost < final_cost:
                final_graph = graph
                final_mapping = mapping
                final_cost = cost
        return final_mapping

    @staticmethod
    def cost(graph, gates_qubits_pairs):
        """
        Compute the cost associated to an initial layout as the lengh of the reduced circuit.

        Args:
            graph (networkx.Graph): current hardware qubit mapping.
            gates_qubits_pairs (list): circuit representation.

        Returns:
            (int): lengh of the reduced circuit.
        """
        for allowed, gate in enumerate(gates_qubits_pairs):
            if gate not in graph.edges():
                break
        return len(gates_qubits_pairs) - allowed


class ReverseTraversal(Placer):
    """
    Place qubits based on the algorithm proposed in https://doi.org/10.48550/arXiv.1809.02573.
    Works with ShortestPaths routing.

    Attributes:
        connectivity (networkx.Graph): chip connectivity.
        routing_algorithm (qibolab.transpilers.routing.Transpiler): routing algorithm.
        depth (int): number of two qubit gates considered before finding initial layout.
            if 'None' just one backward step will be implemented.
            If depth is greater than the number of two qubit gates in the circuit, the circuit will be routed more than once.
            Example: on a circuit with four two qubit gates A-B-C-D using depth = 6,
            the routing will be performed on the circuit C-D-D-C-B-A.
    """

    def __init__(self, connectivity: nx.Graph, routing_algorithm: Router, depth=None):
        self.connectivity = connectivity
        self.routing_algorithm = routing_algorithm
        self.depth = depth

    def __call__(self, circuit: Circuit):
        """Find the initial layout of the given circuit using Reverse Traversal placement.

        Args:
            circuit (qibo.models.Circuit): circuit to be transpiled.

        Returns:
            (dict): physical to logical qubit mapping.
        """

        initial_placer = Trivial(self.connectivity)
        initial_placement = initial_placer(circuit=circuit)
        new_circuit = self.assemble_circuit(circuit)
        final_placement = self.routing_step(initial_placement, new_circuit)
        return final_placement

    def assemble_circuit(self, circuit: Circuit):
        """Assemble a single circuit to apply Reverse Traversal placement based on depth.
        Example: for a circuit with four two qubit gates A-B-C-D using depth = 6,
        the function will return the circuit C-D-D-C-B-A.

        Args:
            circuit (qibo.models.Circuit): circuit to be transpiled.

        Returns:
            new_circuit (qibo.models.Circuit): assembled circuit to perform Reverse Traversal placement.
        """

        if self.depth is None:
            return circuit.invert()
        gates_qubits_pairs = find_gates_qubits_pairs(circuit)
        circuit_gates = len(gates_qubits_pairs)
        if circuit_gates == 0:
            raise ValueError("The circuit must contain at least a two qubit gate.")
        repetitions, remainder = divmod(self.depth, circuit_gates)
        assembled_gates_qubits_pairs = []
        for _ in range(repetitions):
            assembled_gates_qubits_pairs += gates_qubits_pairs[:]
            gates_qubits_pairs.reverse()
        assembled_gates_qubits_pairs += gates_qubits_pairs[0:remainder]
        new_circuit = Circuit(circuit.nqubits)
        for qubits in assembled_gates_qubits_pairs:
            # As only the connectivity is important here we can replace everything with CZ gates
            new_circuit.add(gates.CZ(qubits[0], qubits[1]))
        return new_circuit.invert()

    def routing_step(self, layout: dict, circuit: Circuit):
        """Perform routing of the circuit.

        Args:
            layout (dict): intial qubit layout.
            circuit (qibo.models.Circuit): circuit to be routed.
        """

        _, final_mapping = self.routing_algorithm(circuit, layout)
        return final_mapping