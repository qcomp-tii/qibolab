from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
import numpy.typing as npt

ExecRes = np.dtype([("i", np.float64), ("q", np.float64)])


@dataclass
class ExecutionResults:
    """Data structure to deal with the output of :func:`qibolab.platforms.abstract.AbstractPlatform.execute_pulse_sequence`"""

    array: npt.NDArray[ExecRes]
    shots: Optional[npt.NDArray[np.uint32]] = None

    @classmethod
    def from_components(cls, is_, qs_, shots=None):
        ar = np.empty(is_.shape, dtype=ExecRes)
        ar["i"] = is_
        ar["q"] = qs_
        ar = np.rec.array(ar)
        return cls(ar, shots)

    @property
    def i(self):
        return self.array.i

    @property
    def q(self):
        return self.array.q

    def __add__(self, data):
        i = np.append(self.i, data.i, axis=0)
        q = np.append(self.q, data.q, axis=0)

        new_execution_results = self.__class__.from_components(i, q)

        return new_execution_results

    @cached_property
    def measurement(self):
        """Resonator signal voltage mesurement (MSR) in volts."""
        return np.sqrt(self.i**2 + self.q**2)

    @cached_property
    def phase(self):
        """Computes phase value."""
        phase = np.angle(self.i + 1.0j * self.q)
        return phase
        # return signal.detrend(np.unwrap(phase))

    @cached_property
    def ground_state_probability(self):
        """Computes ground state probability"""
        return 1 - np.mean(self.shots)

    def to_dict_probability(self, state=1):
        """Serialize probabilities in dict.
        Args:
            state (int): if 0 stores the probabilities of finding
                        the ground state. If 1 stores the
                        probabilities of finding the excited state.
        """
        if state == 1:
            return {"probability": 1 - self.ground_state_probability}
        elif state == 0:
            return {"probability": self.ground_state_probability}

    @property
    def average(self):
        """Perform average over i and q"""
        return AveragedResults.from_components(self.i.mean(), self.q.mean())

    def to_dict(self):
        """Serialize output in dict."""

        return {
            "MSR[V]": self.measurement,
            "i[V]": float(self.i),  # float to convert from array to float type
            "q[V]": float(self.q),  # float to convert from array to float type
            "phase[rad]": self.phase,
        }

    def __len__(self):
        assert len(self.i) == len(self.q)
        return len(self.i)


@dataclass
class AveragedResults(ExecutionResults):
    """Data structure containing averages of ``ExecutionResults``."""
