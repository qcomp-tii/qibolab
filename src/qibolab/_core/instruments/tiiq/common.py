from dataclasses import dataclass, asdict

__all__ = [
    "Unpackable",
    "OutputId",
    "InputId",
    "PortId",
    "get_rf_output_from_port_id",
    "get_dc_output_from_port_id",
    "get_rf_input_from_port_id",
    "Modes",
    "modes",
    "DEMO",
    "DEBUG",
    "SIMULATION",
    "debug_print",
]


@dataclass
class Unpackable:
    def keys(self) -> list[int]:
        return asdict(self).keys()
    def __getitem__(self, key) -> int:
        return asdict(self)[key]


InputId = OutputId = str
"""Unique identifier for Outputs (DACs) and Inputs (ADCs).
   'tile#_block#' example: '20'
"""

PortId = str
"""Unique identifier for a physical port.
    RFO_n for drive and probe channels
    DCO_m:RFO_n for flux channels
    RFI_n for acquisition channels
"""

def _parse_port_id(element, port_id):
    if not element in ["DCO", "RFO", "RFI"]:
        raise ValueError(f"port name {port_id} is not valid.")
    import re
    
    match = re.search(f'{element}_(\\d+)', port_id)
    if match:
        return match.group(1)
    else:
        return None

def get_rf_output_from_port_id(port_id:PortId, require_valid: bool = True) -> int|None:
    rf_output = _parse_port_id("RFO", port_id)
    if require_valid and rf_output is None:
        raise ValueError(f"port id {port_id} does not contain a valid RF output port.")
    return rf_output

def get_dc_output_from_port_id(port_id:PortId) -> int:
    dc_output = _parse_port_id("DCO", port_id)
    if dc_output is None:
        raise ValueError(f"port id {port_id} does not contain a valid DC output port.")
    return dc_output

def get_rf_input_from_port_id(port_id:PortId) -> int:
    rf_input = _parse_port_id("RFI", port_id)
    if rf_input is None:
        raise ValueError(f"port id {port_id} does not contain a valid RF input port.")
    return rf_input


class Modes:
    def __init__(self):
        # Initialize a set to hold active modes
        self._active_modes = set()

    def enable(self, mode):
        """Enable a mode."""
        self._active_modes.add(mode)

    def disable(self, mode):
        """Disable a mode."""
        self._active_modes.discard(mode)

    def is_enabled(self, mode):
        """Check if a mode is active."""
        return mode in self._active_modes

    def active_modes(self):
        """Get all active modes."""
        return list(self._active_modes)
    

DEMO = "DEMO"
DEBUG = "DEBUG"
SIMULATION = "SIMULATION"
modes = Modes()


def debug_print(*values: object):
    if modes.is_enabled(DEBUG):
        print(*values)