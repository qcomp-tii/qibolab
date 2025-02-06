from dataclasses import dataclass, field
from typing import Union

from qibolab._core.identifier import ChannelId
from qibolab._core.instruments.tiiq.common import (
    OutputId,
    InputId,
)
from qibolab._core.pulses import Align, Pulse, Delay, VirtualZ, Acquisition, Readout, PulseId, PulseLike

__all__ = [
    "Scheduled",
    "ScheduledPulse",
    "ScheduledAcquisition",
    "ScheduledReadout",
    "ScheduledDelay",
    "ScheduledItem",
    "ScheduledMultiplexedAcquisitions",
    "ScheduledMultiplexedPulses",
    "ScheduledMultiplexedItem",
    "ScheduledPulseSequence",
    ]


class Mutable():
    channel_id: ChannelId
    _original: PulseLike
    attr_names: list[str]
    id: PulseId

    def __init__(self, channel_id: ChannelId, pulse_like: PulseLike):
        self.channel_id = channel_id
        self._original = pulse_like
        self.attr_names = ['id']

    def __getattr__(self, name):
        # check if the attribute exists
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self._original, name)
    def __setattr__(self, name, value):
        # overrides _original attributes
        self.__dict__[name] = value

    def __repr__(self):
        attrs = ", ".join(f"{name}={getattr(self, name, None)!r}" for name in self.attr_names)
        return f"{self.__class__.__name__}({attrs})"

    def copy(self):
        """Return a copy of the mutable item."""
        new_item = self.__class__(self.channel_id, self._original)
        for k in self.attr_names:
            v = getattr(self, k, None)
            setattr(new_item, k, v)
        return new_item


class MutablePulse(Mutable):
    def __init__(self, channel_id: ChannelId, pulse_like: PulseLike):
        assert isinstance(pulse_like, Pulse)
        super().__init__(channel_id, pulse_like)
        self.attr_names = ['channel_id', 'duration', 'amplitude', 'envelope', 'relative_phase', 'id']


class MutableAcquisition(Mutable):
    def __init__(self, channel_id: ChannelId, pulse_like: PulseLike):
        assert isinstance(pulse_like, Acquisition)
        super().__init__(channel_id, pulse_like)
        self.attr_names = ['channel_id', 'duration', 'id']


class MutableReadout(Mutable):
    def __init__(self, channel_id: ChannelId, pulse_like: PulseLike):
        assert isinstance(pulse_like, Readout)
        super().__init__(channel_id, pulse_like)
        self.attr_names = ['channel_id', 'duration', 'amplitude', 'envelope', 'relative_phase', 'id']


class MutableDelay(Mutable):
    def __init__(self, channel_id: ChannelId, pulse_like: PulseLike):
        assert isinstance(pulse_like, Delay)
        super().__init__(channel_id, pulse_like)
        self.attr_names = ['channel_id', 'duration', 'id']


class MutableVirtualZ(Mutable):
    def __init__(self, channel_id: ChannelId, pulse_like: PulseLike):
        assert isinstance(pulse_like, VirtualZ)
        super().__init__(channel_id, pulse_like)
        self.attr_names = ['channel_id', 'phase', 'id']


MutableItem = Union[MutablePulse, MutableAcquisition, MutableReadout, MutableDelay, MutableVirtualZ]


class MutablePulseSequence():
    _data: list[MutableItem]
    _id_map: dict[PulseId, MutableItem]

    @property
    def channels(self) -> set[ChannelId]:
        """Channels involved in the sequence."""
        _channels = set()
        for item in self._data:
            if not isinstance(item, MutableItem):
                raise TypeError(f"Item {item} is not an instance of MutableItem type.")
            else:
                _channels |= {item.channel_id}
        return _channels

    def __init__(self):
        self._data: list[MutableItem] = []
        self._id_map: dict[PulseId, MutableItem] = {}


    def add_item(self, item: MutableItem):
        """
        Add a mutable item to the mutable pulse sequence.
        """
        if not isinstance(item, (MutableItem)):
            raise TypeError(f"Item {item} is not an instance of MutableItem type.")
        
        if isinstance(item, MutableItem):
            self._data.append(item)
            self._id_map[item.id] = item

    def get_item(self, item_id: PulseId) -> MutableItem | None:
        """
        Retrieve an item by its ID.

        :param item_id: The unique ID of the item to retrieve.
        :return: The corresponding MutableItem if found, None otherwise.
        """
        return self._id_map.get(item_id)

    def remove_item(self, item: MutableItem):
        self._data.remove(self._id_map.pop(item.id))
    
    def get_channel_items(self, channel_id: ChannelId) -> list[MutableItem]:
        """
        Retrieve all items associated with a given channel.

        :param channel_id: The unique ID of the channel.
        :return: A list of MutableItem objects associated with the channel.
        """
        return [item for item in self._data if item.channel_id == channel_id]

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, index: int) -> MutableItem:
        """Get an item by index."""
        return self._data[index]

    def __len__(self) -> int:
        """Return the length of the scheduled pulse sequence."""
        return len(self._data)

    def __repr__(self) -> str:
        """Return a string representation of the scheduled pulse sequence."""
        attr_names = ['channels']
        attrs = ", ".join(f"{name}={getattr(self, name, None)!r}" for name in attr_names)

        items = "\n\t".join(repr(item) for item in self._data)
        return f"{self.__class__.__name__}({attrs})\n\t{items}"

    def copy(self):
        """Return a copy of the mutable pulse sequence."""
        new_sequence = MutablePulseSequence()
        for item in self._data:
            new_item: Mutable = item.copy()
            new_sequence.add_item(new_item)
        return new_sequence

class Scheduled(Mutable):
    start: float
    lag: float|None
    duration: float

    def __init__(self, channel_id: ChannelId, start:float, item: MutableItem):
        pulse_like: PulseLike = item._original
        super().__init__(channel_id, pulse_like)
        for k in item.attr_names:
            v = getattr(item, k, None)
            setattr(self, k, v)
        self.attr_names = item.attr_names + ['start', 'lag']
        self.start = start
        self.lag = None


class ScheduledPulse(Scheduled):
    output: OutputId
    is_multiplexed: bool

    def __init__(self, channel_id: ChannelId, start:float, item: MutableItem):
        assert isinstance(item._original, Pulse)
        super().__init__(channel_id, start, item)
        self.attr_names = ['channel_id', 'start', 'lag', 'duration', 'amplitude', 'envelope', 'relative_phase', 'output', 'is_multiplexed', 'id']
        self.output = None
        self.is_multiplexed = False


class ScheduledAcquisition(Scheduled):
    input: InputId
    is_multiplexed: bool

    def __init__(self, channel_id: ChannelId, start:float, item: MutableItem):
        assert isinstance(item._original, Acquisition)
        super().__init__(channel_id, start, item)
        self.attr_names = ['channel_id', 'start', 'lag', 'duration', 'input', 'is_multiplexed', 'id']
        self.input = None
        self.is_multiplexed = False


class ScheduledReadout(Scheduled):
    output: InputId
    input: InputId
    is_multiplexed: bool
    probe: Pulse
    acquisition: Acquisition

    def __init__(self, channel_id: ChannelId, start:float, item: MutableItem):
        assert isinstance(item._original, Readout)
        super().__init__(channel_id, start, item)
        self.attr_names = ['channel_id', 'start', 'lag', 'duration', 'amplitude', 'envelope', 'relative_phase', 'ouput', 'input', 'is_multiplexed', 'id']
        self.output = None
        self.input = None
        self.is_multiplexed = False


class ScheduledDelay(Scheduled):
    def __init__(self, channel_id: ChannelId, start:float, item: MutableItem):
        assert isinstance(item._original, Delay)
        super().__init__(channel_id, start, item)
        self.attr_names = ['channel_id', 'start', 'lag', 'duration', 'id']


class ScheduledVirtualZ(Scheduled):
    def __init__(self, channel_id: ChannelId, start:float, item: MutableItem):
        assert isinstance(item._original, VirtualZ)
        super().__init__(channel_id, start, item)
        self.attr_names = ['channel_id', 'start', 'lag', 'duration', 'id']


ScheduledItem = Union[ScheduledPulse, ScheduledAcquisition, ScheduledReadout, ScheduledDelay, ScheduledVirtualZ]


@dataclass
class ScheduledMultiplexedAcquisitions():
    input: InputId
    acquisitions: list[ScheduledAcquisition] = field(default_factory=list, init=False)
    lag: float = 0.0

    @property
    def id(self) -> PulseId:
        """Identifier."""
        return id(self)

    @property
    def start(self) -> float|None:
        acquisition: ScheduledAcquisition
        starts: list[float] = []
        for acquisition in self.acquisitions:
            starts.append(acquisition.start)
        return min(starts) if len(starts) > 0 else None

    @property
    def duration(self) -> float|None:
        acquisition: ScheduledAcquisition
        starts: list[float] = []
        finishes: list[float] = []
        for acquisition in self.acquisitions:
            starts.append(acquisition.start)
            finishes.append(acquisition.start + acquisition.duration)
        return (max(finishes) - min(starts)) if len(starts) > 0 else None
    
    @property
    def channel_ids(self) -> list[ChannelId]:
        return [acquisition.channel_id for acquisition in self.acquisitions]
    
    def __repr__(self) -> str:
        """Return a string representation of the scheduled multiplexed acquisitions."""
        attr_names = ['channel_ids', 'start', 'lag', 'duration', 'input', 'id']
        attrs = ", ".join(f"{name}={getattr(self, name, None)!r}" for name in attr_names)

        items = "\n\t\t".join(repr(item) for item in self.acquisitions)
        return f"{self.__class__.__name__}({attrs})\n\t\t{items}"
    
    def append(self, acquisition: ScheduledAcquisition):
        """
        Add an individual acquisition.
        """
        if not isinstance(acquisition, ScheduledAcquisition):
            raise TypeError(f"Item {acquisition} is not an instance of ScheduledAcquisition.")
        self.acquisitions.append(acquisition)

    # TODO add custom repr to show start and duration

@dataclass
class ScheduledMultiplexedPulses():
    output: OutputId
    pulses: list[ScheduledPulse] = field(default_factory=list, init=False)
    lag: float = 0.0

    @property
    def id(self) -> PulseId:
        """Identifier."""
        return id(self)

    @property
    def start(self) -> float:
        pulse: ScheduledPulse
        starts: list[float] = []
        for pulse in self.pulses:
            starts.append(pulse.start)
        return min(starts) if len(starts) > 0 else None

    @property
    def duration(self) -> float:
        pulse: ScheduledPulse
        starts: list[float] = []
        finishes: list[float] = []
        for pulse in self.pulses:
            starts.append(pulse.start)
            finishes.append(pulse.start + pulse.duration)
        return (max(finishes) - min(starts)) if len(starts) > 0 else None
    
    @property
    def channel_ids(self) -> list[ChannelId]:
        return [pulse.channel_id for pulse in self.pulses]
        
    def __repr__(self) -> str:
        """Return a string representation of the scheduled multiplexed pulses."""
        attr_names = ['channel_ids', 'start', 'lag', 'duration', 'output', 'id']
        attrs = ", ".join(f"{name}={getattr(self, name, None)!r}" for name in attr_names)

        items = "\n\t\t".join(repr(item) for item in self.pulses)
        return f"{self.__class__.__name__}({attrs})\n\t\t{items}"
    
    def append(self, pulse: ScheduledPulse):
        """
        Add an individual pulse.
        """
        if not isinstance(pulse, ScheduledPulse):
            raise TypeError(f"Item {pulse} is not an instance of ScheduledPulse.")
        self.pulses.append(pulse)


ScheduledMultiplexedItem = Union[ScheduledMultiplexedAcquisitions, ScheduledMultiplexedPulses]


class ScheduledPulseSequence():
    _data: list[ScheduledItem|ScheduledMultiplexedItem]
    _id_map: dict[PulseId, ScheduledItem]
    multiplexed_are_grouped: bool = False

    @property
    def channels(self) -> set[ChannelId]:
        """Channels involved in the sequence."""
        _channels = set()
        for item in self._data:
            if isinstance(item, (ScheduledPulse, ScheduledAcquisition, ScheduledReadout)):
                _channels |= {item.channel_id}
            elif isinstance(item, (ScheduledMultiplexedAcquisitions, ScheduledMultiplexedPulses)):
                _channels |= set(item.channel_ids)
        return _channels

    @property
    def start(self) -> float:
        """Start time of the first item of the sequence."""
        item: ScheduledItem|ScheduledMultiplexedItem
        starts: list[float] = []
        for item in self._data:
            starts.append(item.start)
        return min(starts, default=0.0)
    
    @property
    def duration(self) -> float:
        """Duration of the entire sequence."""
        return self.finish - self.start
    
    @property
    def finish(self) -> float:
        """Finish time of the last item of the sequence."""
        item: ScheduledItem|ScheduledMultiplexedItem
        finishes: list[float] = []
        for item in self._data:
            finishes.append(item.start + item.duration)
        return max(finishes, default=0.0)

    def __init__(self):
        self._data: list[ScheduledItem|ScheduledMultiplexedItem] = []
        self._id_map: dict[PulseId, ScheduledItem] = {}

    # def schedule_item(self, start: float, port: OutputId|InputId, item: PulseLike):
    #     """
    #     Schedule an item and add it to the pulse sequence.
    #     """
    #     if isinstance(item, Pulse):
    #         scheduled_item = ScheduledPulse(start=start, channel_id=channel_id, output=port, pulse=item, is_multiplexed=is_multiplexed)
    #     elif isinstance(item, Acquisition):
    #         scheduled_item = ScheduledAcquisition(start=start, channel_id=channel_id, input=port, acquisition=item, is_multiplexed=is_multiplexed)
    #     else:
    #         raise TypeError(f"Item {item} is not an instance of Pulse or Acquisition.")
    #     self.add_item(scheduled_item)

    def sort(self):
        self._data.sort(key=lambda x: x.start)

    
    def add_item(self, item: ScheduledItem|ScheduledMultiplexedItem):
        """
        Add an item to the scheduled pulse sequence and sort all items by their start property.
        """
        if self.multiplexed_are_grouped:
            raise RuntimeError("No more items can be scheduled once multiplexed items are grouped.")
        if not isinstance(item, (ScheduledItem,ScheduledMultiplexedItem)):
            raise TypeError(f"Item {item} is not an instance of ScheduledItem type.")
        
        if isinstance(item, ScheduledItem):
            self._data.append(item)
            self._id_map[item.id] = item
        elif isinstance(item, ScheduledMultiplexedPulses):
            self._data.append(item)
            for pulse in item.pulses:
                self._id_map[pulse.id] = pulse
        elif isinstance(item, ScheduledMultiplexedAcquisitions):
            self._data.append(item)
            for acquisition in item.acquisitions:
                self._id_map[acquisition.id] = acquisition
        self.sort()


    def get_item(self, item_id: PulseId) -> ScheduledItem | None:
        """
        Retrieve an item by its ID.

        :param item_id: The unique ID of the item to retrieve.
        :return: The corresponding ScheduledItem if found, None otherwise.
        """
        return self._id_map.get(item_id)

    def remove_item(self, item: ScheduledItem):
        self._data.remove(self._id_map.pop(item.id))

    def get_acquisitions(self) -> list[ScheduledAcquisition|ScheduledMultiplexedAcquisitions]: 
        return [
            item
            for item in self._data 
            if isinstance(item, (ScheduledAcquisition, ScheduledMultiplexedAcquisitions))
        ]
    
    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, index: int) -> ScheduledItem|ScheduledMultiplexedItem:
        """Get an item by index."""
        return self._data[index]

    def __len__(self) -> int:
        """Return the length of the scheduled pulse sequence."""
        return len(self._data)

    def __repr__(self) -> str:
        """Return a string representation of the scheduled pulse sequence."""
        attr_names = ['channels', 'start', 'duration', 'finish']
        attrs = ", ".join(f"{name}={getattr(self, name, None)!r}" for name in attr_names)

        items = "\n\t".join(repr(item) for item in self._data)
        return f"{self.__class__.__name__}({attrs})\n\t{items}"

    # def group_multiplexed(self):
    #     """Group acquisitions into scheduled multiplexed acquisitions.
    #     """
    #     muxed_acquisition_group_start = 0.0
    #     muxed_acquisition_group_finish = 0.0
    #     is_first_acquisition = True
    #     muxed_pulse_group_start = 0.0
    #     muxed_pulse_group_finish = 0.0
    #     is_first_pulse = True

    #     new_data: list[ScheduledMultiplexedPulses|ScheduledMultiplexedAcquisitions] = []
    #     for item in self._data:
    #         if isinstance(item, (ScheduledMultiplexedPulses, ScheduledMultiplexedAcquisitions)):
    #             raise ValueError("")
    #         if item.is_multiplexed:
    #             if isinstance(item, ScheduledPulse):
    #                 scheduled_pulse: ScheduledPulse = item
    #                 if is_first_pulse or scheduled_pulse.start > muxed_pulse_group_finish:
    #                     multiplexed_pulse = ScheduledMultiplexedPulses(scheduled_pulse.output)
    #                     multiplexed_pulse.append(scheduled_pulse)
    #                     new_data.append(multiplexed_pulse)
    #                     muxed_pulse_group_start = scheduled_pulse.start
    #                     muxed_pulse_group_finish = scheduled_pulse.start + scheduled_pulse.duration
    #                     is_first_pulse = False
    #                 else:
    #                     if scheduled_pulse.start >= muxed_pulse_group_start and scheduled_pulse.start < muxed_pulse_group_finish:
    #                         if scheduled_pulse.start + scheduled_pulse.duration > muxed_pulse_group_finish:
    #                             muxed_pulse_group_finish = scheduled_pulse.start + scheduled_pulse.duration
    #                         multiplexed_pulse.append(scheduled_pulse)
    #             elif isinstance(item, ScheduledAcquisition):
    #                 scheduled_acquisition: ScheduledAcquisition = item
    #                 if is_first_acquisition or scheduled_acquisition.start > muxed_acquisition_group_finish:
    #                     multiplexed_acquisition = ScheduledMultiplexedAcquisitions(scheduled_acquisition.input)
    #                     multiplexed_acquisition.append(scheduled_acquisition)
    #                     new_data.append(multiplexed_acquisition)
    #                     muxed_acquisition_group_start = scheduled_acquisition.start
    #                     muxed_acquisition_group_finish = scheduled_acquisition.start + scheduled_acquisition.duration
    #                     is_first_acquisition = False
    #                 else:
    #                     if scheduled_acquisition.start >= muxed_acquisition_group_start and scheduled_acquisition.start < muxed_acquisition_group_finish:
    #                         if scheduled_acquisition.start + scheduled_acquisition.duration > muxed_acquisition_group_finish:
    #                             muxed_acquisition_group_finish = scheduled_acquisition.start + scheduled_acquisition.duration
    #                         multiplexed_acquisition.append(scheduled_acquisition)
    #         else:
    #             new_data.append(item)

    #     new_data.sort(key=lambda x: x.start) # it shouldn't be necessary
    #     self._data = new_data
    #     self.multiplexed_are_grouped = True

    #     # TODO group by DAC / ADC (to support more than one multiplexed generator & processor)
    

