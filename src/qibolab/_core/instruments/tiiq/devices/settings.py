import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, Literal

from qibolab._core.components import (
    AcquisitionChannel,
    Config,
    Channel,
    DcChannel,
    IqChannel,
    IqConfig,
    OscillatorConfig,
)
from qibolab._core.identifier import ChannelId
from qibolab._core.instruments.tiiq.common import (
    PortId,
    OutputId,
    InputId,
    get_rf_output_from_port_id,
    get_dc_output_from_port_id,
    get_rf_input_from_port_id,
)
from qibolab._core.instruments.tiiq.components.components import DriveComponent, FluxComponent, ProbeComponent, AcquisitionComponent
from qibolab._core.instruments.tiiq.components.configs import TIIqDriveConfig, TIIqFluxConfig, TIIqProbeConfig, TIIqAcquisitionConfig
from qibolab._core.instruments.tiiq.scheduled import (
    ScheduledPulse,
    ScheduledMultiplexedPulses,
    ScheduledAcquisition,
    ScheduledMultiplexedAcquisitions,
    ScheduledPulseSequence
)
from qibolab._core.pulses import Envelope, IqWaveform, Waveform, Acquisition, Align, Delay, Pulse, Readout, PulseLike, PulseId

from qick.qick_asm import QickConfig

__all__ = [
    "TIIqSettings",
    "FullSpeedSignalGenerator",
    "InterpolatedSignalGenerator",
    "MuxedSignalGenerator",
    "MuxedSignalProcessorTone",
    "BiassingChannel",
    "SignalGeneratorsType",
    "SignalProcessorsType",
    "BiassingChannelsType",
    ]

MASK = [0, 1, 2, 3, 4] 
# TODO check if this can be determined from firmware configuration
# keep the mask fixedso that power levels don't change when channels are added or removed


@dataclass
class FullSpeedSignalGenerator():
    ch: int 
    cfg: dict
    # TODO check if it is possible we can do without ch, and work just with the output str
    # if so, we need to change the port names in platform.py

    frequency: Optional[float] = 0.0
    # pulses: list[PulseLike] = field(default_factory=list)
    # waveforms: dict[str, IqWaveform] = field(default_factory=list)


@dataclass
class InterpolatedSignalGenerator():
    ch: int
    cfg: dict

    frequency: Optional[float|None] = None
    digital_mixer_frequency: Optional[float|None] = None

    # pulses: list[PulseLike] = field(default_factory=list)
    # waveforms: dict[str, IqWaveform] = field(default_factory=list)

DEFAULT_FREQUENCY: float = 7000000000.0
DEFAULT_GAIN: float = 0.0
DEFAULT_PHASE: float = 0.0

@dataclass
class MuxedSignalGeneratorTone():
    frequency: Optional[float] = DEFAULT_FREQUENCY
    gain: Optional[float] = DEFAULT_GAIN
    phase: Optional[float] = DEFAULT_PHASE


@dataclass
class MuxedSignalGenerator():
    ch: int
    cfg: dict

    digital_mixer_frequency: Optional[float|None] = None # TODO: try to remove and auto calculate it
    ro_ch: Optional[int|None] = None
    mask: Optional[list[int]|None] = field(default_factory=list)
    tones: defaultdict[ChannelId, MuxedSignalGeneratorTone] = field(default_factory=lambda: defaultdict(MuxedSignalGeneratorTone))
    
    # pulses: list[PulseLike] = field(default_factory=list)
    # waveforms: dict[str, IqWaveform] = field(default_factory=list)

    def __post_init__(self):
        self.mask = MASK

    def get_mux_freqs(self, include_default:bool = True):
        tone: MuxedSignalGeneratorTone
        values = [tone.frequency for tone in self.tones.values()]
        if include_default:
            values += [DEFAULT_FREQUENCY] * (len(self.mask) - len(values))
        return  values
    
    def get_mux_gains(self, include_default:bool = True):
        tone: MuxedSignalGeneratorTone
        values = [tone.gain for tone in self.tones.values()]
        if include_default:
            values += [DEFAULT_GAIN] * (len(self.mask) - len(values))
        return  values

    def get_mux_phases(self, include_default:bool = True):
        tone: MuxedSignalGeneratorTone
        values = [tone.phase for tone in self.tones.values()]
        if include_default:
            values += [DEFAULT_PHASE] * (len(self.mask) - len(values))
        return  values


@dataclass
class MuxedSignalProcessorTone():
    ch: Optional[int|None] = None
    cfg: Optional[dict|None] = None

    duration: Optional[float|None] = None
    frequency: Optional[float|None] = None
    phase: Optional[float] = 0.0
    sel: Optional[Literal['product']|Literal['dds']|Literal['input']] = 'product'
    dac: Optional[int] = None
    
    acquisitions: list[Acquisition] = field(default_factory=list)


@dataclass
class MuxedSignalProcessor():
    tone_processors: defaultdict[ChannelId, MuxedSignalProcessorTone] = field(default_factory=lambda: defaultdict(MuxedSignalProcessorTone))
    tones: dict[ChannelId, MuxedSignalProcessorTone] = field(default_factory=dict)

@dataclass
class BiassingChannel():
    dac: str
    bias: float = 0.0


SignalGeneratorsType = Union[FullSpeedSignalGenerator, InterpolatedSignalGenerator, MuxedSignalGenerator]
SignalProcessorsType = Union[MuxedSignalProcessor]
BiassingChannelsType = Union[BiassingChannel]





@dataclass
class TIIqSettings:

    # Registered components
    components_drive: dict[ChannelId, DriveComponent] = field(default_factory=dict, init=False)
    components_flux: dict[ChannelId, FluxComponent] = field(default_factory=dict, init=False)
    components_probe: dict[ChannelId, ProbeComponent] = field(default_factory=dict, init=False)
    components_acquisition: dict[ChannelId, AcquisitionComponent] = field(default_factory=dict, init=False)
    
    # Registered signal generators

    signal_generators: dict[OutputId, SignalGeneratorsType] = field(default_factory=dict,init=False, repr=False)
    signal_processors: dict[InputId, SignalProcessorsType] = field(default_factory=dict,init=False, repr=False)
    signal_generators_pulses: defaultdict[OutputId, list[PulseLike]] = field(default_factory=lambda: defaultdict(list))
    
    drive_signal_generators: dict[OutputId, SignalGeneratorsType] = field(default_factory=dict, init=False)
    flux_signal_generators: dict[OutputId, SignalGeneratorsType] = field(default_factory=dict, init=False)
    flux_biassing_channels: dict[ChannelId, BiassingChannelsType] = field(default_factory=dict, init=False)
    probe_signal_generators: dict[OutputId, SignalGeneratorsType] = field(default_factory=dict, init=False)

    # Registered signal processors
    acquisition_signal_processors: dict[InputId, SignalProcessorsType] = field(default_factory=dict, init=False)

    rt_sweepers: list[tuple[str, int]] = field(default_factory=list, init=False)
    fl_sweepers: list[tuple[str, int]] = field(default_factory=list, init=False)

    # envelopes: dict[PortId, Envelope] = field(default_factory=dict, init=False)
    # pulses: dict[PortId, Pulse|Readout] = field(default_factory=dict, init=False)
    # acquisitions: dict[PortId, Acquisition|Readout] = field(default_factory=dict, init=False)



    def register_drive_channel(self, channel_id: ChannelId, output: OutputId, signal_generator:InterpolatedSignalGenerator, config: TIIqDriveConfig, port_id: PortId):
        self.components_drive[channel_id] = drive_component = DriveComponent.from_config(config, port_id)
        # TODO check that components are useful for anything. as far as it seems now, 
        # the information from config could be transferred to the sgs and sps directly
        # TODO check that portid is needed here anymore, 
        #############################################################
        self.drive_signal_generators[output] = signal_generator
        self.signal_generators[output] = signal_generator
        signal_generator.frequency = drive_component.frequency
        signal_generator.digital_mixer_frequency = drive_component.digital_mixer_frequency
        
    def register_flux_channel(self, channel_id: ChannelId, output: OutputId, signal_generator: FullSpeedSignalGenerator, config: TIIqFluxConfig, port_id: PortId):
        self.components_flux[channel_id] = flux_component = FluxComponent.from_config(config, port_id)
        #############################################################
        # register dc biassing channel
        dc_dac = get_dc_output_from_port_id(port_id)
        self.flux_biassing_channels[channel_id] = BiassingChannel(
            dac=dc_dac,
            bias=flux_component.offset
        )
        # register rf signal processor
        if signal_generator:
            self.flux_signal_generators[output] = signal_generator
            self.signal_generators[output] = signal_generator
            signal_generator.frequency = flux_component.frequency

    def register_probe_channel(self, channel_id: ChannelId, output: OutputId, mux_signal_generator: MuxedSignalGenerator, config: TIIqProbeConfig, port_id: PortId):
        if not channel_id in self.components_probe:
            # TODO: this was done to avoid registering a probe twice (one as part of its presence in)
            # channels, one as part of acquisition
            # but right now the info is going to be added to dicts within the already instantiated 
            # sg, so there should be no problem
            self.components_probe[channel_id] = probe_component = ProbeComponent.from_config(config, port_id)
            #############################################################

            self.probe_signal_generators[output] = mux_signal_generator
            self.signal_generators[output] = mux_signal_generator
            mux_signal_generator.digital_mixer_frequency = probe_component.digital_mixer_frequency
            # TODO: this is going to be run multiple times for each probe channel associated with this mux generator.
            # it doesn't yet check that they are all the same. ideally we remove it and have digital_mixer_frequency calculated
            tone: MuxedSignalGeneratorTone = mux_signal_generator.tones[channel_id]
            tone.frequency =  probe_component.frequency
            tone.gain = 1.0

    def register_acquisition_channel(self, 
                                     channel_id: ChannelId,
                                     input: InputId,
                                     mux_signal_processor: MuxedSignalProcessor,
                                     config: TIIqAcquisitionConfig,
                                     port_id: PortId, 
                                     probe_channel_id: ChannelId = None,
                                     ):
        if not channel_id in self.components_acquisition:
            self.components_acquisition[channel_id] = acquisition_component = AcquisitionComponent.from_config(config, port_id)
        #############################################################

            self.acquisition_signal_processors[input] = mux_signal_processor
            self.signal_processors[input] = mux_signal_processor
            # TODO common fields ...
            if not acquisition_component.frequency is None:
                frequency = acquisition_component.frequency
            elif probe_channel_id:
                # defaulting to probe frequency 
                probe_component: ProbeComponent = self.components_probe[probe_channel_id]
                frequency = probe_component.frequency
            else:
                raise ValueError("TODO missing frequency")
            if not channel_id in mux_signal_processor.tones:
                keys = list(mux_signal_processor.tone_processors.keys())
                key = keys[0]
                assert isinstance(key, int)
                mux_signal_processor.tones[channel_id] = mux_signal_processor.tone_processors.pop(key)
            tone: MuxedSignalProcessorTone = mux_signal_processor.tones[channel_id]
            tone.frequency = frequency


    def register_pulse(self, pulse: ScheduledPulse):
        output: OutputId = pulse.output
        self.signal_generators_pulses[output].append(pulse)

    def register_multiplexed_pulses(self, multiplexed_pulses: ScheduledMultiplexedPulses):
        output: OutputId = multiplexed_pulses.output
        # TODO check that all pulse envelopes are rectangular, start and finish at the same time
        self.signal_generators_pulses[output].append(multiplexed_pulses)

    def register_acquisitions(self, acquisition):
        pass

    def register_multiplexed_acquisitions(self, multiplexed_acquisition: ScheduledMultiplexedAcquisitions):
        input:InputId = multiplexed_acquisition.input
        mux_signal_processor: MuxedSignalProcessor = self.signal_processors[input]
        tone: MuxedSignalProcessorTone
        acquisition: ScheduledAcquisition
        for acquisition in multiplexed_acquisition.acquisitions:
            channel_id: ChannelId = acquisition.channel_id
            tone = mux_signal_processor.tones[channel_id]
            tone.duration = acquisition.duration
            # TODO register associated DAC
            


