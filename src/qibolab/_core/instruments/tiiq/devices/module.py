import numpy as np
import pathlib
from dataclasses import dataclass, field, asdict
from Pyro4 import Proxy
from typing import Optional, Literal

from qibo.config import log
from qibolab._core.components import (
    AcquisitionChannel,
    Config,
    Channel,
    DcChannel,
    IqChannel,
    IqConfig,
    OscillatorConfig,
)
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.instruments.tiiq.common import (
    PortId,
    OutputId,
    InputId,
    modes,
    SIMULATION,
    DEBUG,
    get_rf_output_from_port_id,
    get_dc_output_from_port_id,
    get_rf_input_from_port_id,
    debug_print,
)
from qibolab._core.instruments.tiiq.components.configs import TIIqDriveConfig, TIIqFluxConfig, TIIqProbeConfig, TIIqAcquisitionConfig
from qibolab._core.instruments.tiiq.devices.settings import (
    TIIqSettings,
    FullSpeedSignalGenerator,
    InterpolatedSignalGenerator,
    MuxedSignalGenerator,
    MuxedSignalGeneratorTone,
    MuxedSignalProcessor,
    MuxedSignalProcessorTone,
    BiassingChannel,
    SignalGeneratorsType,
    SignalProcessorsType,
    BiassingChannelsType,
)
from qibolab._core.instruments.tiiq.programs.program import TIIqProgram 
from qibolab._core.instruments.tiiq.scheduled import (
    Mutable,
    MutablePulse,
    MutableAcquisition,
    MutableReadout,
    MutableDelay,
    MutableVirtualZ,
    MutablePulseSequence, 
    MutableItem,
    Scheduled,
    ScheduledPulse,
    ScheduledAcquisition,
    ScheduledReadout,
    ScheduledDelay,
    ScheduledVirtualZ,
    ScheduledMultiplexedPulses,
    ScheduledMultiplexedAcquisitions,
    ScheduledPulseSequence
)
from qibolab._core.pulses import Align, Pulse, Delay, VirtualZ, Acquisition, Readout, PulseId, PulseLike
from qibolab._core.sequence import PulseSequence, InputOps, _synchronize
from qibolab._core.sweeper import ParallelSweepers, Parameter, Sweeper

from qick.qick_asm import QickConfig
try:
    from qick.qick import QickSoc
except:
    QickSoc = None

__all__ = ["TIIqModule"]

FIRMWARE_CONFIGURATION_PATH = str(pathlib.Path(__file__).parent.joinpath('firmware', 'firmware_configuration.json'))
TIDAC_NCHANNELS = 8

# from pydantic import BaseModel
# def replace(model: BaseModel, **update):
#     """Replace interface for pydantic models."""
#     return model.model_copy(update=update)


@dataclass
class TIIqModule:
    # static
    address: str
    channels: dict[ChannelId, Channel]

    soc: Proxy|None = field(init=False, repr=False)
    firmware_configuration: QickConfig|None = field(init=False, repr=False)
    tidac: Proxy|None = field(init=False, repr=False) # TIDAC80508
    libgpio_control: Proxy|None = field(init=False, repr=False) # libgpio_control

    outputs: list[str]|None = field(init=False, repr=False)
    inputs: list[str]|None = field(init=False, repr=False)
    signal_generators: dict[OutputId, SignalGeneratorsType] = field(init=False, repr=False)
    signal_processors: dict[InputId, SignalProcessorsType] = field(init=False, repr=False)
    channel_id_to_output_map: dict[ChannelId, OutputId] = field(init=False, repr=False)
    channel_id_to_input_map: dict[ChannelId, InputId] = field(init=False, repr=False)
    trigger_source: Optional[Literal['internal']|Literal['external']] = 'internal'

    acquisition_to_probe_map: dict[ChannelId, ChannelId] = field(init=False, repr=False)


    def _validate_channels(self, channels: dict[ChannelId, Channel]):
        assert isinstance(channels, dict)
        channel_id: ChannelId
        for channel_id in channels:
            if not (
                ("drive" in channel_id) or
                ("flux" in channel_id) or
                ("probe" in channel_id) or
                ("acquisition" in channel_id)
            ):
                raise ValueError("Channel id must contain 'drive', 'flux', 'probe', or 'acquisition'.")
            if "drive" in channel_id and not isinstance(channels[channel_id], Channel):
                raise ValueError("drive channels require a Channel object.")
            if "flux" in channel_id and not isinstance(channels[channel_id], DcChannel):
                raise ValueError("flux channels require a DcChannel object.")
            if "probe" in channel_id and not isinstance(channels[channel_id], Channel):
                raise ValueError("probe channels require a Channel object.")
            if "acquisition" in channel_id:
                if not isinstance(channels[channel_id], AcquisitionChannel):
                    raise ValueError("acquisition channels require an AcquisitionChannel object.")

    def __post_init__(self):
        # instantiate objects that are not passed as arguments
        self.soc = None
        self.firmware_configuration = None
        self.tidac = None
        self.libgpio_control = None

        self.inputs = None
        self.outputs = None
        self.signal_generators = {}
        self.signal_processors = {}
        self.channel_id_to_output_map = {}
        self.channel_id_to_input_map = {}

        self.acquisition_to_probe_map = {}

        self.settings = None
        self.configs = None
        self.options = None
        self.active_channels = None
        self.program = None
        self.scheduled_sequence = None
        self.fl_sweepers = None
        self.rt_sweepers = None
        self._validate_channels(self.channels)

    def _load_cached_firmware_configuration(self) -> QickConfig:
        try:
            return QickConfig(FIRMWARE_CONFIGURATION_PATH)
        except Exception as e:
            raise RuntimeError(f"Unable to load cached firmware configuration: ", e)

    def _connect_to_pyro_server(self):
        try:
            from Pyro4 import config as pyro_config, locateNS
            from Pyro4.util import getPyroTraceback
            from Pyro4.naming import NameServer

            pyro_config.SERIALIZER = "pickle"
            pyro_config.PICKLE_PROTOCOL_VERSION=4

            ns: NameServer = locateNS(host=self.address, port=8888)
            # # print the nameserver entries: you should see the QickSoc proxy
            # for k,v in ns.list().items():
            #     print(k,v)

            self.soc = Proxy(ns.lookup("soc"))
            self.firmware_configuration = QickConfig(self.soc.get_cfg())
            with open(FIRMWARE_CONFIGURATION_PATH, "w") as f:
                f.write(self.firmware_configuration.dump_cfg())

            self.tidac = Proxy(ns.lookup("tidac")) # TIDAC80508
            self.libgpio_control = Proxy(ns.lookup("libgpio_control"))

            # import sys
            # sys.path.append(peripherals_drivers)
            # from TIDAC80508 import TIDAC80508

            remote_traceback=True
            # adapted from https://pyro4.readthedocs.io/en/stable/errors.html and https://stackoverflow.com/a/70433500
            if remote_traceback:
                try:
                    import IPython
                    import sys
                    ip = IPython.get_ipython()
                    if ip is not None:
                        def exception_handler(self, etype, evalue, tb, tb_offset=None):
                            sys.stderr.write("".join(getPyroTraceback()))
                            # self.showtraceback((etype, evalue, tb), tb_offset=tb_offset)  # standard IPython's printout
                        ip.set_custom_exc((Exception,), exception_handler)  # register your handler
                except Exception as e:
                    raise RuntimeError("Failed to set up Pyro exception handler: ", e)
            # TODO: throw exception if it fails to connect
        except Exception as e:
            log.warning(f"""
                        Failed to connect to TIIqModule at IP {self.address}:
                        {e}
                        Switching to simulation mode.
                        """)
            raise Exception
            modes.enable(SIMULATION)
            self.firmware_configuration = self._load_cached_firmware_configuration()

    def _parse_firmware_configuration(self, firmware_configuration: QickConfig):
        # identify inputs and outputs
        self.outputs = list(firmware_configuration._cfg['dacs'])
        self.inputs = list(firmware_configuration._cfg['adcs'])

        # create signal generators and processors
        for ch, cfg in enumerate(firmware_configuration._cfg['gens']):
            if cfg['type'] == 'axis_sg_int4_v2':
                output = cfg['dac']
                self.signal_generators[output] = (
                        InterpolatedSignalGenerator(
                        ch=ch,
                        cfg=cfg,
                    )
                )
        
            elif cfg['type'] == 'axis_signal_gen_v6':
                output = cfg['dac']
                self.signal_generators[output] = (
                        FullSpeedSignalGenerator(
                        ch=ch,
                        cfg=cfg,
                    )
                )
            elif cfg['type'] == 'axis_sg_mixmux8_v1':
                output = cfg['dac']
                self.signal_generators[output] = (
                        MuxedSignalGenerator(
                        ch=ch,
                        cfg=cfg,
                    )
                )
            else:
                log.warning(f"Signal generator type {cfg['type']} is not supported.")

        inputs: list[InputId] = set([cfg['adc'] for cfg in firmware_configuration._cfg['readouts']])
        for input in inputs:
            self.signal_processors[input] = MuxedSignalProcessor()

        for ch, cfg in enumerate(firmware_configuration._cfg['readouts']):
            if cfg['ro_type'] == 'axis_pfb_readout_v4':
                input: InputId = cfg['adc']
                mux_signal_processor: MuxedSignalProcessor = self.signal_processors[input]
                mux_signal_processor.tone_processors[ch].ch = ch
                mux_signal_processor.tone_processors[ch].cfg = cfg
            else:
                log.warning(f"Signal processor type {cfg['ro_type']} is not supported.")

    def _map_channels_to_outputs_inputs(self, channels: dict[ChannelId, Channel]):
        for channel_id in channels:
            channel: Channel = channels[channel_id]
            port_id: PortId = channel.path
            if isinstance(channel, AcquisitionChannel):
                input: InputId = get_rf_input_from_port_id(port_id)
                self.channel_id_to_input_map[channel_id] = input
            else:
                output: OutputId = get_rf_output_from_port_id(port_id, require_valid=False)
                self.channel_id_to_output_map[channel_id] = output

    def _reset_bias(self):
        tidac: Proxy = self.tidac

        for ch in range(TIDAC_NCHANNELS):
            tidac.set_bias(ch, bias_value=0)

    def _initialize_peripherals(self):
        if not modes.is_enabled(SIMULATION):
            self._reset_bias()
        # # TODO expose gain so that SetDACVOP can be adjusted, instead of hardcoding 40000

    def connect(self):
        """Establish connection with the module."""
        log.info(f"Connecting to TIIqModule at IP {self.address}.")

        if modes.is_enabled(SIMULATION):
            self.firmware_configuration = self._load_cached_firmware_configuration()
        else:
            self._connect_to_pyro_server()

        self._parse_firmware_configuration(self.firmware_configuration)
        self._map_channels_to_outputs_inputs(self.channels)
        self._initialize_peripherals()

    def disconnect(self):
        """Close connection to the physical instrument."""
        log.info(f"Disconnecting from TIIqModule at IP {self.address}.")
        # TODO: disconnect from proxy


    # generated for every execution
    # settings: TIIqSettings = field(init=False, repr=False)
    # configs: dict[str, Config] = field(init=False, repr=False)
    # options: ExecutionParameters = field(init=False, repr=False)
    # active_channels: dict[ChannelId, Channel]|None = field(init=False, repr=False)
    # program: TIIqProgram|None = field(init=False, repr=False)
    # scheduled_sequence: ScheduledPulseSequence|None = field(init=False, repr=False)
    # fl_sweepers: list[ParallelSweepers]|None = field(init=False, repr=False)
    # rt_sweepers: list[ParallelSweepers]|None = field(init=False, repr=False)
    # programs: list[TIIqProgram]|None = field(init=False, repr=False)

    def _is_rt_sweeper(self, sweeper: Sweeper) -> bool:
        is_rt_sweeper: bool = False
        if sweeper.parameter == Parameter.frequency:
            supported_channel_types = ['drive', 'flux']
            if all(supported_channel_type in channel for channel in sweeper.channels for supported_channel_type in supported_channel_types):
                is_rt_sweeper = True
            # drive pulses -> True
            # flux pulses -> True
            # standalone probe pulses -> True
            # multiplexed probe pulses -> False
            # standalone acquisition -> True
            # multiplexed acquisition -> False
        elif sweeper.parameter == Parameter.amplitude:
            # TODO Qibolab issue, sweeper pulse does not contain information about the channel
            # to get it I have to check in the PulseSequence.pulse_channels()
            # or else I have to create my own Sweeper object and pass that info...
            is_rt_sweeper = True
            # if all(pulse.channel in supported for pulse in sweeper.pulses):
            #     is_rt_sweeper = True
            # drive pulses -> True
            # flux pulses -> True
            # standalone probe pulses -> True
            # multiplexed probe pulses -> False
        elif sweeper.parameter == Parameter.duration:
            is_rt_sweeper = False
        elif sweeper.parameter == Parameter.duration_interpolated:
            is_rt_sweeper = False
        elif sweeper.parameter == Parameter.relative_phase:
            is_rt_sweeper = True
        elif sweeper.parameter == Parameter.offset:
            supported = []
            if all(channel in supported for channel in sweeper.channels):
                is_rt_sweeper = True
            # drive pulses -> True
            # flux pulses -> True
            # standalone probe pulses -> True
            # multiplexed probe pulses -> False
            # standalone acquisition -> True
            # multiplexed acquisition -> False
        
        return is_rt_sweeper

    def _separate_fl_and_rt_sweepers(self, sweepers: list[ParallelSweepers]) -> tuple[list[ParallelSweepers], list[ParallelSweepers]]:
        fl_sweepers: list[ParallelSweepers] = []
        rt_sweepers: list[ParallelSweepers] = []
        for parallel_sweeper_set in sweepers:
            if all(self._is_rt_sweeper(sweeper) for sweeper in parallel_sweeper_set):
                rt_sweepers.append(parallel_sweeper_set)
            else:
                fl_sweepers.append(parallel_sweeper_set)
        # TODO check all sweeps in the set have the same number of steps
        # TODO: provide a map from the order of sweepers to the new order of fl_sweepers concatenated with rt_sweepers
        # so that we can reshape the data
        return fl_sweepers, rt_sweepers

    #####################################################################################

    def _create_sequence_copy(self, sequence: PulseSequence) -> PulseSequence:
        """Create a copy of the sequence."""
        # TODO This routine creates a new sequence but it references the same pulses and delays
        new_sequence = PulseSequence()
        for channel_id, pulse_like in sequence:
            new_sequence.append((channel_id, pulse_like)) 
        return new_sequence

    def _replace_aligns_with_delays(self, sequence: PulseSequence) -> PulseSequence:
        """Replace Align instructions with Delays."""
        new_sequence: PulseSequence = sequence.align_to_delays()
        return new_sequence

    def _equalize_channel_durations(self, sequence: PulseSequence) -> PulseSequence:
        """Equalize the durations of all channels in the sequence by adding delays."""	
        # TODO: make a new sequence instead of modifying the original one
        # TODO: fuse consecutive delays

        durations = {ch: sequence.channel_duration(ch) for ch in sequence.channels}
        max_duration = max(durations.values(), default=0.0)
        for ch, duration in durations.items():
            delay = max_duration - duration
            if delay > 0:
                sequence.append((ch, Delay(duration=delay)))
        return sequence

    def _generate_mutable_sequence(self, sequence: PulseSequence) -> PulseSequence:
        mutable_sequence: MutablePulseSequence = MutablePulseSequence()
        mutable_class_map={
            Pulse: MutablePulse,
            Acquisition: MutableAcquisition,
            Readout: MutableReadout,
            Delay: MutableDelay,
            VirtualZ: MutableVirtualZ,
        }

        for channel_id, pulse_like in sequence:
            if isinstance(pulse_like, Align):
                raise ValueError
            else:
                item_class = type(pulse_like)
                mutable_class = mutable_class_map[item_class]
                mutable_item: Mutable = mutable_class(channel_id, pulse_like)
                mutable_sequence.add_item(mutable_item)
        return mutable_sequence

    #####################################################################################

    def _process_rt_sweepers(self, sequence: MutablePulseSequence, rt_sweepers: list[ParallelSweepers], settings: TIIqSettings) -> MutablePulseSequence:
        # modifies sequence and settings in place
        
        def scheduled(items):
            for item in items:
                yield sequence.get_item(item.id)
        
        for parallel_sweeper_set in rt_sweepers:
            qnsteps = len(parallel_sweeper_set[0].values)
            name = f"Sweeper({parallel_sweeper_set[0].parameter}, {qnsteps})"

            from qick.asm_v2 import QickSweep1D
            import numpy as np
            for sweeper in parallel_sweeper_set:
                values = sweeper.values # np.linspace(start, stop, nsteps)
                nsteps = len(values)
                start = values[0]
                stop = values[-1]
                step = (stop - start) / (nsteps - 1)

                if sweeper.parameter == Parameter.amplitude:
                    amplitude = QickSweep1D(name, start, stop)
                    for pulse in scheduled(sweeper.pulses):
                        pulse.amplitude = amplitude
                elif sweeper.parameter == Parameter.duration:
                    pass
                    # NS_TO_US = 1e-3
                    # duration = QickSweep1D(name, start * NS_TO_US, stop * NS_TO_US) 
                    # for pulse in scheduled(sweeper.pulses):
                    #     pulse.duration = duration
                elif sweeper.parameter == Parameter.duration_interpolated:
                    pass
                    # for pulse in scheduled(sweeper.pulses):
                    #     pass
                elif sweeper.parameter == Parameter.relative_phase:
                    RAD_TO_DEG = 180/np.pi
                    relative_phase = QickSweep1D(name, start * RAD_TO_DEG, stop * RAD_TO_DEG) 
                    for pulse in scheduled(sweeper.pulses):
                        pulse.relative_phase = relative_phase
                elif sweeper.parameter == Parameter.frequency:
                    HZ_TO_MHZ = 1e-6
                    for channel_id in sweeper.channels:
                        if channel_id in self.channels:
                            if 'drive' in channel_id:
                                sg: InterpolatedSignalGenerator = settings.drive_signal_generators[channel_id]
                                sg.frequency = QickSweep1D(name, start, stop)
                            elif 'flux' in channel_id:
                                sg: FullSpeedSignalGenerator = settings.flux_signal_generators[channel_id]
                                sg.frequency = QickSweep1D(name, start, stop)
                            elif 'probe' in channel_id:
                                # TODO: implement a for loop
                                # ouput: OutputId = settings.channel_id_to_output_map[channel_id]
                                # sg: MuxedSignalGenerator = settings.probe_signal_generators[ouput]
                                # tone: MuxedSignalGeneratorTone = sg.tones[channel_id]
                                # tone.frequency = QickSweep1D(name, qstart * HZ_TO_MHZ, qstop * HZ_TO_MHZ)
                                pass
                            elif 'acquisition' in channel_id:
                                # TODO: implement a for loop
                                # input: OutputId = settings.channel_id_to_input_map[channel_id]
                                # sg: MuxedSignalGenerator = settings.acquisition_signal_processors[channel_id]
                                # tone: MuxedSignalProcessorTone = sg.tones[channel_id]
                                # tone.frequency = QickSweep1D(name, qstart * HZ_TO_MHZ, qstop * HZ_TO_MHZ)
                                pass
                elif sweeper.parameter == Parameter.offset:
                    for channel_id in sweeper.channels:
                        if not 'flux' in channel_id:
                            raise ValueError(f"Offset sweepers are only supported on flux channels {channel_id}.")
                        # TODO: implement a for loop
                else:
                    raise NotImplementedError(f"Sweeper parameter {sweeper.parameter} is not supported.")
        return sequence

    def _generate_scheduled_sequence(self, sequence: MutablePulseSequence) -> ScheduledPulseSequence:
        channels: dict[ChannelId, Channel] = sequence.channels
        time_at_channel: dict[ChannelId, float] = {}
        for channel_id in channels:
            time_at_channel[channel_id] = 0.0
        scheduled_sequence: ScheduledPulseSequence = ScheduledPulseSequence()
        scheduled_class_map={
            MutablePulse: ScheduledPulse,
            MutableAcquisition: ScheduledAcquisition,
            MutableReadout: ScheduledReadout,
            MutableDelay: ScheduledDelay,
            MutableVirtualZ: ScheduledVirtualZ,
        }

        item: Mutable
        for item in sequence:
            channel_id: ChannelId = item.channel_id
            start: float = time_at_channel[channel_id]
            if not isinstance(item, MutableItem):
                raise ValueError
            else:
                item_class = type(item)
                scheduled_class = scheduled_class_map[item_class]
                scheduled_item: Scheduled = scheduled_class(channel_id, start, item)
                duration: float = float(scheduled_item.duration)
                scheduled_sequence.add_item(scheduled_item)
            time_at_channel[channel_id] = duration + time_at_channel[channel_id]
        return scheduled_sequence

    #####################################################################################

    def _map_acquisitions_to_probes(self, configs: dict[str, Config], channels: dict[ChannelId, Channel]):
        """Analyzes channels configuration and saves the relationships between acquisition and probe channels."""
        for channel_id in channels:
            config: Config = configs[channel_id]
            channel: Channel = channels[channel_id]

            if isinstance(config, TIIqAcquisitionConfig) and isinstance(channel, AcquisitionChannel):
                probe_channel_id: ChannelId|None = channel.probe
                if probe_channel_id is not None:
                    self.acquisition_to_probe_map[channel_id] = probe_channel_id
            # elif isinstance(config, TIIqProbeConfig) and isinstance(channel, ProbeChannel):
            #     acquisition_channel_id: ChannelId|None = channel.acquisition
            #     if acquisition_channel_id is not None:
            #         self.acquisition_to_probe_map[acquisition_channel_id] = channel_id

    def _replace_readouts(self, scheduled_sequence: ScheduledPulseSequence, acquisition_to_probe_map: dict[ChannelId, ChannelId]) -> ScheduledPulseSequence:
        no_readout_scheduled_sequence = ScheduledPulseSequence()
        for item in scheduled_sequence:
            if isinstance(item, ScheduledReadout):
                readout: ScheduledReadout = item

                acquisition_channel_id = readout.channel_id
                acquisition: Acquisition = readout.acquisition
                scheduled_acquisition = ScheduledAcquisition(acquisition_channel_id, readout.start, MutableAcquisition(acquisition_channel_id, acquisition))
                no_readout_scheduled_sequence.add_item(scheduled_acquisition)

                # probe channel information is not included in the Readout object
                # try to find it within the module channels 
                if acquisition_channel_id in acquisition_to_probe_map:
                    probe_channel_id = acquisition_to_probe_map[acquisition_channel_id]
                    probe: Pulse = readout.probe
                    scheduled_probe = ScheduledPulse(probe_channel_id, readout.start, MutablePulse(probe_channel_id, probe))
                    no_readout_scheduled_sequence.add_item(scheduled_probe)
            else:
                no_readout_scheduled_sequence.add_item(item)

        return no_readout_scheduled_sequence

    #####################################################################################

    def _determine_active_channels(self, scheduled_sequence: ScheduledPulseSequence, channels: dict[ChannelId, Channel], configs: dict[str, Config]) -> list[ChannelId]:
        sequence_channels: set[ChannelId] = scheduled_sequence.channels

        channel_id: ChannelId
        active_channels:list[ChannelId] = []
        for channel_id in channels:
            config: Config = configs[channel_id]
            # active channels are all used in the sequence plus all flux channels
            if channel_id in sequence_channels or isinstance(config, TIIqFluxConfig):
                active_channels.append(channel_id)
        return active_channels
    
    def _filter_channels(self, scheduled_sequence: ScheduledPulseSequence, active_channels: list[ChannelId]) -> ScheduledPulseSequence:
        filtered_scheduled_sequence = ScheduledPulseSequence()
        for item in scheduled_sequence:
            if item.channel_id in active_channels:
                filtered_scheduled_sequence.add_item(item)
        return filtered_scheduled_sequence

    #####################################################################################

    def _is_channel_multiplexed(self, channel_id:ChannelId) -> bool:
        is_multiplexed: bool = False
        if channel_id in self.channel_id_to_output_map:
            output: OutputId = self.channel_id_to_output_map[channel_id]
            is_multiplexed = isinstance(self.signal_generators[output], MuxedSignalGenerator)
            # TODO check if the output is None and throw error explaining that there was an attempt to 
            # play a pulse in a flux channel that only supports DC biasing 
        elif channel_id in self.channel_id_to_input_map:
            input: InputId = self.channel_id_to_input_map[channel_id]
            is_multiplexed = isinstance(self.signal_processors[input], MuxedSignalProcessor)
        else:
            raise ValueError("TODO")
        return is_multiplexed

    def _add_port_n_multiplex_info(self, scheduled_sequence: ScheduledPulseSequence):
        for item in scheduled_sequence:
            if isinstance(item, ScheduledPulse):
                channel_id: ChannelId = item.channel_id
                output: OutputId = self.channel_id_to_output_map[channel_id]
                is_multiplexed: bool = self._is_channel_multiplexed(channel_id)
                item.output = output
                item.is_multiplexed = is_multiplexed
            elif isinstance(item, ScheduledAcquisition):
                channel_id: ChannelId = item.channel_id
                input: InputId = self.channel_id_to_input_map[channel_id]
                is_multiplexed: bool = self._is_channel_multiplexed(channel_id)
                item.input = input
                item.is_multiplexed = is_multiplexed

    #####################################################################################

    def _add_role_info(self, scheduled_sequence: ScheduledPulseSequence) -> ScheduledPulseSequence:
        # TODO
        return scheduled_sequence

    #####################################################################################

    def _sort_items(self, scheduled_sequence: ScheduledPulseSequence):
        scheduled_sequence.sort()

    def _calculate_lags(self, scheduled_sequence: ScheduledPulseSequence):
        previous_item_start: float = 0.0
        for item in scheduled_sequence:
            item.lag = item.start - previous_item_start
            previous_item_start = item.start

    # TODO consider merging sort and calculate lags

    #####################################################################################

    def _group_multiplexed(self, scheduled_sequence: ScheduledPulseSequence) -> ScheduledPulseSequence:
        # it assumes the pulse sequence was sorted beforehand
        # it currently only groups pulses and acquisitions with the same start time
        # TODO: verify all durations are the same
        # in the future implement a more general grouping algorithm that pads overlapping items

        grouped_scheduled_sequence = ScheduledPulseSequence()
        output_start_tracker: dict[OutputId, float] = {}
        multiplexed_pulses: ScheduledMultiplexedPulses
        input_start_tracker: dict[InputId, float] = {}
        multiplexed_acquisitions: ScheduledMultiplexedAcquisitions

        for item in scheduled_sequence:
            if isinstance(item, ScheduledPulse) and item.is_multiplexed:
                output: OutputId = item.output
                if not output in output_start_tracker or item.start > output_start_tracker[output]:
                    # create a new multiplexed item
                    multiplexed_pulses = ScheduledMultiplexedPulses(output=output, lag=item.lag)
                    multiplexed_pulses.append(item)
                    grouped_scheduled_sequence.add_item(multiplexed_pulses)
                    output_start_tracker[output] = item.start
                else:
                    multiplexed_pulses.append(item)

            elif isinstance(item, ScheduledAcquisition) and item.is_multiplexed:
                input: InputId = item.input
                if not input in input_start_tracker or item.start > input_start_tracker[input]:
                    # create a new multiplexed item
                    multiplexed_acquisitions = ScheduledMultiplexedAcquisitions(input=input, lag=item.lag)
                    multiplexed_acquisitions.append(item)
                    grouped_scheduled_sequence.add_item(multiplexed_acquisitions)
                    input_start_tracker[input] = item.start
                else:
                    multiplexed_acquisitions.append(item)

            else:
                    grouped_scheduled_sequence.add_item(item)

        return grouped_scheduled_sequence

    def _add_final_delay(self, total_duration: float, scheduled_sequence: ScheduledPulseSequence):
        # TODO eliminate, the delay is never added because the sequence is equalized at the begining and thus all channels have same duration
        # debug_print(f"""
        #             total_duration: {total_duration}
        #             scheduled_sequence duration: {scheduled_sequence.duration}
        #             """)
        if total_duration > scheduled_sequence.duration:
            for channel_id in scheduled_sequence.channels:
                start = scheduled_sequence.duration
                delay = ScheduledDelay(channel_id, start, MutableDelay(channel_id, Delay(kind='delay', duration=total_duration - scheduled_sequence.duration)))
                scheduled_sequence.add_item(delay)

    #####################################################################################

    def _configure_active_channels(self, active_channels: list[ChannelId], configs: dict[str, Config], channels: dict[ChannelId, Channel], settings: TIIqSettings):
        for channel_id in active_channels:
            config: Config = configs[channel_id]
            channel: Channel = channels[channel_id]
            port_id: PortId = channel.path

            if isinstance(config, TIIqDriveConfig):
                output: OutputId = self.channel_id_to_output_map[channel_id]
                signal_generator:InterpolatedSignalGenerator = self.signal_generators[output]
                settings.register_drive_channel(channel_id, output, signal_generator, config, port_id)

            elif isinstance(config, TIIqFluxConfig):
                output: OutputId = self.channel_id_to_output_map[channel_id]
                if output:
                    signal_generator: FullSpeedSignalGenerator = self.signal_generators[output]
                else:
                    signal_generator = None
                settings.register_flux_channel(channel_id, output, signal_generator, config, port_id)

            elif isinstance(config, TIIqProbeConfig):
                output: OutputId = self.channel_id_to_output_map[channel_id]
                mux_signal_generator: MuxedSignalGenerator = self.signal_generators[output]
                settings.register_probe_channel(channel_id, output, mux_signal_generator, config, port_id)

            elif isinstance(config, TIIqAcquisitionConfig) and isinstance(channel, AcquisitionChannel):
                probe_channel_id: ChannelId|None = channel.probe
                probe_config: TIIqProbeConfig|None = None
                probe_channel: Channel|None = None
                probe_port_id: PortId|None = None

                if probe_channel_id is not None:
                    probe_config = configs[probe_channel_id]
                    if not isinstance(probe_config, TIIqProbeConfig):
                        raise TypeError(f"Unsupported configuration {type(probe_config)} for channel {probe_channel_id}.")
                    probe_channel = channels[probe_channel_id]
                    probe_port_id = probe_channel.path

                    output: OutputId = self.channel_id_to_output_map[probe_channel_id]
                    if not output in settings.probe_signal_generators:
                        settings.register_probe_channel(probe_channel_id, probe_config, probe_port_id)
                    mux_signal_generator: MuxedSignalGenerator = settings.probe_signal_generators[output]
                    if not probe_channel_id in mux_signal_generator.tones:
                        settings.register_probe_channel(probe_channel_id, probe_config, probe_port_id)
                    tone: MuxedSignalGeneratorTone = mux_signal_generator.tones[probe_channel_id]

                input: InputId = self.channel_id_to_input_map[channel_id]
                mux_signal_processor: MuxedSignalProcessor = self.signal_processors[input]
                settings.register_acquisition_channel(channel_id, input, mux_signal_processor, config, port_id, probe_channel_id)

            else:
                raise TypeError(f"Unsupported configuration {type(config)} for channel {channel_id}.")

    def _register_pulses_and_acquisitions(self, scheduled_sequence: ScheduledPulseSequence, settings: TIIqSettings):
        # sg: SignalGeneratorsType
        # for sg in self.signal_generators: 
        #     sg.pulses = []
        #     sg.waveforms = {}
        # sp: SignalProcessorsType
        # for sp in self.signal_processors:
        #     tone: MuxedSignalProcessorTone
        #     for tone in sp.tones:
        #         tone.acquisitions = []

        for item in scheduled_sequence:
            if isinstance(item, ScheduledPulse):
                settings.register_pulse(item)
            elif isinstance(item, ScheduledMultiplexedPulses):
                settings.register_multiplexed_pulses(item)
            elif isinstance(item, ScheduledMultiplexedAcquisitions):
                settings.register_multiplexed_acquisitions(item)
            elif isinstance(item, ScheduledAcquisition):
                raise NotImplementedError("TODO")
                settings.register_acquisition(item)
            # else:
            #     raise ValueError("TODO")

    #####################################################################################

    def _fl_sweeper_recursive(self, fl_sweepers: list[ParallelSweepers], mutable_sequence: MutablePulseSequence):
        if len (fl_sweepers) > 0:
            parallel_sweeper_set = fl_sweepers.pop(0)
            nreps = len(parallel_sweeper_set[0].values)
            for iteration in range(nreps):
                for sweeper in parallel_sweeper_set:
                    if sweeper.parameter is Parameter.duration:
                        for pulse in sweeper.pulses:
                            p = mutable_sequence.get_item(pulse.id)
                            p.duration = sweeper.values[iteration]
                            if iteration == 11:
                                pass
                    elif sweeper.parameter is Parameter.amplitude:
                        for pulse in sweeper.pulses:
                            p = mutable_sequence.get_item(pulse.id)
                            p.amplitude = sweeper.values[iteration]
                    elif sweeper.parameter is Parameter.frequency:
                        for channel in sweeper.channels:
                            self.configs[channel] = self.configs[channel].model_copy(update={'frequency': sweeper.values[iteration]})
                    pass
                self._fl_sweeper_recursive(fl_sweepers.copy(), mutable_sequence.copy())
        else:
            # inputs
            channels: dict[ChannelId, Channel] = self.channels
            settings: TIIqSettings = TIIqSettings()
            configs: dict[str, Config] = self.configs
            options: ExecutionParameters = self.options
            firmware_configuration: QickConfig = self.firmware_configuration
            acquisition_to_probe_map: dict[ChannelId, ChannelId] = self.acquisition_to_probe_map
            rt_sweepers: list[ParallelSweepers] = self.rt_sweepers
            # local variables
            sequence_total_duration: float
            mutable_sequence: MutablePulseSequence
            program: TIIqProgram
            # outputs
            active_channels: list[ChannelId]
            scheduled_sequence: ScheduledPulseSequence

            # debug_print(mutable_sequence)

            mutable_sequence = self._process_rt_sweepers(mutable_sequence, rt_sweepers, settings)
            scheduled_sequence = self._generate_scheduled_sequence(mutable_sequence)
            self._map_acquisitions_to_probes(configs, channels)
            scheduled_sequence = self._replace_readouts(scheduled_sequence, acquisition_to_probe_map)
            sequence_total_duration = scheduled_sequence.duration
            active_channels = self._determine_active_channels(scheduled_sequence, channels, configs)
            scheduled_sequence = self._filter_channels(scheduled_sequence, active_channels)
            self._add_port_n_multiplex_info(scheduled_sequence)
            self._add_role_info(scheduled_sequence)
            self._sort_items(scheduled_sequence) 
            self._add_final_delay(sequence_total_duration, scheduled_sequence)
            self._calculate_lags(scheduled_sequence)
            scheduled_sequence = self._group_multiplexed(scheduled_sequence)
            
            self.active_channels = active_channels
            self.scheduled_sequence = scheduled_sequence
            
            # TODO: fix readout pulse gain not working
            # TODO: pulse duration sweeper does not affect the start of the next pulses

            # debug_print(scheduled_sequence)

            # # configure modules and ports
            self._configure_active_channels(active_channels, configs, channels, settings)
            
            # # register pulses and acquisitions
            self._register_pulses_and_acquisitions(scheduled_sequence, settings)
            settings.rt_sweepers = [(f"Sweeper({rt_sweeper_set[0].parameter}, {len(rt_sweeper_set[0].values)})", len(rt_sweeper_set[0].values)) for rt_sweeper_set in rt_sweepers]

            # generate program
            program = TIIqProgram(
                firmware_configuration=firmware_configuration,
                module_settings = settings,
                scheduled_sequence=scheduled_sequence,
                sequence_total_duration=sequence_total_duration,
                nshots=options.nshots,
                relaxation_time=options.relaxation_time
            )
            # debug_print(program)
            self.programs.append(program)

            if modes.is_enabled(DEBUG):
                import json
                from qick.helpers import progs2json
                # debug_print(json.dumps(self.program.envelopes, indent=2))
                # debug_print(json.dumps(self.program.waves, indent=2))
                dump = program.dump_prog()
                # del dump["prog_list"]
                # debug_print(progs2json(dump))
                with open(f"prog_{self.address}.json", "w") as file:
                    file.write(progs2json(dump))

    def prepare_execution(
        self,
        configs: dict[str, Config],
        sequence: PulseSequence,
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers]
    ):
        # debug_print(f"\nModule {self.address}")
        
        # for item in sequence:
        #     debug_print(item)

        new_sequence: PulseSequence
        mutable_sequence: MutablePulseSequence

        # process sequence and sweepers
        new_sequence = self._create_sequence_copy(sequence)
        new_sequence = self._replace_aligns_with_delays(new_sequence)
        new_sequence = self._equalize_channel_durations(new_sequence)
        mutable_sequence = self._generate_mutable_sequence(new_sequence)

        fl_sweepers, rt_sweepers = self._separate_fl_and_rt_sweepers(sweepers)
        self.configs = configs
        self.options = options
        self.fl_sweepers = fl_sweepers
        self.rt_sweepers = rt_sweepers
        self.programs: list[TIIqProgram] = []
        self._fl_sweeper_recursive(fl_sweepers.copy(), mutable_sequence.copy())

    def execute(
        self,
    ) -> dict[int, Result]:
        """Play a pulse sequence and retrieve feedback.

        If :class:`qibolab.Sweeper` objects are passed as arguments, they are
        executed in real-time. If not possible, an error is raised.

        Returns a mapping with the id of the probe pulses used to acquired data.
        """





        if not modes.is_enabled(SIMULATION):
            # set flux
            # bias_channel: BiassingChannel
            # for bias_channel in self.settings.flux_biassing_channels.values():
            #     self.tidac.set_bias(int(bias_channel.dac), bias_value=bias_channel.bias)
            # debug_print(self.programs[0])

            results = {}
            acquisition_counter = 0
            for item in self.scheduled_sequence:
                if isinstance(item, ScheduledMultiplexedAcquisitions):
                    for acquisition in item.acquisitions:
                        results[acquisition.id] = np.empty((0, 2))
                        acquisition_counter += 1
                        
                elif isinstance(item, ScheduledAcquisition):
                    results[item.id] = np.empty((0, 2))
                    acquisition_counter += 1



            program: TIIqProgram
            for program in self.programs:

                if any([isinstance(item, (ScheduledAcquisition, ScheduledMultiplexedAcquisitions)) for item in self.scheduled_sequence]):
                    iq_list = program.acquire(self.soc, soft_avgs=1, load_pulses=True, start_src=self.trigger_source, threshold=None, angle=None, progress=False, remove_offset=True)
                    # iq_list = self.program.acquire_decimated(self.soc, soft_avgs=1, start_src=self.trigger_source)
                    acquisition_counter = 0
                    for item in self.scheduled_sequence:
                        if isinstance(item, ScheduledMultiplexedAcquisitions):
                            for acquisition in item.acquisitions:
                                results[acquisition.id] = np.vstack([results[acquisition.id], iq_list[acquisition_counter]])
                                acquisition_counter += 1
                                
                        elif isinstance(item, ScheduledAcquisition):
                            results[item.id] = np.vstack([results[item.id], iq_list[acquisition_counter]])
                            acquisition_counter += 1

                else:
                    # self.program.run(self.soc, load_prog=True, load_pulses=True, start_src=self.trigger_source)
                    program.run_rounds(self.soc, rounds=1, load_pulses=True, start_src=self.trigger_source, progress=False)
                
            
            self._reset_bias()
            print(f"{self.address} completed execution")


        # TODO check if there are not acquisitions, if so, then run rounds instead of acquire
        # TODO test that all of the programs have the same length
        # TODO resonator frequency sweeps don't work 

        # upload settings & program
        # set bias
        # trigger execution
        # monitor execution
        # clear bias
        # download results & status

        # upload waveforms to memory
        # build loops

        #   schedule pulses & acquisitions
        #   modify paramenters
        return results




        # drive
        # channel_id 1---m> module 1---1> port 1---1> sg_ch 1---1> interpolated generator 1---1> output

        # flux
        #                                             sg_ch 1---1> full speed generator
        # channel_id 1---m> module 1---1> port 1---2>                                  2---1> output
        #                                             sg_ch 1---1>   biassing channel

        # probe
        #              channel_id
        # group 1---8> channel_id 8---m> module 1---1> port 1---1> sg_ch 1---1> multiplexed generator 1---1> output
        #              channel_id

        # acquisition
        #              channel_id <1---1 module <1---1 port <1---1 sp_ch <1---1 signal processor
        # group <1---8 channel_id <1---1 module <1---1 port <1---1 sp_ch <1---1 signal processor <8---1 input
        #              channel_id <1---1 module <1---1 port <1---1 sp_ch <1---1 signal processor

        # * 1---m> currently, one channel can be connected to more than one output,
        #          provided that they are on separate boards. This would be useful to
        #          see the pulses generated in the oscilloscope.



        ##############################################################################################################
        # INIT/CONNECT ###############################################################################################
        ##############################################################################################################
        # parse firmware configuration [firmware_configuration, channels] -> settings
        #   identify inputs and outputs
        #   create signal generators and processors
        #     drive_signal_generators[output]: dict[str, InterpolatedSignalGenerator]
        #     flux_signal_generators[output]: dict[str, FullSpeedSignalGenerator]
        #     flux_biassing_channels[output]: dict[str, BiassingChannel]
        #     probe_signal_generators[output]: dict[str, MuxedSignalGenerator]
        #     acquisition_signal_processors[input]: dict[str, MuxedSignalProcessor]

        #     signal_generators[output]: dict[str, SignalGeneratorsType]
        #     signal_processors[input]: dict[str, SignalProcessorsType]

        # map channels to outputs/inputs
        #   channel_id_to_output_map
        #   channel_id_to_input_map
        
        # initialize peripherals

        ##############################################################################################################
        # PLAY #######################################################################################################
        ##############################################################################################################
        # preprocess pulse sequence [sequence] -> tiiq_sequence
        #   ignore any item linked to a channel that does not belong to the module
        #   convert Readout into Pulse + Acquisition
        #   schedule Pulses and Acquisitions with absolute start times, remove Delays
        #   append input/output to all items
        #   append role (drive, flux, probe, acquisition) to all items
        #   sort Pulses and Acquisitions by their start time
        #   group acquisitions for the same output, same time, and replace them with MultiplexedAcquisition
        #   group probes for the same input, same time, and replace them with MultiplexedPulse
        #   calculate lag an element and the previous element

        # process sweepers [sweepers] -> tiiq_sweepers

        # process pulse sequence [tiiq_sequence, configs] -> settings
        #   determine active channels
        #   register active generators and processors
        #   configure active generators and processors
        #   register pulses and acquisitions

        # generate program [settings, firmware_configuration, tiiq_sequence, tiiq_sweepers, options] -> program
        #   initialize
        #     initialize_drive_signal_generators
        #     initialize_flux_signal_generators
        #     initialize_probe_signal_generators
        #     initialize_acquisition_signal_processors
        #   body
        #     play_drive_pulse(pulse) # ScheduledPulse
        #     play_flux_pulse(pulse) # ScheduledPulse
        #     play_probe_pulse(pulse) # ScheduledMultiplexedPulse
        #     trigger_acquisition(acquisition) # ScheduledMultiplexedAcquisition

        # run 
        #   set bias
        #   spawn new thread
        #       spawn new thread
        #       trigger tProc
        #   acquire [program] -> results
        #   join threads
        #   close bias
        #   process results
