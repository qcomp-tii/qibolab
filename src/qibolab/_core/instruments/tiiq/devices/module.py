import pathlib
import threading
import time
from dataclasses import dataclass, field, asdict
from threading import Event
from Pyro4 import Proxy

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
)
from qibolab._core.instruments.tiiq.programs.program import TIIqProgram 
from qibolab._core.instruments.tiiq.scheduled import (
    Scheduled,
    ScheduledPulse,
    ScheduledAcquisition,
    ScheduledReadout,
    ScheduledDelay,
    ScheduledMultiplexedPulses,
    ScheduledMultiplexedAcquisitions,
    ScheduledPulseSequence
)
from qibolab._core.pulses import Align, Pulse, Delay, VirtualZ, Acquisition, Readout, PulseId, PulseLike
from qibolab._core.sequence import PulseSequence, InputOps
from qibolab._core.sweeper import ParallelSweepers, Parameter, Sweeper

from qick.qick_asm import QickConfig
try:
    from qick.qick import QickSoc
except:
    QickSoc = None

__all__ = ["TIIqModule"]

FIRMWARE_CONFIGURATION_PATH = str(pathlib.Path(__file__).parent.joinpath('firmware', 'firmware_configuration.json'))
TIDAC_NCHANNELS = 8

from pydantic import BaseModel
def replace(model: BaseModel, **update):
    """Replace interface for pydantic models."""
    return model.model_copy(update=update)


@dataclass
class TIIqModule:
    # static
    address: str
    channels: dict[ChannelId, Channel]
    soc: Proxy|None = field(init=False, repr=False)
    firmware_configuration: QickConfig = field(init=False, repr=False)
    tidac: Proxy|None = field(init=False, repr=False) # TIDAC80508
    libgpio_control: Proxy|None|None = field(init=False, repr=False) # libgpio_control

    # generated for every execution
    settings: TIIqSettings = field(init=False, repr=False)
    configs: dict[str, Config] = field(init=False, repr=False)
    active_channels: dict[ChannelId, Channel]|None = field(init=False, repr=False)
    program: TIIqProgram|None = field(init=False, repr=False)
    scheduled_sequence: ScheduledPulseSequence|None = field(init=False, repr=False)
    acquisition_completed_flag: Event|None = field(init=False, repr=False)

    def _validate_channels(self):
        # inputs
        channels: dict[ChannelId, Channel] = self.channels

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

        self.settings = TIIqSettings()
        self.configs = None
        self.active_channels = None
        self.program = None
        self.scheduled_sequence = None
        self.acquisition_completed_flag = Event()
        self._validate_channels()


    def _load_cached_firmware_configuration(self):
        try:
            self.firmware_configuration = QickConfig(FIRMWARE_CONFIGURATION_PATH)
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
            modes.enable(SIMULATION)
            self._load_cached_firmware_configuration()

    def _reset_bias(self):
        tidac = self.tidac

        for ch in range(TIDAC_NCHANNELS):
            tidac.set_bias(ch, bias_value=0)

    def _parse_firmware_configuration(self):
        channels: dict[ChannelId, Channel] = self.channels
        firmware_configuration: QickConfig = self.firmware_configuration
        settings: TIIqSettings = self.settings

        # identify inputs and outputs
        settings.outputs = list(firmware_configuration._cfg['dacs'])
        settings.inputs = list(firmware_configuration._cfg['adcs'])

        # create signal generators and processors
        for ch, cfg in enumerate(firmware_configuration._cfg['gens']):
            if cfg['type'] == 'axis_sg_int4_v2':
                output = cfg['dac']
                settings.signal_generators[output] = (
                        InterpolatedSignalGenerator(
                        ch=ch,
                        cfg=cfg,
                    )
                )
        
            elif cfg['type'] == 'axis_signal_gen_v6':
                output = cfg['dac']
                settings.signal_generators[output] = (
                        FullSpeedSignalGenerator(
                        ch=ch,
                        cfg=cfg,
                    )
                )
            elif cfg['type'] == 'axis_sg_mixmux8_v1':
                output = cfg['dac']
                settings.signal_generators[output] = (
                        MuxedSignalGenerator(
                        ch=ch,
                        cfg=cfg,
                    )
                )
            else:
                log.warning(f"Signal generator type {cfg['type']} is not supported.")

        inputs: list[InputId] = set([cfg['adc'] for cfg in firmware_configuration._cfg['readouts']])
        for input in inputs:
            settings.signal_processors[input] = MuxedSignalProcessor()

        for ch, cfg in enumerate(firmware_configuration._cfg['readouts']):
            if cfg['ro_type'] == 'axis_pfb_readout_v4':
                input: InputId = cfg['adc']
                mux_signal_processor: MuxedSignalProcessor = settings.signal_processors[input]
                mux_signal_processor.tone_processors[ch].ch = ch
                mux_signal_processor.tone_processors[ch].cfg = cfg
            else:
                log.warning(f"Signal processor type {cfg['ro_type']} is not supported.")

    def _map_channels_to_outputs_inputs(self):
        channels: dict[ChannelId, Channel] = self.channels
        settings: TIIqSettings = self.settings

        for channel_id in channels:
            channel: Channel = channels[channel_id]
            port_id: PortId = channel.path
            if isinstance(channel, AcquisitionChannel):
                input: InputId = get_rf_input_from_port_id(port_id)
                settings.channel_id_to_input_map[channel_id] = input
            else:
                output: OutputId = get_rf_output_from_port_id(port_id, require_valid=False)
                settings.channel_id_to_output_map[channel_id] = output

    def _initialize_peripherals(self):
        # # TODO move this to board code (the one that starts the pyro server)
        # import sys
        # sys.path.append('/home/xilinx/jupyter_notebooks/qick/qick_demos/custom/drivers')

        # from TIDAC80508 import TIDAC80508
        # tidac = TIDAC80508()

        # ### SET POWER FOR DACs ###
        # dac_2280 = soc.usp_rf_data_converter_0.dac_tiles[0].blocks[0]
        # dac_2280.SetDACVOP(40000)
        # dac_2281 = soc.usp_rf_data_converter_0.dac_tiles[0].blocks[1]
        # dac_2281.SetDACVOP(40000)
        # dac_2282 = soc.usp_rf_data_converter_0.dac_tiles[0].blocks[2]
        # dac_2282.SetDACVOP(40000)
        # dac_2283 = soc.usp_rf_data_converter_0.dac_tiles[0].blocks[3]
        # dac_2283.SetDACVOP(40000)
        # dac_2290 = soc.usp_rf_data_converter_0.dac_tiles[1].blocks[0]
        # dac_2290.SetDACVOP(40000)

        # dac_2292 = soc.usp_rf_data_converter_0.dac_tiles[1].blocks[2]
        # dac_2292.SetDACVOP(40000)

        # dac_2230 = soc.usp_rf_data_converter_0.dac_tiles[2].blocks[0]
        # dac_2230.SetDACVOP(40000)
        # dac_2231 = soc.usp_rf_data_converter_0.dac_tiles[2].blocks[1]
        # dac_2231.SetDACVOP(40000)
        # dac_2232 = soc.usp_rf_data_converter_0.dac_tiles[2].blocks[2]
        # dac_2232.SetDACVOP(40000)        

        # ### ENABLE MULTI TILE SYNCHRONIZATION ###
        # soc.usp_rf_data_converter_0.mts_dac_config.RefTile = 2
        # soc.usp_rf_data_converter_0.mts_dac_config.Tiles = 0b0011
        # soc.usp_rf_data_converter_0.mts_dac_config.SysRef_Enable = 1
        # soc.usp_rf_data_converter_0.mts_dac_config.Target_Latency = -1
        # soc.usp_rf_data_converter_0.mts_dac()

        
        pass
            
    def connect(self):
        """Establish connection with the module."""
        log.info(f"Connecting to TIIqModule at IP {self.address}.")

        if modes.is_enabled(SIMULATION):
            self._load_cached_firmware_configuration()
        else:
            self._connect_to_pyro_server()

        self.settings.firmware_configuration = self.firmware_configuration 
        # TODO: firmware_configuration may not be needed in settings
        
        if not modes.is_enabled(SIMULATION):
            self._reset_bias()
        self._parse_firmware_configuration()
        self._map_channels_to_outputs_inputs()
        self._initialize_peripherals()

    def disconnect(self):
        """Close connection to the physical instrument."""
        log.info(f"Disconnecting from TIIqModule at IP {self.address}.")
        # TODO: disconnect from proxy


    def _map_acquisitions_to_probes(self):
        # inputs
        settings: TIIqSettings = self.settings
        configs: dict[str, Config] = self.configs
        channels: dict[ChannelId, Channel] = self.channels

        # outputs
        # settings.acquisition_to_probe_map: dict[ChannelId, ChannelId]

        for channel_id in channels:
            config: Config = configs[channel_id]
            channel: Channel = channels[channel_id]

            if isinstance(config, TIIqAcquisitionConfig) and isinstance(channel, AcquisitionChannel):
                probe_channel_id: ChannelId|None = channel.probe
                if probe_channel_id is not None:
                    settings.acquisition_to_probe_map[channel_id] = probe_channel_id
            # elif isinstance(config, TIIqProbeConfig) and isinstance(channel, ProbeChannel):
            #         acquisition_channel_id: ChannelId|None = channel.acquisition
            #         settings.acquisition_to_probe_map[acquisition_channel_id] = channel_id

    def _generate_scheduled_sequence(self, sequence: PulseSequence):
        # inputs
        channels = sequence.channels

        # outputs
        # self.scheduled_sequence: ScheduledPulseSequence

        time_at_channel: dict[ChannelId, float] = {}
        for channel_id in channels:
            time_at_channel[channel_id] = 0.0
        scheduled_sequence: ScheduledPulseSequence = ScheduledPulseSequence()
        scheduled_class_map={
            Pulse: ScheduledPulse,
            Acquisition: ScheduledAcquisition,
            Readout: ScheduledReadout,
            Delay: ScheduledDelay,
        }

        for channel_id, pulse_like in sequence:
            start: float = time_at_channel[channel_id]
            if isinstance(pulse_like, Align):
                raise ValueError
            elif isinstance(pulse_like, VirtualZ):
                raise NotImplementedError
            else:
                item_class = type(pulse_like)
                scheduled_class = scheduled_class_map[item_class]
                scheduled_item: Scheduled = scheduled_class(channel_id, start, pulse_like)
            duration: float = float(scheduled_item.duration)
            time_at_channel[channel_id] += duration
            scheduled_sequence.add_item(scheduled_item)
        self.scheduled_sequence = scheduled_sequence

    def _process_sweepers(self, sweepers: list[ParallelSweepers]):
        # inputs
        scheduled_sequence: ScheduledPulseSequence = self.scheduled_sequence

        # outputs
        # self.settings.rt_sweepers: list[tuple[str, int]]
        # TODO

        def scheduled(items):
            for item in items:
                yield scheduled_sequence.get_item(item.id)
        
        for parallel_sweeper_set in sweepers:
            # self.validate_sweeper(sweeper)
            qnsteps = len(parallel_sweeper_set[0].values)
            name = f"Sweeper({parallel_sweeper_set[0].parameter}, {qnsteps})"
            # TODO depending on type add to rt or fl sweepers
            self.settings.rt_sweepers.append((name, qnsteps))
            # self.program.add_loop(name, qnsteps)
            from qick.asm_v2 import QickSweep1D
            import numpy as np
            # TODO check all can be done in real time

            # TODO check if the item is reaout, then change its probe and acquisition
            for sweeper in parallel_sweeper_set:
                values = sweeper.values # np.arange(start, stop, step)
                qstart = values[0]
                qstop = values[-1]

                if sweeper.parameter == Parameter.amplitude:
                    amplitude = QickSweep1D(name, qstart, qstop)
                    for pulse in scheduled(sweeper.pulses):
                        pulse.amplitude = amplitude
                elif sweeper.parameter == Parameter.duration:
                    NS_TO_US = 1e-3
                    duration = QickSweep1D(name, qstart, qstop) * NS_TO_US
                    for pulse in scheduled(sweeper.pulses):
                        pulse.duration = duration
                elif sweeper.parameter == Parameter.duration_interpolated:
                    for pulse in scheduled(sweeper.pulses):
                        pass
                elif sweeper.parameter == Parameter.relative_phase:
                    RAD_TO_DEG = 180/np.pi
                    relative_phase = QickSweep1D(name, qstart, qstop) * RAD_TO_DEG
                    for pulse in scheduled(sweeper.pulses):
                        pulse.relative_phase = relative_phase
                elif sweeper.parameter == Parameter.frequency:
                    HZ_TO_MHZ = 1e-6
                    for channel_id in sweeper.channels:
                        if channel_id in self.channels:
                            if 'drive' in channel_id:
                                sg: InterpolatedSignalGenerator = self.settings.drive_signal_generators[channel_id]
                                sg.frequency = QickSweep1D(name, qstart, qstop)
                            elif 'flux' in channel_id:
                                sg: FullSpeedSignalGenerator = self.settings.flux_signal_generators[channel_id]
                                sg.frequency = QickSweep1D(name, qstart, qstop)
                            elif 'probe' in channel_id:
                                # TODO: implement a for loop
                                # ouput: OutputId = self.settings.channel_id_to_output_map[channel_id]
                                # sg: MuxedSignalGenerator = self.settings.probe_signal_generators[ouput]
                                # tone: MuxedSignalGeneratorTone = sg.tones[channel_id]
                                # tone.frequency = QickSweep1D(name, qstart * HZ_TO_MHZ, qstop * HZ_TO_MHZ)
                                pass
                            elif 'acquisition' in channel_id:
                                # TODO: implement a for loop
                                # input: OutputId = self.settings.channel_id_to_input_map[channel_id]
                                # sg: MuxedSignalGenerator = self.settings.acquisition_signal_processors[channel_id]
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

        # self.add_loop("myloop", self.cfg["steps"])
        # 'phase': QickSweep1D("myloop", -360, 720),
        # 'gain': QickSweep1D("myloop", 0.0, 1.0),
        # self.delay(QickSweep1D("myloop", 0.9, 1.3))

        # self.add_loop("loop1", self.cfg["steps1"]) # this will be the outer loop
        # self.add_loop("loop2", self.cfg["steps2"]) # this will be the inner loop

    def _replace_readouts(self):
        # inputs
        scheduled_sequence: ScheduledPulseSequence = self.scheduled_sequence

        # outputs
        # self.scheduled_sequence: ScheduledPulseSequence

        no_readout_scheduled_sequence = ScheduledPulseSequence()
        for item in scheduled_sequence:
            if isinstance(item, ScheduledReadout):
                readout: ScheduledReadout = item

                acquisition_channel_id = readout.channel_id
                acquisition: Acquisition = readout.acquisition
                scheduled_acquisition = ScheduledAcquisition(acquisition_channel_id, readout.start, acquisition)
                no_readout_scheduled_sequence.add_item(scheduled_acquisition)

                # probe channel information is not included in the Readout object
                # try to find it within the module channels 
                if acquisition_channel_id in self.settings.acquisition_to_probe_map:
                    probe_channel_id = self.settings.acquisition_to_probe_map[acquisition_channel_id]
                    probe: Pulse = readout.probe
                    scheduled_probe = ScheduledPulse(probe_channel_id, readout.start, probe)
                    no_readout_scheduled_sequence.add_item(scheduled_probe)
            else:
                no_readout_scheduled_sequence.add_item(item)

            self.scheduled_sequence = no_readout_scheduled_sequence

    def _determine_active_channels(self):
        # inputs
        channels: dict[ChannelId, Channel] = self.channels
        sequence_channels: set[ChannelId] = self.scheduled_sequence.channels
        configs: dict[str, Config] = self.configs

        # outputs
        # self.active_channels: list[ChannelId]

        channel_id: ChannelId
        channel: Channel
        active_channels:list[ChannelId] = []
        for channel_id in channels:
            config: Config = configs[channel_id]
            # active channels are all used in the sequence plus all flux channels
            if channel_id in sequence_channels or isinstance(config, TIIqFluxConfig):
                active_channels.append(channel_id)
        self.active_channels = active_channels
    
    def _filter_channels(self):
        # inputs
        scheduled_sequence: ScheduledPulseSequence = self.scheduled_sequence

        # outputs
        # self.scheduled_sequence: ScheduledPulseSequence   

        filtered_scheduled_sequence = ScheduledPulseSequence()
        for item in scheduled_sequence:
            if item.channel_id in self.active_channels:
                filtered_scheduled_sequence.add_item(item)
        self.scheduled_sequence = filtered_scheduled_sequence

    def _add_port_n_multiplex_info(self):
        # inputs
        scheduled_sequence: ScheduledPulseSequence = self.scheduled_sequence

        # outputs
        # self.scheduled_sequence: ScheduledPulseSequence 

        for item in scheduled_sequence:
            if isinstance(item, ScheduledPulse):
                channel_id: ChannelId = item.channel_id
                output: OutputId = self.settings.channel_id_to_output_map[channel_id]
                is_multiplexed: bool = self.settings.is_channel_multiplexed(channel_id)
                item.output = output
                item.is_multiplexed = is_multiplexed
            elif isinstance(item, ScheduledAcquisition):
                channel_id: ChannelId = item.channel_id
                input: InputId = self.settings.channel_id_to_input_map[channel_id]
                is_multiplexed: bool = self.settings.is_channel_multiplexed(channel_id)
                item.input = input
                item.is_multiplexed = is_multiplexed

    def _add_role_info(self):
        # TODO
        scheduled_sequence: ScheduledPulseSequence = self.scheduled_sequence

    def _sort_items(self):
        # inputs
        scheduled_sequence: ScheduledPulseSequence = self.scheduled_sequence

        # outputs
        # self.scheduled_sequence: ScheduledPulseSequence   

        scheduled_sequence.sort()

    def _calculate_lags(self):
        # inputs
        scheduled_sequence: ScheduledPulseSequence = self.scheduled_sequence

        # outputs
        # self.scheduled_sequence: ScheduledPulseSequence   

        previous_item_start: float = 0.0
        for item in scheduled_sequence:
            item.lag = item.start - previous_item_start
            previous_item_start = item.start

    # TODO consider merging sort and calculate lags

    def _group_multiplexed(self):
        scheduled_sequence: ScheduledPulseSequence = self.scheduled_sequence
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
                    output_start_tracker[output] = item.start

                if item.start == output_start_tracker[output]:
                    grouped_scheduled_sequence.add_item(multiplexed_pulses)

            elif isinstance(item, ScheduledAcquisition) and item.is_multiplexed:
                input: InputId = item.input
                if not input in input_start_tracker or item.start > input_start_tracker[input]:
                    # create a new multiplexed item
                    multiplexed_acquisitions = ScheduledMultiplexedAcquisitions(input=input, lag=item.lag)
                    multiplexed_acquisitions.append(item)
                    input_start_tracker[input] = item.start

                if item.start == input_start_tracker[input]:
                    grouped_scheduled_sequence.add_item(multiplexed_acquisitions)

            else:
                    grouped_scheduled_sequence.add_item(item)

        self.scheduled_sequence = grouped_scheduled_sequence

    def _add_final_delay(self, total_duration: float):
        scheduled_sequence: ScheduledPulseSequence = self.scheduled_sequence
        debug_print(f"""
                    total_duration: {total_duration}
                    scheduled_sequence duration: {scheduled_sequence.duration}
                    """)

    def _configure_active_channels(self, ):
        channels: list[ChannelId] = self.active_channels
        configs: dict[str, Config] = self.configs

        for channel_id in channels:
            config: Config = configs[channel_id]
            channel: Channel = self.channels[channel_id]
            port_id: PortId = channel.path

            if isinstance(config, TIIqDriveConfig):
                self.settings.register_drive_channel(channel_id, config, port_id)

            elif isinstance(config, TIIqFluxConfig):
                self.settings.register_flux_channel(channel_id, config, port_id)

            elif isinstance(config, TIIqProbeConfig):
                self.settings.register_probe_channel(channel_id, config, port_id)

            elif isinstance(config, TIIqAcquisitionConfig) and isinstance(channel, AcquisitionChannel):
                probe_channel_id: ChannelId|None = channel.probe
                probe_config: TIIqProbeConfig|None = None
                probe_channel: Channel|None = None
                probe_port_id: PortId|None = None

                if probe_channel_id is not None:
                    probe_config = configs[probe_channel_id]
                    if not isinstance(probe_config, TIIqProbeConfig):
                        raise TypeError(f"Unsupported configuration {type(probe_config)} for channel {probe_channel_id}.")
                    probe_channel = self.channels[probe_channel_id]
                    probe_port_id = probe_channel.path
                self.settings.register_acquisition_channel(channel_id, config, port_id, probe_channel_id, probe_config, probe_port_id)

            else:
                raise TypeError(f"Unsupported configuration {type(config)} for channel {channel_id}.")

    def _register_pulses_and_acquisitions(self):
        scheduled_sequence: ScheduledPulseSequence = self.scheduled_sequence
        
        for item in scheduled_sequence:
            if isinstance(item, ScheduledPulse):
                self.settings.register_pulse(item)
            elif isinstance(item, ScheduledMultiplexedPulses):
                self.settings.register_multiplexed_pulses(item)
            elif isinstance(item, ScheduledMultiplexedAcquisitions):
                self.settings.register_multiplexed_acquisitions(item)
            elif isinstance(item, ScheduledAcquisition):
                raise NotImplementedError("TODO")
                self.settings.register_acquisition(item)
            # else:
            #     raise ValueError("TODO")


    def _low_high_gpio(self, t_sleep=0.01):
        self.libgpio_control.set_gpio_low()
        time.sleep(t_sleep)
        self.libgpio_control.set_gpio_high()

    def _trigger_start_pulse(self):
        self.libgpio_control.initialize_gpio()
        self._low_high_gpio(0.01)
        last_reset_time = time.monotonic()
        while not self.acquisition_completed_flag.is_set():
            time.sleep(0.01)
            # if (time.monotonic() - last_reset_time >= 10):  # Check if 10 seconds have passed
            #     print(f"{self.address} RESETING IP FSM...")
            #     self.low_high_gpio(0.1)
            #     last_reset_time = time.monotonic()
        self.libgpio_control.set_gpio_low()
        self.libgpio_control.cleanup_gpio()

    def prepare_execution(
        self,
        configs: dict[str, Config],
        sequence: PulseSequence,
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers]
    ):
        debug_print(f"\nModule {self.address}")
        self.configs = configs
        
        for item in sequence:
            debug_print(item)

        # process sequence and sweepers
        sequence_total_duration: float = sequence.duration
        self._map_acquisitions_to_probes()
        self._generate_scheduled_sequence(sequence)
        self._process_sweepers(sweepers)
        self._replace_readouts()
        self._determine_active_channels()
        self._filter_channels()
        self._add_port_n_multiplex_info()
        self._add_role_info()
        self._sort_items()
        self._calculate_lags()
        self._group_multiplexed()
        self._add_final_delay(sequence_total_duration)
        
        debug_print(self.scheduled_sequence)

        # configure modules and ports
        self._configure_active_channels()
        
        # register pulses and acquisitions
        self._register_pulses_and_acquisitions()

        # generate program
        self.program = TIIqProgram(
            module_settings = self.settings,
            firmware_configuration=self.firmware_configuration,
            scheduled_sequence=self.scheduled_sequence,
            sequence_total_duration=sequence_total_duration,
            nshots=options.nshots,
            relaxation_time=options.relaxation_time
        )
        debug_print(self.program)
        
        # import json
        # debug_print(json.dumps(self.program.envelopes, indent=2))
        # debug_print(json.dumps(self.program.waves, indent=2))
        # dump = self.program.dump_prog()
        # del dump["prog_list"]
        # debug_print(json.dumps(dump, indent=2))

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

            if self.settings.trigger_source=='external':                    
                self.acquisition_completed_flag.clear()
                trigger_thread = threading.Thread(target=self._trigger_start_pulse)
                trigger_thread.start()

            if any([isinstance(item, (ScheduledAcquisition, ScheduledMultiplexedAcquisitions)) for item in self.scheduled_sequence]):
                iq_list = self.program.acquire(self.soc, soft_avgs=1, load_pulses=True, start_src=self.settings.trigger_source, threshold=None, angle=None, progress=False, remove_offset=True)
                # iq_list = self.program.acquire_decimated(self.soc, soft_avgs=1, start_src=self.settings.trigger_source)
            else:
                # self.program.run(self.soc, load_prog=True, load_pulses=True, start_src=self.settings.trigger_source)
                self.program.run_rounds(self.soc, rounds=1, load_pulses=True, start_src=self.settings.trigger_source, progress=False)

            if self.settings.trigger_source=='external':                    
                self.acquisition_completed_flag.set()
                trigger_thread.join()
            
            self._reset_bias()
            print(f"{self.address} completed execution")

        # TODO check if there are not acquisitions, if so, then run rounds instead of acquire
        # TODO test that all of the programs have the same length

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
        return {}




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
