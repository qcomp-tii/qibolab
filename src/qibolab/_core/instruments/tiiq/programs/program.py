import numpy as np

from qibolab._core.identifier import ChannelId, Result
from qibolab._core.instruments.tiiq.common import OutputId, InputId
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
from qibolab._core.instruments.tiiq.scheduled import (
    ScheduledPulse,
    ScheduledAcquisition,
    ScheduledMultiplexedAcquisitions,
    ScheduledMultiplexedPulses,
    ScheduledPulseSequence,
    ScheduledItem
)
from qibolab._core.pulses import Acquisition, Align, Delay, Pulse, Readout, PulseLike
from qibolab._core.pulses.envelope import Envelope, IqWaveform, Waveform, Rectangular, Gaussian, GaussianSquare
from qibolab._core.sequence import PulseSequence

from qick.asm_v2 import AveragerProgramV2
from qick.qick_asm import QickConfig

__all__ = ["TIIqProgram"]

HZ_TO_MHZ = 1e-6
MHZ_TO_GHZ = 1e-3
NS_TO_US = 1e-3
RAD_TO_DEG = 180/np.pi

INITIALIZATION_TIME: float = 2000.0 # ns
READOUT_LAG: float = 51 # ns
FLUX_LAG: float = 22 # ns  # TODO Agustin to measure


class TIIqProgram(AveragerProgramV2):
    module_settings: TIIqSettings| None = None
    firmware_configuration: QickConfig
    scheduled_sequence: list[tuple[float, float, float, ChannelId, PulseLike]]
    sequence_total_duration: float

    
    def __init__(self, *args, **kwargs):
        self.module_settings = kwargs.pop("module_settings")
        self.firmware_configuration = kwargs.pop("firmware_configuration")
        self.scheduled_sequence = kwargs.pop("scheduled_sequence")
        self.sequence_total_duration = kwargs.pop("sequence_total_duration")

        kwargs["reps"] = kwargs.pop("nshots")
        kwargs["initial_delay"] = INITIALIZATION_TIME * NS_TO_US
        kwargs["final_delay"] = kwargs.pop("relaxation_time") * NS_TO_US
        kwargs["soccfg"] = self.firmware_configuration
        super().__init__(*args, **kwargs)

    def _initialize_drive_signal_generators(self):
        qick = self
        # drive
        # declare signal generators (axis_sg_int4_v2)
        sg: InterpolatedSignalGenerator
        for sg in self.module_settings.drive_signal_generators.values():
            cfg: dict = sg.cfg
            f_dds: float = cfg['f_dds']
            sampling_rate: float = cfg['fs']
            f_fabric: float = cfg['f_fabric']
            interpolation: int = cfg['interpolation']
            maxv: int = int(cfg['maxv'] * cfg['maxv_scale'])

            qick.declare_gen(
                    ch=sg.ch,
                    mixer_freq=sg.digital_mixer_frequency * HZ_TO_MHZ,
                    nqz=1 if sg.frequency * HZ_TO_MHZ < sampling_rate / 2 else 2
                )

            # envelopes and pulses
            pulse: Pulse
            for pulse in sg.pulses:
                if isinstance(pulse.envelope, Rectangular):
                    qick.add_pulse(
                        ch=sg.ch,
                        freq=sg.frequency * HZ_TO_MHZ, 
                        # ro_ch=ro_chs[0], 
                        name=pulse.id,
                        style="const", 
                        length=pulse.duration * NS_TO_US,
                        phase=pulse.relative_phase * RAD_TO_DEG,
                        gain=pulse.amplitude, 
                    )
                else:
                    num_samples: int = int(pulse.duration*NS_TO_US*f_fabric)
                    qick.add_envelope(
                        ch=sg.ch,
                        name=pulse.id,
                        idata=pulse.envelope.i(num_samples) * maxv,
                        qdata=pulse.envelope.q(num_samples) * maxv
                    )
                    # qick.add_gauss(
                    #     ch=sg.dac,
                    #     name=pulse.id,
                    #     sigma=pulse.duration*NS_TO_US/5,
                    #     length=pulse.duration*NS_TO_US,
                    #     even_length=True
                    # )
                    qick.add_pulse(
                        ch=sg.ch,
                        freq=sg.frequency * HZ_TO_MHZ, 
                        # ro_ch=ro_ch,  
                        name=pulse.id, 
                        style="arb", 
                        envelope=pulse.id,
                        phase=pulse.relative_phase,
                        gain=pulse.amplitude, 
                    )

    def _initialize_flux_signal_generators(self):
        qick = self
        # flux
        # declare signal generators (axis_signal_gen_v6)
        sg: FullSpeedSignalGenerator
        for sg in self.module_settings.flux_signal_generators.values():
            gencfg: dict = self.firmware_configuration['gens'][sg.ch]
            f_dds: float = gencfg['f_dds']
            sampling_rate: float = gencfg['fs']
            f_fabric: float = gencfg['f_fabric']
            interpolation: int = gencfg['interpolation']
            maxv: int = int(gencfg['maxv'] * gencfg['maxv_scale'])

            qick.declare_gen(
                    ch=sg.ch,
                    nqz=1 if sg.frequency * HZ_TO_MHZ < sampling_rate / 2 else 2
                )
            # envelopes and pulses
            pulse: Pulse
            for pulse in sg.pulses:
                if isinstance(pulse.envelope, Rectangular):
                    qick.add_pulse(
                        ch=sg.ch,
                        freq=sg.frequency * HZ_TO_MHZ, 
                        # ro_ch=ro_chs[0], 
                        name=pulse.id,
                        style="const", 
                        length=pulse.duration * NS_TO_US,
                        phase=pulse.relative_phase * RAD_TO_DEG,
                        gain=pulse.amplitude, 
                    )
                else:
                    num_samples: int = int(pulse.duration*NS_TO_US*f_fabric)
                    qick.add_envelope(
                        ch=sg.ch,
                        name=pulse.id,
                        idata=pulse.envelope.i(num_samples) * maxv,
                        qdata=pulse.envelope.q(num_samples) * maxv
                    )
                    qick.add_pulse(
                        ch=sg.ch,
                        freq=sg.frequency * HZ_TO_MHZ, 
                        # ro_ch=ro_ch,  
                        name=pulse.id, 
                        style="arb", 
                        envelope=pulse.id,
                        phase=pulse.relative_phase,
                        gain=pulse.amplitude, 
                    )

    def _initialize_probe_signal_generators(self):
        qick = self
        # readout
        # declare signal generators (axis_sg_mixmux8_v1)
        sg: MuxedSignalGenerator
        for output, sg in self.module_settings.probe_signal_generators.items():
            
            gencfg: dict = self.firmware_configuration['gens'][sg.ch]
            f_dds: float = gencfg['f_dds']
            sampling_rate: float = gencfg['fs']
            f_fabric: float = gencfg['f_fabric']
            interpolation: int = gencfg['interpolation']
            maxv: int = int(gencfg['maxv'] * gencfg['maxv_scale'])
            nqz = 1 if sg.digital_mixer_frequency * HZ_TO_MHZ < sampling_rate / 2 else 2
            nqzs = [1 if freq * HZ_TO_MHZ < sampling_rate / 2 else 2 for freq in sg.get_mux_freqs(include_default=False)]
            if len(set(nqzs)) != 1 or nqz != nqzs[0]:
                raise ValueError("TODO")
            qick.declare_gen(
                    ch=sg.ch,
                    nqz=nqz,
                    ro_ch=sg.ro_ch,  
                    mux_freqs=[freq * HZ_TO_MHZ for freq in sg.get_mux_freqs()],
                    mux_gains=[gain for gain in sg.get_mux_gains()],
                    mux_phases=[phase * RAD_TO_DEG for phase in sg.get_mux_phases()],
                    mixer_freq=sg.digital_mixer_frequency * HZ_TO_MHZ, 
                )

            # envelopes and pulses
            pulse: ScheduledMultiplexedPulses
            for pulse in sg.pulses:
                qick.add_pulse(
                    ch=sg.ch,
                    name=pulse.id,
                    style="const", 
                    length=pulse.duration * NS_TO_US,
                    mask=sg.mask
                )
            
    def _initialize_acquisition_signal_processors(self):
        qick = self
        # declare signal processors (axis_pfb_readout_v4)
        sp: MuxedSignalProcessor
        for sp in self.module_settings.acquisition_signal_processors.values():
            tone: MuxedSignalProcessorTone
            for tone in sp.tones.values():
                cfg: dict = tone.cfg
                qick.declare_readout(
                    ch=tone.ch,
                    length=tone.duration * NS_TO_US,
                    freq=tone.frequency * HZ_TO_MHZ,
                    sel='product',
                    phase=tone.phase * RAD_TO_DEG,
                    gen_ch=tone.dac,
                )

    def _initialize_sweepers(self):
        for name, nsteps in self.module_settings.rt_sweepers:
            self.add_loop(name, nsteps)

    def _play_drive_pulse(self, channel_id: ChannelId, pulse: Pulse):
        qick = self
        # axis_sg_int4_v2
        sg: InterpolatedSignalGenerator
        sg = self.module_settings.drive_signal_generators[channel_id]
        qick.pulse(ch=sg.ch, name=pulse.id, t=0)

    def _play_flux_pulse(self, channel_id: ChannelId, pulse: Pulse):
        qick = self
        # axis_signal_gen_v6
        sg: FullSpeedSignalGenerator
        sg = self.module_settings.flux_signal_generators[channel_id]
        qick.pulse(ch=sg.ch, name=pulse.id, t=FLUX_LAG * NS_TO_US)

    def _play_probe_pulses(self, output: OutputId, pulse: Pulse):
        qick = self
        # axis_sg_mixmux8_v1
        sg: MuxedSignalGenerator
        sg = self.module_settings.probe_signal_generators[output]
        qick.pulse(ch=sg.ch, name=pulse.id, t=READOUT_LAG * NS_TO_US)

    def _trigger_acquisition(self, acquisition: ScheduledMultiplexedAcquisitions):
        qick = self
        input: InputId = acquisition.input
        sp: MuxedSignalProcessor = self.module_settings.signal_processors[input]
        ro_chs: list[int] = [tone.ch for tone in sp.tones.values()]
        qick.trigger(ros=ro_chs, t=READOUT_LAG * NS_TO_US)

    def _initialize(self, cfg:dict):
        qick = self
        self._initialize_drive_signal_generators()
        self._initialize_flux_signal_generators()
        self._initialize_probe_signal_generators()
        self._initialize_acquisition_signal_processors()
        self._initialize_sweepers()

    def _body(self, cfg:dict):
        qick = self
        start: float
        duration: float
        scheduled_item: ScheduledItem
        for scheduled_item in self.scheduled_sequence:
            lag: float = scheduled_item.lag
            if lag != 0.0:
                qick.delay(lag * NS_TO_US)
            if isinstance(scheduled_item, ScheduledPulse):
                pulse: ScheduledPulse = scheduled_item
                channel_id: ChannelId = scheduled_item.channel_id
                if "drive" in channel_id:
                    self._play_drive_pulse(channel_id, pulse)
                elif "flux" in channel_id:
                    self._play_flux_pulse(channel_id, pulse)
                else:
                    raise ValueError("TODO")
            elif isinstance(scheduled_item, ScheduledMultiplexedPulses):
                pulse: ScheduledMultiplexedPulses = scheduled_item
                output: OutputId = pulse.output
                self._play_probe_pulses(output, pulse)
            elif isinstance(scheduled_item, ScheduledMultiplexedAcquisitions):
                multiplexed_acquisition: ScheduledMultiplexedAcquisitions = scheduled_item
                self._trigger_acquisition(multiplexed_acquisition)
        if (self.sequence_total_duration - self.scheduled_sequence.duration) != 0: # TODO allow for approximate values
            qick.delay_auto(t=(self.sequence_total_duration - self.scheduled_sequence.duration - 46) * NS_TO_US, gens=True, ros=True)
            # TODO find out where this 45 comes from. 45 and 46 render the same results so I suspect is related to the 
            # number ot tProc instructions. How can we estimate them. assmebly duration?
            # That doesn't work!!
            pass