import json
import numpy as np
from qibo.config import raise_error, log
from qibolab.instruments.abstract import AbstractInstrument, InstrumentException


class QRM(AbstractInstrument):
    """
    Generic driver for Qblox Readout Modules.
    
    Args:
        name (str): unique name given to the instrument
        ip (str): IP address (IPv4) of the instrument

    """
    def __init__(self, name, ip):
        super().__init__(name, ip)
        self.device_class = None
        self.sequencer_channel_map = {} 
        self.last_pulsequence_hash = ""
        self.current_pulsesequence_hash = ""


    def connect(self):
        """
        Connects to the instrument using the IP address set in the runcard.
        """
        if not self.is_connected:
            try:
                self.device = self.device_class(self.name, self.ip)
            except Exception as exc:
                raise InstrumentException(self, str(exc))
            self.is_connected = True
        else:
            raise_error(Exception,'There is an open connection to the instrument already')

    # Device Property Wrappers
    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        #TODO: Add support for independent control gains for each sequencer / path
        self._gain = gain
        for n in range(self.device._num_sequencers):
            self.device.set(f"sequencer{n}_gain_awg_path0", gain)
            self.device.set(f"sequencer{n}_gain_awg_path1", gain)

    def setup(self, **kwargs):
        """
        Sets up the instrument using the parameters of the runcard. 
        A connection needs to be established before calling this method.
        
        Args:
            ref_clock (str): {'internal', 'external'} the source of the instrument clock
            sync_en (bool): {True, False} syncronise with other instruments
            scope_acq_avg_mode_en (bool): {True, False} average the results of the multiple acquisitions
            scope_acq_trigger_mode (str): {'sequencer', 'level'} 
            gain (float): {0 .. 1} the gain applied to all sequencers on their output paths
            acquisition_start (int): the delay between playing the readout pulse and the start of the acquisition (minimum 4ns)
            acquisition_duration (int): the duration of the acquisition in ns.
            mode: ssb
            channel_port_map (dict): a dictionary of {channel (int): port (str) {'o1', 'o2', ...}}
        """
        # Load settings as class attributes
        gain = kwargs.pop('gain')
        self.__dict__.update(kwargs)

        # Hardcoded values used to generate sequence program
        self.wait_loop_step = 1000
        self.waveform_max_length = 16384//2 # maximum length of the combination of waveforms, per sequencer, in number of samples (defined by the sequencer memory).

        if self.is_connected:
            # Reset
            self.device.reset()
            self.gain = gain
            # Configure clock source
            self.device.reference_source(self.ref_clock)
            for n in range(self.device._num_sequencers):
                # Configure the sequencers synchronization.
                self.device.set(f"sequencer{n}_sync_en", False)
                # Disable all sequencer - port connections
                for out in range(0, 4):
                    if hasattr(self.device, f"sequencer{n}_channel_map_path{out%2}_out{out}_en"):
                        self.device.set(f"sequencer{n}_channel_map_path{out%2}_out{out}_en", False)
                # The mapping of sequencers to ports is done in upload() as the number of sequencers needed 
                # can only be determined after examining the pulse sequence

            self.device.scope_acq_trigger_mode_path0(self.scope_acq_trigger_mode) # sets scope acquisition trigger mode for input path 0 (‘sequencer’ = triggered by sequencer, ‘level’ = triggered by input level).
            self.device.scope_acq_trigger_mode_path1(self.scope_acq_trigger_mode)
            sequencer = 0 # TODO: move to yaml?
            self.device.scope_acq_sequencer_select(sequencer) # specifies which sequencer triggers the scope acquisition when using sequencer trigger mode
            
            self.device.scope_acq_avg_mode_en_path0(self.scope_acq_avg_mode_en) # sets scope acquisition averaging mode enable for input path 0
            self.device.scope_acq_avg_mode_en_path1(self.scope_acq_avg_mode_en)

        else:
            raise_error(Exception,'There is no connection to the instrument')


    def process_pulse_sequence(self, channel_pulses, nshots):
        """
        Processes a list of pulses, generating the waveforms and sequence program required by the instrument to synthesise them.
        
        Args:
        channel_pulses (dict): a dictionary of {channel (int): pulses (list)}
        nshots (int): the number of times the sequence of pulses will be repeated
        """
        # Load the channels to which the instrument is connected
        channels = self.channel_port_map.keys()

        # Check if the sequence to be processed is the same as the last one. If so, there is no need to generate waveforms and program
        self.current_pulsesequence_hash = ""
        for channel in channels:
            for pulse in channel_pulses[channel]:
                self.current_pulsesequence_hash += pulse.serial
        if not self.current_pulsesequence_hash == self.last_pulsequence_hash:
            # Sort pulses by their start time 
            for channel in channels:
                channel_pulses[channel].sort(key=lambda pulse: pulse.start) 
            
            # Check if pulses on the same channel overlap
            for channel in channels:
                for m in range(1, len(channel_pulses[channel])):
                    channel_pulses[channel][m].overlaps = []
                    for n in range(m):
                        if channel_pulses[channel][m].start - channel_pulses[channel][n].start < channel_pulses[channel][n].duration:
                            channel_pulses[channel][m].overlaps.append(n)
                for m in range(1, len(channel_pulses[channel])):
                    if len(pulse.overlaps) > 0:
                        # TODO: Urgently needed in order to implement multiplexed readout
                        raise_error(NotImplementedError, "Overlaping pulses on the same channel are not yet supported.")

            # Allocate channel pulses to sequencers. 
            #   At least one sequencer is needed for each channel
            #   If the memory required for the sequence of pulses exceeds 16384/2, 
            #   additional sequencers are used to synthesise it
            self.sequencers = []        # a list of sequencers (int) used
            sequencer_pulses = {}       # a dictionary of {sequencer (int): pulses (list)}     
            sequencer = -1              # initialised to -1 so that in the first iteration it becomes 0, the first sequencer number
            self.waveforms = {}
            for channel in channels:
                # Select a sequencer and add it to the sequencer_channel_map
                sequencer += 1
                if sequencer > self.device._num_sequencers:
                    raise_error(Exception, f"The number of sequencers requried to play the sequence exceeds the number available {self.device._num_sequencers}.")
                # Initialise the corresponding variables 
                self.sequencers.append(sequencer)
                self.sequencer_channel_map[sequencer]=channel
                sequencer_pulses[sequencer]=[]
                self.waveforms[sequencer]={}
                unique_pulses = {}      # a dictionary of {unique pulse IDs (str): and their I & Q indices (int)}
                wc = 0                  # unique waveform counter
                waveforms_length = 0        # accumulates the length of unique pulses per sequencer

                # Iterate over the list of pulses to check if they are unique and if the overal length of the waveform exceeds the memory available
                n = 0
                while n < len(channel_pulses[channel]):
                    pulse = channel_pulses[channel][n]
                    pulse_serial = pulse.serial[pulse.serial.find(',',pulse.serial.find(',')+1)+2:-1] # removes the channel and start information from Pulse.serial to compare between pulses
                    if pulse_serial not in unique_pulses.keys():
                        # If the pulse is unique (it hasn't been saved before):
                        I, Q = self.generate_waveforms_from_pulse(pulse)
                        # Check if the overall length of the waveform exceeds memory size (waveform_max_length) and split it if necessary
                        is_split = False
                        part = 0
                        while waveforms_length + pulse.duration > self.waveform_max_length:
                            if pulse.type == 'ro':
                                raise_error(NotImplementedError, f"Readout pulses longer than the memory available for a sequencer ({self.waveform_max_length}) are not supported.")
                            import copy
                            first_part = copy.deepcopy(pulse)
                            first_part.duration = self.waveform_max_length - waveforms_length 
                            channel_pulses[channel].insert(n, first_part)
                            n += 1
                            sequencer_pulses[sequencer].append(first_part)
                            first_part.waveform_indexes = [0 + wc, 1 + wc]
                            self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_I"] = {"data": I[:first_part.duration], "index": 0 + wc}
                            self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_Q"] = {"data": Q[:first_part.duration], "index": 1 + wc}
                            
                            I, Q = I[first_part.duration:], Q[first_part.duration:]
                            pulse.start = pulse.start + self.waveform_max_length - waveforms_length
                            pulse.duration = pulse.duration - (self.waveform_max_length - waveforms_length)
                            is_split = True
                            part += 1

                            # Select a new sequencer
                            sequencer += 1
                            if sequencer > self.device._num_sequencers:
                                    raise_error(Exception, f"The number of sequencers requried to play the sequence exceeds the number available {self.device._num_sequencers}.")
                            # Initialise the corresponding variables 
                            self.sequencers.append(sequencer)
                            self.sequencer_channel_map[sequencer]=channel
                            sequencer_pulses[sequencer]=[]
                            self.waveforms[sequencer]={}
                            unique_pulses = {} 
                            wc = 0                 
                            waveforms_length = 0
 
                        # Add the pulse to the list of pulses for the current sequencer and save the waveform indexes
                        sequencer_pulses[sequencer].append(pulse)
                        pulse.waveform_indexes = [0 + wc, 1 + wc]
                        if not is_split:
                            unique_pulses[pulse_serial] =  [0 + wc, 1 + wc]
                        wc += 2                      
                        waveforms_length += pulse.duration
                        self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_I"] = {"data": I, "index": 0 + wc}
                        self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_Q"] = {"data": Q, "index": 1 + wc}

                    else:
                        sequencer_pulses[sequencer].append(pulse)
                        pulse.waveform_indexes = unique_pulses[pulse_serial]
                    n += 1

            # Generate programs for each sequencer
            pulses = sequencer_pulses
            self.acquisitions = {}
            self.program = {}
            self.weights = {}

            for sequencer in self.sequencers:
                # Acquisitions & weights
                self.acquisitions[sequencer] = {}
                self.weights[sequencer] = {}
                ac = 0 # Acquisition counter
                for pulse in pulses[sequencer]:
                    if pulse.type == 'ro':
                        pulse.acquisition_index = ac
                        pulse.num_bins = 1
                        self.acquisitions[sequencer][pulse.serial] = {"num_bins": pulse.num_bins, "index":pulse.acquisition_index}
                        ac += 1

                # Program
                sequence_total_duration = pulses[sequencer][-1].start + pulses[sequencer][-1].duration + self.minimum_delay_between_instructions # the minimum delay between instructions is 4ns
                time_between_repetitions = self.repetition_duration - sequence_total_duration
                assert time_between_repetitions > 0
                extra_wait = time_between_repetitions % self.wait_loop_step
                num_wait_loops = (time_between_repetitions - extra_wait) // self. wait_loop_step

                header = f"""
                move {nshots},R0 # nshots
                nop
                wait_sync {self.minimum_delay_between_instructions}
                loop:"""
                body = ""

                footer = f"""
                    # wait {time_between_repetitions} ns"""
                if num_wait_loops > 0:
                    footer += f"""
                    move {num_wait_loops},R1
                    nop
                    waitloop:
                        wait {self.wait_loop_step}
                        loop R1,@waitloop"""
                footer += f"""
                    wait {extra_wait}
                loop R0,@loop
                stop 
                """

                # Add an initial wait instruction for the first pulse of the sequence
                if pulses[sequencer][0].start != 0:
                        initial_wait_instruction = f"                    wait {pulses[sequencer][0].start}"
                else:
                    initial_wait_instruction = "                    # wait 0"
                body += "\n" + initial_wait_instruction

                for n in range(len(pulses[sequencer])):
                    if pulses[sequencer][n].type == 'ro':
                        delay_after_play = self.acquisition_start - self.minimum_delay_between_instructions #FIXME: This would not work for split pulses
                        
                        if len(pulses[sequencer]) > n + 1:
                            # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                            delay_after_acquire = pulses[sequencer][n + 1].start - pulses[sequencer][n].start - self.acquisition_start
                        else:
                            delay_after_acquire = sequence_total_duration - pulses[sequencer][n].start - self.acquisition_start
                            
                        if delay_after_acquire < self.minimum_delay_between_instructions:
                                raise_error(Exception, f"The minimum delay before starting acquisition is {self.minimum_delay_between_instructions}ns.")
                        
                        # Prepare play instruction: play arg0, arg1, arg2. 
                        #   arg0 is the index of the I waveform 
                        #   arg1 is the index of the Q waveform
                        #   arg2 is the delay between starting the instruction and the next instruction
                        play_instruction = f"                    play {pulses[sequencer][n].waveform_indexes[0]},{pulses[sequencer][n].waveform_indexes[1]},{delay_after_play}"
                        # Add the serial of the pulse as a comment
                        play_instruction += " "*(34-len(play_instruction)) + f"# play waveforms {pulses[sequencer][n]}" 
                        body += "\n" + play_instruction

                        # Prepare acquire instruction: acquire arg0, arg1, arg2. 
                        #   arg0 is the index of the acquisition 
                        #   arg1 is the index of the data bin
                        #   arg2 is the delay between starting the instruction and the next instruction
                        acquire_instruction = f"                    acquire {pulses[sequencer][n].acquisition_index},0,{delay_after_acquire}"
                        # Add the serial of the pulse as a comment
                        body += "\n" + acquire_instruction

                    else:
                        # Calculate the delay_after_play that is to be used as an argument to the play instruction
                        if len(pulses[sequencer]) > n + 1:
                            # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                            delay_after_play = pulses[sequencer][n + 1].start - pulses[sequencer][n].start
                        else:
                            delay_after_play = sequence_total_duration - pulses[sequencer][n].start
                            
                        if delay_after_play < self.minimum_delay_between_instructions:
                                raise_error(Exception, f"The minimum delay between pulses is {self.minimum_delay_between_instructions}ns.")
                        
                        # Prepare play instruction: play arg0, arg1, arg2. 
                        #   arg0 is the index of the I waveform 
                        #   arg1 is the index of the Q waveform
                        #   arg2 is the delay between starting the instruction and the next instruction
                        play_instruction = f"                    play {pulses[sequencer][n].waveform_indexes[0]},{pulses[sequencer][n].waveform_indexes[1]},{delay_after_play}"
                        # Add the serial of the pulse as a comment
                        play_instruction += " "*(34-len(play_instruction)) + f"# play waveforms {pulses[sequencer][n]}" 
                        body += "\n" + play_instruction

                self.program[sequencer] = header + body + footer

                # DEBUG:
                print(f"{self.name} sequencer {sequencer} program:\n" + self.program[sequencer]) 

    def generate_waveforms_from_pulse(self, pulse, modulate = True):
        """
        Generates I & Q waveforms for the pulse passed as a parameter.

        Args:
        pulse (Pulse): a Pulse object
        modulate (bool): {True, False} module the waveform with the pulse frequency

        Returns:
        envelope_i (list) of (float), envelope_q (list) of (float): a tuple with I & Q waveform samples
        """
        envelope_i = pulse.envelope_i
        envelope_q = pulse.envelope_q 
        assert len(envelope_i) == len(envelope_q)
        envelopes = np.array([envelope_i, envelope_q])
        if modulate:
            time = np.arange(pulse.duration) * self.sampling_rate
            cosalpha = np.cos(2 * np.pi * pulse.frequency * time + pulse.phase)
            sinalpha = np.sin(2 * np.pi * pulse.frequency * time + pulse.phase)
            mod_matrix = np.array([[ cosalpha, sinalpha], 
                                   [-sinalpha, cosalpha]])
            mod_signals = np.einsum("abt,bt->ta", mod_matrix, envelopes)
            return mod_signals[:, 0] + pulse.offset_i, mod_signals[:, 1] + pulse.offset_q
        else:
            return envelope_i, envelope_q

    def upload(self):
        """Uploads waveforms and programs all sequencers and arms them in preparation for execution."""
        if not self.current_pulsesequence_hash == self.last_pulsequence_hash:
            self.last_pulsequence_hash = self.current_pulsesequence_hash

            # TODO: dont upload if the same 
            # Upload waveforms and program
            qblox_dict = {}
            for sequencer in self.sequencers:
                # Reformat waveforms to lists
                for name, waveform in self.waveforms[sequencer].items():
                    if isinstance(waveform["data"], np.ndarray):
                        self.waveforms[sequencer][name]["data"] = self.waveforms[sequencer][name]["data"].tolist()  # JSON only supports lists

                # Add sequence program and waveforms to single dictionary and write to JSON file
                filename = f"{self.name}_sequencer{sequencer}_sequence.json"
                qblox_dict[sequencer] = {
                    "waveforms": self.waveforms[sequencer],
                    "weights": {}, #self.weights,
                    "acquisitions": self.acquisitions[sequencer],
                    "program": self.program[sequencer]
                    }
                with open(self.data_folder / filename, "w", encoding="utf-8") as file:
                    json.dump(qblox_dict[sequencer], file, indent=4)

                # Route sequencers to specific outputs.

                port = int(self.channel_port_map[self.sequencer_channel_map[sequencer]][1:])-1
                if hasattr(self.device, f"sequencer{sequencer}_channel_map_path0_out{2*port}_en"):
                    self.device.set(f"sequencer{sequencer}_channel_map_path0_out{2*port}_en", True)
                    self.device.set(f"sequencer{sequencer}_channel_map_path1_out{2*port+1}_en", True)
                else:
                    raise_error(Exception, f"The device does not have port {port}")
                
                self.device.set(f"sequencer{sequencer}_sync_en", self.sync_en)


                # Upload json file to the device sequencers
                self.device.set(f"sequencer{sequencer}_waveforms_and_program", str(self.data_folder / filename))
        # Arm all sequencers
        self.device.arm_sequencer()
        # DEBUG:
        # print(self.name)
        # self.device.print_readable_snapshot(update=True)

    def play_sequence(self):
        """Executes the sequence of instructions."""
        # Start playing sequences in all sequencers
        self.device.start_sequencer()

    def play_sequence_and_acquire(self):
        """Executes the sequence of instructions and retrieves the readout results.
        """
        # Start playing sequences in all sequencers
        self.device.start_sequencer()
        
        acquisition_results = {}
        # Retrieve data
        for sequencer in self.sequencers:
            # Wait for the sequencer to stop with a timeout period of one minute.
            self.device.get_sequencer_state(sequencer, timeout = 1)
            #Wait for the acquisition to finish with a timeout period of one second.
            self.device.get_acquisition_state(sequencer, timeout = 1)
            acquisition_results[sequencer] = {}
            for acquisition in self.acquisitions[sequencer]:
                #Move acquisition data from temporary memory to acquisition list.
                self.device.store_scope_acquisition(sequencer, acquisition)
                #Get acquisitions from instrument.
                raw_results = self.device.get_acquisitions(sequencer)
                i, q = self._demodulate_and_integrate(raw_results, acquisition)
                acquisition_results[sequencer][acquisition] = np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q
        
        return acquisition_results

    def _demodulate_and_integrate(self, raw_results, acquisition):
        acquisition_name = acquisition
        acquisition_frequency = 20_000_000
        #TODO: obtain from acquisition info
        #DOWN Conversion
        norm_factor = 1. / (self.acquisition_duration)
        n0 = 0 # self.acquisition_start
        n1 = self.acquisition_duration # self.acquisition_start + self.acquisition_duration
        input_vec_I = np.array(raw_results[acquisition_name]["acquisition"]["scope"]["path0"]["data"][n0: n1])
        input_vec_Q = np.array(raw_results[acquisition_name]["acquisition"]["scope"]["path1"]["data"][n0: n1])
        input_vec_I -= np.mean(input_vec_I)
        input_vec_Q -= np.mean(input_vec_Q)

        if self.mode == 'ssb':
            modulated_i = input_vec_I
            modulated_q = input_vec_Q
            time = np.arange(modulated_i.shape[0])*1e-9
            cosalpha = np.cos(2 * np.pi * acquisition_frequency * time)
            sinalpha = np.sin(2 * np.pi * acquisition_frequency * time)
            demod_matrix = 2 * np.array([[cosalpha, -sinalpha], [sinalpha, cosalpha]])
            result = []
            for it, t, ii, qq in zip(np.arange(modulated_i.shape[0]), time,modulated_i, modulated_q):
                result.append(demod_matrix[:,:,it] @ np.array([ii, qq]))
            demodulated_signal = np.array(result)
            integrated_signal = norm_factor*np.sum(demodulated_signal,axis=0)

        elif self.mode == 'optimal':
            raise_error(NotImplementedError, "Optimal Demodulation Mode not coded yet.")
        else:
            raise_error(NotImplementedError, "Demodulation mode not understood.")
        return integrated_signal


    def start(self):
        pass

    def stop(self):
        """Stops all sequencers"""
        self.device.stop_sequencer()

    def disconnect(self):
        """Disconnects from the instrument."""
        if self.is_connected:
            self.stop()
            self.device.close()
            self.is_connected = False
    
    def __del__(self):
        self.disconnect()


class QCM(AbstractInstrument):
    """
    Generic driver for Qblox Control Modules.
    
    Args:
        name (str): unique name given to the instrument
        ip (str): IP address (IPv4) of the instrument

    """
    def __init__(self, name, ip):
        super().__init__(name, ip)
        self.device_class = None
        self.sequencer_channel_map = {} 
        self.last_pulsequence_hash = ""
        self.current_pulsesequence_hash = ""


    def connect(self):
        """
        Connects to the instrument using the IP address set in the runcard.
        """
        if not self.is_connected:
            try:
                self.device = self.device_class(self.name, self.ip)
            except Exception as exc:
                raise InstrumentException(self, str(exc))
            self.is_connected = True
        else:
            raise_error(Exception,'There is an open connection to the instrument already')

    # Device Property Wrappers
    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        #TODO: Add support for independent control gains for each sequencer / path
        self._gain = gain
        for n in range(self.device._num_sequencers):
            self.device.set(f"sequencer{n}_gain_awg_path0", gain)
            self.device.set(f"sequencer{n}_gain_awg_path1", gain)

    def setup(self, **kwargs):
        """
        Sets up the instrument using the parameters of the runcard. 
        A connection needs to be established before calling this method.
        
        Args:
            ref_clock (str): {'internal', 'external'} the source of the instrument clock
            sync_en (bool): {True, False} syncronise with other instruments
            gain (float): {0 .. 1} the gain applied to all sequencers on their output paths
            channel_port_map (dict): a dictionary of {channel (int): port (str) {'o1', 'o2', ...}}
        """
        # Load settings as class attributes
        gain = kwargs.pop('gain')
        self.__dict__.update(kwargs)

        # Hardcoded values used to generate sequence program
        self.wait_loop_step = 1000
        self.waveform_max_length = 16384//2 # maximum length of the combination of waveforms, per sequencer, in number of samples (defined by the sequencer memory).

        if self.is_connected:
            # Reset
            self.device.reset()
            self.gain = gain
            # Configure clock source
            self.device.reference_source(self.ref_clock)
            for n in range(self.device._num_sequencers):
                # Configure the sequencers synchronization.
                self.device.set(f"sequencer{n}_sync_en", False)
                # Disable all sequencer - port connections
                for out in range(0, 4):
                    if hasattr(self.device, f"sequencer{n}_channel_map_path{out%2}_out{out}_en"):
                        self.device.set(f"sequencer{n}_channel_map_path{out%2}_out{out}_en", False)
                # The mapping of sequencers to ports is done in upload() as the number of sequencers needed 
                # can only be determined after examining the pulse sequence
        else:
            raise_error(Exception,'There is no connection to the instrument')

    def process_pulse_sequence(self, channel_pulses, nshots):
        """
        Processes a list of pulses, generating the waveforms and sequence program required by the instrument to synthesise them.
        
        Args:
        channel_pulses (dict): a dictionary of {channel (int): pulses (list)}
        nshots (int): the number of times the sequence of pulses will be repeated
        """
        # Load the channels to which the instrument is connected
        channels = self.channel_port_map.keys()

        # Check if the sequence to be processed is the same as the last one. If so, there is no need to generate waveforms and program
        self.current_pulsesequence_hash = ""
        for channel in channels:
            for pulse in channel_pulses[channel]:
                self.current_pulsesequence_hash += pulse.serial
        if not self.current_pulsesequence_hash == self.last_pulsequence_hash:
            # Sort pulses by their start time 
            for channel in channels:
                channel_pulses[channel].sort(key=lambda pulse: pulse.start) 
            
            # Check if pulses on the same channel overlap
            for channel in channels:
                for m in range(1, len(channel_pulses[channel])):
                    channel_pulses[channel][m].overlaps = []
                    for n in range(m):
                        if channel_pulses[channel][m].start - channel_pulses[channel][n].start < channel_pulses[channel][n].duration:
                            channel_pulses[channel][m].overlaps.append(n)
                for m in range(1, len(channel_pulses[channel])):
                    if len(channel_pulses[channel][m].overlaps) > 0:
                        raise_error(NotImplementedError, "Overlaping pulses on the same channel are not yet supported.")
            
            # Allocate channel pulses to sequencers. 
            #   At least one sequencer is needed for each channel
            #   If the memory required for the sequence of pulses exceeds 16384/2, 
            #   additional sequencers are used to synthesise it
            self.sequencers = []        # a list of sequencers (int) used
            sequencer_pulses = {}       # a dictionary of {sequencer (int): pulses (list)}     
            sequencer = -1              # initialised to -1 so that in the first iteration it becomes 0, the first sequencer number
            self.waveforms = {}
            for channel in channels:
                # Select a sequencer and add it to the sequencer_channel_map
                sequencer += 1
                if sequencer > self.device._num_sequencers:
                    raise_error(Exception, f"The number of sequencers requried to play the sequence exceeds the number available {self.device._num_sequencers}.")
                # Initialise the corresponding variables 
                self.sequencers.append(sequencer)
                self.sequencer_channel_map[sequencer]=channel
                sequencer_pulses[sequencer]=[]
                self.waveforms[sequencer]={}
                unique_pulses = {}      # a dictionary of {unique pulse IDs (str): and their I & Q indices (int)}
                wc = 0                  # unique waveform counter
                waveforms_length = 0        # accumulates the length of unique pulses per sequencer

                # Iterate over the list of pulses to check if they are unique and if the overal length of the waveform exceeds the memory available
                n = 0
                while n < len(channel_pulses[channel]):
                    pulse = channel_pulses[channel][n]
                    pulse_serial = pulse.serial[pulse.serial.find(',',pulse.serial.find(',')+1)+2:-1] # removes the channel and start information from Pulse.serial to compare between pulses
                    if pulse_serial not in unique_pulses.keys():
                        # If the pulse is unique (it hasn't been saved before):
                        I, Q = self.generate_waveforms_from_pulse(pulse)
                        # Check if the overall length of the waveform exceeds memory size (waveform_max_length) and split it if necessary
                        is_split = False
                        part = 0
                        while waveforms_length + pulse.duration > self.waveform_max_length:
                            import copy
                            first_part = copy.deepcopy(pulse)
                            first_part.duration = self.waveform_max_length - waveforms_length 
                            channel_pulses[channel].insert(n, first_part)
                            n += 1
                            sequencer_pulses[sequencer].append(first_part)
                            first_part.waveform_indexes = [0 + wc, 1 + wc]
                            self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_I"] = {"data": I[:first_part.duration], "index": 0 + wc}
                            self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_Q"] = {"data": Q[:first_part.duration], "index": 1 + wc}
                            
                            I, Q = I[first_part.duration:], Q[first_part.duration:]
                            pulse.start = pulse.start + self.waveform_max_length - waveforms_length
                            pulse.duration = pulse.duration - (self.waveform_max_length - waveforms_length)
                            is_split = True
                            part += 1

                            # Select a new sequencer
                            sequencer += 1
                            if sequencer > self.device._num_sequencers:
                                    raise_error(Exception, f"The number of sequencers requried to play the sequence exceeds the number available {self.device._num_sequencers}.")
                            # Initialise the corresponding variables 
                            self.sequencers.append(sequencer)
                            self.sequencer_channel_map[sequencer]=channel
                            sequencer_pulses[sequencer]=[]
                            self.waveforms[sequencer]={}
                            unique_pulses = {} 
                            wc = 0                 
                            waveforms_length = 0
 
                        # Add the pulse to the list of pulses for the current sequencer and save the waveform indexes
                        sequencer_pulses[sequencer].append(pulse)
                        pulse.waveform_indexes = [0 + wc, 1 + wc]
                        if not is_split:
                            unique_pulses[pulse_serial] =  [0 + wc, 1 + wc]
                        wc += 2                      
                        waveforms_length += pulse.duration
                        self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_I"] = {"data": I, "index": 0 + wc}
                        self.waveforms[sequencer][f"{self.name}_{pulse}_{part}_pulse_Q"] = {"data": Q, "index": 1 + wc}

                    else:
                        sequencer_pulses[sequencer].append(pulse)
                        pulse.waveform_indexes = unique_pulses[pulse_serial]
                    n += 1

            # Generate programs for each sequencer
            pulses = sequencer_pulses
            self.program = {}

            for sequencer in self.sequencers:
                # Program
                sequence_total_duration = pulses[sequencer][-1].start + pulses[sequencer][-1].duration + self.minimum_delay_between_instructions # the minimum delay between instructions is 4ns
                time_between_repetitions = self.repetition_duration - sequence_total_duration
                assert time_between_repetitions > 0
                extra_wait = time_between_repetitions % self.wait_loop_step
                num_wait_loops = (time_between_repetitions - extra_wait) // self. wait_loop_step

                header = f"""
                move {nshots},R0 # nshots
                nop
                wait_sync {self.minimum_delay_between_instructions}
                loop:"""
                body = ""

                footer = f"""
                    # wait {time_between_repetitions} ns"""
                if num_wait_loops > 0:
                    footer += f"""
                    move {num_wait_loops},R1
                    nop
                    waitloop:
                        wait {self.wait_loop_step}
                        loop R1,@waitloop"""
                footer += f"""
                    wait {extra_wait}
                loop R0,@loop
                stop 
                """

                # Add an initial wait instruction for the first pulse of the sequence
                if pulses[sequencer][0].start != 0:
                    initial_wait_instruction = f"                    wait {pulses[sequencer][0].start}"
                else:
                    initial_wait_instruction = "                    # wait 0"
                body += "\n" + initial_wait_instruction

                for n in range(len(pulses[sequencer])):
                    # Calculate the delay_after_play that is to be used as an argument to the play instruction
                    if len(pulses[sequencer]) > n + 1:
                        # If there are more pulses to be played, the delay is the time between the pulse end and the next pulse start
                        delay_after_play = pulses[sequencer][n + 1].start - pulses[sequencer][n].start
                    else:
                        delay_after_play = sequence_total_duration - pulses[sequencer][n].start

                    if delay_after_play < self.minimum_delay_between_instructions:
                            raise_error(Exception, f"The minimum delay between pulses is {self.minimum_delay_between_instructions}ns.")
                    
                    # Prepare play instruction: play arg0, arg1, arg2. 
                    #   arg0 is the index of the I waveform 
                    #   arg1 is the index of the Q waveform
                    #   arg2 is the delay between starting the instruction and the next instruction
                    play_instruction = f"                    play {pulses[sequencer][n].waveform_indexes[0]},{pulses[sequencer][n].waveform_indexes[1]},{delay_after_play}"
                    # Add the serial of the pulse as a comment
                    play_instruction += " "*(34-len(play_instruction)) + f"# play waveforms {pulses[sequencer][n]}" #TODO: change for split pulses
                    body += "\n" + play_instruction

                self.program[sequencer] = header + body + footer

                # DEBUG:
                print(f"{self.name} sequencer {sequencer} program:\n" + self.program[sequencer]) 


    def generate_waveforms_from_pulse(self, pulse, modulate = True):
        """
        Generates I & Q waveforms for the pulse passed as a parameter.

        Args:
        pulse (Pulse): a Pulse object
        modulate (bool): {True, False} module the waveform with the pulse frequency

        Returns:
        envelope_i (list) of (float), envelope_q (list) of (float): a tuple with I & Q waveform samples
        """
        envelope_i = pulse.envelope_i
        envelope_q = pulse.envelope_q 
        assert len(envelope_i) == len(envelope_q)
        envelopes = np.array([envelope_i, envelope_q])
        if modulate:
            time = np.arange(pulse.duration) * self.sampling_rate
            cosalpha = np.cos(2 * np.pi * pulse.frequency * time + pulse.phase)
            sinalpha = np.sin(2 * np.pi * pulse.frequency * time + pulse.phase)
            mod_matrix = np.array([[ cosalpha, sinalpha], 
                                   [-sinalpha, cosalpha]])
            mod_signals = np.einsum("abt,bt->ta", mod_matrix, envelopes)
            return mod_signals[:, 0] + pulse.offset_i, mod_signals[:, 1] + pulse.offset_q
        else:
            return envelope_i, envelope_q

    def upload(self):
        """Uploads waveforms and programs all sequencers and arms them in preparation for execution."""
        if not self.current_pulsesequence_hash == self.last_pulsequence_hash:
            self.last_pulsequence_hash = self.current_pulsesequence_hash
            # TODO: dont upload if the same 
            # Upload waveforms and program
            qblox_dict = {}
            for sequencer in self.sequencers:
                # Reformat waveforms to lists
                for name, waveform in self.waveforms[sequencer].items():
                    if isinstance(waveform["data"], np.ndarray):
                        self.waveforms[sequencer][name]["data"] = self.waveforms[sequencer][name]["data"].tolist()  # JSON only supports lists

                # Add sequence program and waveforms to single dictionary and write to JSON file
                filename = f"{self.name}_sequencer{sequencer}_sequence.json"
                qblox_dict[sequencer] = {
                    "waveforms": self.waveforms[sequencer],
                    "weights": {}, #self.weights,
                    "acquisitions": {}, #self.acquisitions,
                    "program": self.program[sequencer]
                    }
                with open(self.data_folder / filename, "w", encoding="utf-8") as file:
                    json.dump(qblox_dict[sequencer], file, indent=4)

                # Route sequencers to specific outputs.

                port = int(self.channel_port_map[self.sequencer_channel_map[sequencer]][1:])-1
                if hasattr(self.device, f"sequencer{sequencer}_channel_map_path0_out{2*port}_en"):
                    self.device.set(f"sequencer{sequencer}_channel_map_path0_out{2*port}_en", True)
                    self.device.set(f"sequencer{sequencer}_channel_map_path1_out{2*port+1}_en", True)
                else:
                    raise_error(Exception, f"The device does not have port {port}")
                
                # Configure syncronisation
                self.device.set(f"sequencer{sequencer}_sync_en", self.sync_en)

                # Upload json file to the device sequencers
                self.device.set(f"sequencer{sequencer}_waveforms_and_program", str(self.data_folder / filename))
        # Arm all sequencers
        self.device.arm_sequencer()
        # DEBUG:
        # print(self.name)
        # self.device.print_readable_snapshot(update=True)

    def play_sequence(self):
        """Executes the sequence of instructions."""
        # Start playing sequences in all sequencers
        self.device.start_sequencer()

    def start(self):
        pass

    def stop(self):
        """Stops all sequencers"""
        self.device.stop_sequencer()

    def disconnect(self):
        """Disconnects from the instrument."""
        if self.is_connected:
            self.stop()
            self.device.close()
            self.is_connected = False
    
    def __del__(self):
        self.disconnect()


class ClusterQRM(QRM):
    
    def __init__(self, name, ip):
        super().__init__(name, ip)
        from cluster.cluster import cluster_qrm
        self.device_class = cluster_qrm


class PulsarQRM(QRM):
    
    def __init__(self, name, ip):
        super().__init__(name, ip)
        from pulsar_qrm.pulsar_qrm import pulsar_qrm
        self.device_class = pulsar_qrm


class ClusterQCM(QCM):
    
    def __init__(self, name, ip):
        super().__init__(name, ip)
        from cluster.cluster import cluster_qcm
        self.device_class = cluster_qcm


class PulsarQCM(QCM):
    
    def __init__(self, name, ip):
        super().__init__(name, ip)
        from pulsar_qcm.pulsar_qcm import pulsar_qcm
        self.device_class = pulsar_qcm