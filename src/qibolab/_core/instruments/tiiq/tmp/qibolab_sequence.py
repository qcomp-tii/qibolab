import numpy as np
import matplotlib.pyplot as plt
import os 

from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
    PulseSequence,
    Pulse,
    Delay,
    Rectangular,
    Gaussian,
    Sweeper,
    create_platform,
)
PLATFORMS = "QIBOLAB_PLATFORMS"
os.environ[PLATFORMS] = "qibolab_platforms_qrc"
platform = create_platform("spinq10q_tiiq")

q1 = platform.qubits['q1']
q6 = platform.qubits['q6']
nativesq1 = platform.natives.single_qubit['q1']
nativesq6 = platform.natives.single_qubit['q6']

sequence = PulseSequence.load(
    [
        (
            "q2/flux",
            Pulse(
                amplitude=0.5, duration=16*4, relative_phase=0, envelope=Rectangular()
            ),
        ),
        (   "q2/flux", Delay(duration=32)),
        (
            "q2/flux",
            Pulse(
                amplitude=0.5, duration=16*4, relative_phase=0, envelope=Rectangular()
            ),
        ),
        # ("q6/flux", Delay(duration=100)),
        (
            "q6/flux",
            Pulse(
                amplitude=0.5, duration=16*10, relative_phase=0, envelope=Rectangular()
            ),
        ),
    ]
)


# # define a sweeper for a frequency scan
# f0 = platform.config(qubit.probe).frequency  # center frequency
# sweeper = Sweeper(
#     parameter=Parameter.frequency,
#     range=(f0 - 2e8, f0 + 2e8, 1e6),
#     channels=[qubit.probe],
# )
platform.connect()
for n in range(1000):

    # # perform the experiment using specific options
    results = platform.execute(
        [sequence],
        [],
        nshots=10000,
        relaxation_time=2000,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.INTEGRATION,
    )
# _, acq = next(iter(sequence.acquisitions))

# # plot the results
# signal = results[acq.id]
# amplitudes = signal[..., 0] + 1j * signal[..., 1]
# frequencies = sweeper.values

# plt.title("Resonator Spectroscopy")
# plt.xlabel("Frequencies [Hz]")
# plt.ylabel("Amplitudes [a.u.]")

# plt.plot(frequencies, amplitudes)