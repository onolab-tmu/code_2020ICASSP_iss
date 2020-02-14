import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from mixiva.mixiva import auxiva, mixiva
from room_builder import callback_noise_mixer
from samples.generate_samples import sampling, wav_read_center

# a useful constant
twodsqtwo = 2.0 / np.sqrt(2)
inv_sq_2 = 1.0 / np.sqrt(2)

# The source locations (0., 90, 180, then 165 degrees)
center = np.array([[4.5, 2.1, 1.7]]).T
source_angles = np.array([0., np.pi / 2, np.pi, np.pi - np.pi / 12])
source_distance = 1.5
S = np.array([np.cos(source_angles), np.sin(source_angles), [-0.03, 0.01, 0.02, 0.02]])
S[:2, :] *= source_distance
S += center
n_sources = S.shape[1] - 1

# the mic locations
mic_angles = np.array([0., np.pi / 2., np.pi])
mic_delta = 0.02
R = np.array([np.cos(mic_angles), np.sin(mic_angles), [0., 0., 0.]])
n_mics = R.shape[1]
radius = 0.5 * mic_delta / np.sin(np.pi / n_mics)
R[:2, :] *= radius
R += center

# The move vector of source 2
move_vector = np.array([1.5, 0, 0])


if __name__ == "__main__":

    model_choices = ["laplace", "gauss"]

    parser = argparse.ArgumentParser(
        description="Experiment where one source moves in the middle of the experiment"
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=12345,
        help="Seed for the random number generator",
    )
    parser.add_argument(
        "--n_iter", "-n", type=int, default=20, help="Number of iterations"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=model_choices[0],
        choices=model_choices,
        help="Source model",
    )
    args = parser.parse_args()

    # STFT parameters
    n_fft = 4096
    hop = n_fft // 2
    win_a = pra.hamming(n_fft)
    win_s = pra.transform.compute_synthesis_window(win_a, hop)

    # Separation parameters
    ref_mic = 1  # center mic
    n_iter = 30

    np.random.seed(args.seed)

    # Create the room
    room_dim = [10, 6, 3.5]
    room = pra.ShoeBox(room_dim, fs=16000, absorption=0.45, max_order=8)

    # Prepare the audio
    audio_sources = sampling(2, n_sources, "samples/metadata.json")
    audio_1 = wav_read_center(audio_sources[0])
    audio_2 = wav_read_center(audio_sources[0])
    audio = np.zeros(
        (n_sources + 1, audio_1.shape[1] + audio_2.shape[1]), dtype=audio_1.dtype
    )
    t_move = audio_1.shape[1]
    audio[[0, 1, 2], :t_move] = audio_1
    audio[[0, 1, 3], t_move:] = audio_2

    for loc, signal in zip(S.T, audio):
        room.add_source(loc, signal=signal)

    # add the mic array
    mic_array = pra.MicrophoneArray(R, room.fs)
    room.add_microphone_array(mic_array)

    # Simulate
    cb_kwargs = {
        "ref_mic": ref_mic,
        "sinr": 20,
        "n_tgt": n_sources + 1,
        "n_src": n_sources + 1,
        "tgt_std": np.array([1.0, 1.0, inv_sq_2, inv_sq_2]),
    }
    premix = room.simulate(
        return_premix=True,
        callback_mix=callback_noise_mixer,
        callback_mix_kwargs=cb_kwargs,
    )

    # Separation Performance Tracking
    def convergence_callback(Y, epoch, X, SDR, SIR, permutation, part):
        Y = Y.copy()

        # projection back
        z = pra.bss.projection_back(Y, X[:, :, ref_mic])
        Y = Y * np.conj(z[None, :, :])

        from mir_eval.separation import bss_eval_sources

        if Y.shape[2] == 1:
            y = pra.transform.synthesis(Y[:, :, 0], n_fft, hop, win=win_s)[:, None]
        else:
            y = pra.transform.synthesis(Y, n_fft, hop, win=win_s)
        y = y[n_fft - hop :, :].T

        if part == 1:
            ref_loc = premix[[0, 1, 2], ref_mic, :t_move]
        else:
            ref_loc = premix[[0, 1, 3], ref_mic, t_move:]

        m = np.minimum(y.shape[1], ref_loc.shape[1])
        sdr, sir, sar, perm = bss_eval_sources(ref_loc[:, :m], y[:, :m])

        SDR.append(sdr.tolist())
        SIR.append(sir.tolist())
        permutation.append(perm.tolist())

    sdr_auxiva = []
    sir_auxiva = []
    perm_auxiva = []
    sdr_mixiva = []
    sir_mixiva = []
    perm_mixiva = []

    # Split the two halfs
    part1 = room.mic_array.signals[:, :t_move]
    part2 = room.mic_array.signals[:, t_move:]

    # STFT
    X1 = pra.transform.analysis(part1.T, n_fft, hop, win=win_a)
    X2 = pra.transform.analysis(part2.T, n_fft, hop, win=win_a)

    # Separate the first half
    def cb_a_1(Y, epoch):
        convergence_callback(Y, epoch, X1, sdr_auxiva, sir_auxiva, perm_auxiva, 1)

    Y1_auxiva, W_auxiva = auxiva(
        X1,
        n_iter=args.n_iter,
        model=args.model,
        proj_back=False,
        return_filters=True,
        callback=cb_a_1,
    )

    def cb_m_1(Y, epoch):
        convergence_callback(Y, epoch, X1, sdr_mixiva, sir_mixiva, perm_mixiva, 1)

    Y1_mixiva, W_mixiva = mixiva(
        X1,
        n_iter=args.n_iter,
        model=args.model,
        proj_back=False,
        return_filters=True,
        callback=cb_m_1,
    )

    # Separate the second half
    def cb_a_2(Y, epoch):
        convergence_callback(Y, epoch, X2, sdr_auxiva, sir_auxiva, perm_auxiva, 2)

    Y2_auxiva = auxiva(
        X2,
        n_iter=args.n_iter,
        model=args.model,
        proj_back=False,
        W0=W_auxiva,
        callback=cb_a_2,
        single_source=perm_auxiva[-1][-1],
    )

    def cb_m_2(Y, epoch):
        convergence_callback(Y, epoch, X2, sdr_mixiva, sir_mixiva, perm_mixiva, 2)

    Y2_mixiva = mixiva(
        X2,
        n_iter=args.n_iter,
        model=args.model,
        proj_back=False,
        W0=W_mixiva,
        callback=cb_m_2,
        single_source=perm_mixiva[-1][-1],
    )

    # Save the data
    results = {
        "auxiva": {
            "sdr": np.array(sdr_auxiva).tolist(),
            "sir": np.array(sir_auxiva).tolist(),
        },
        "mixiva": {
            "sdr": np.array(sdr_mixiva).tolist(),
            "sir": np.array(sir_mixiva).tolist(),
        },
    }
    with open("experiment1_results.json", "w") as f:
        json.dump(results, f)

    # Make the figure
    def plot(sdr, label):
        sdr = np.array(sdr)
        plt.plot(np.arange(sdr.shape[0]), sdr, label=label)

    plot(sdr_auxiva, "AuxIVA")
    plot(sdr_mixiva, "MixIVA")
    plt.legend()
    plt.show()
