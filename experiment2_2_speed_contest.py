import argparse
import json
import time

import numpy as np
import pyroomacoustics as pra
from room_builder import random_room_builder
from bsseval.bsseval.metrics import bss_eval
from mixiva.mixiva import auxiva_cpp, mixiva_cpp
from samples.generate_samples import sampling, wav_read_center

# Simulation parameters
config = {
    "n_repeat": 100,
    "seed": 840808,
    "snr": 30,
    "n_sources_list": [11, 12, 13, 14, 15, 16, 17],
    "algorithms": {
        "auxiva_laplace": {"name": "auxiva", "kwargs": {"model": "laplace"}},
        "mixiva_laplace": {"name": "mixiva", "kwargs": {"model": "laplace"}},
    },
    "separation_params": {"ref_mic": 0, "n_iter_multiplier": 10},
    "stft_params": {"n_fft": 4096, "hop": 2048, "win": "hamming"},
    "samples_metadata": "samples/metadata.json",
    "room_params": {
        "mic_delta": 0.02,
        "fs": 16000,
        "t60_interval": [0.150, 0.350],
        "room_width_interval": [6, 10],
        "room_height_interval": [2.8, 4.5],
        "source_zone_height": [1.0, 2.0],
        "guard_zone_width": 0.5,
    },
    "output_file": "experiment2_2_results.json",
}

# Placeholder to hold all the results
sim_results = {"config": config, "room_info": [], "data": []}


if __name__ == "__main__":

    np.random.seed(config["seed"])

    min_sources = np.min(config["n_sources_list"])
    max_sources = np.max(config["n_sources_list"])
    audio_files = sampling(config["n_repeat"], min_sources, config["samples_metadata"])
    ref_mic = config["separation_params"]["ref_mic"]

    for room_id, file_list in enumerate(audio_files):

        audio = wav_read_center(file_list)

        # Get a random room and simulate
        room, rt60 = random_room_builder(audio, max_sources, **config["room_params"])
        premix = room.simulate(return_premix=True)
        sim_results["room_info"].append(
            {
                "dim": room.shoebox_dim.tolist(),
                "rt60": rt60,
                "id": room_id,
                "samples": file_list,
                "n_samples": premix.shape[2],
                "fs": room.fs,
            }
        )

        # normalize all sources at the reference mic
        premix /= np.std(premix[:, ref_mic, None, :], axis=2, keepdims=True)

        for n_sources in config["n_sources_list"]:

            # Do the mix and add noise
            mix = np.sum(premix[:, :n_sources, :], axis=0)
            noise_std = 10 ** (-config["snr"] / 20) * np.std(mix[ref_mic, :])
            mix += noise_std * np.random.randn(*mix.shape)
            ref = premix[:n_sources, ref_mic, :]

            # STFT
            n_fft = config["stft_params"]["n_fft"]
            hop = config["stft_params"]["hop"]
            if config["stft_params"]["win"] == "hamming":
                win_a = pra.hamming(n_fft)
            else:
                raise ValueError("Undefined window function")
            win_s = pra.transform.compute_synthesis_window(win_a, hop)

            X = pra.transform.analysis(mix.T, n_fft, hop, win=win_a)

            n_iter = config["separation_params"]["n_iter_multiplier"] * n_sources

            # Separation
            for algo, details in config["algorithms"].items():

                t1 = time.perf_counter()

                if details["name"] == "auxiva":
                    Y = auxiva_cpp(
                        X, proj_back=False, n_iter=n_iter, **details["kwargs"]
                    )
                elif details["name"] == "mixiva":
                    Y = mixiva_cpp(
                        X, proj_back=False, n_iter=n_iter, **details["kwargs"]
                    )

                t2 = time.perf_counter()
                sep_time = t2 - t1

                # store the results
                sim_results["data"].append(
                    {
                        "algo": algo,
                        "room_id": room_id,
                        "n_sources": n_sources,
                        "runtime": sep_time,
                    }
                )

                print(
                    f"{room_id} {n_sources} {algo} {sep_time:.3f}"
                )

            # Save to file regularly
            with open(config["output_file"], "w") as f:
                json.dump(sim_results, f)
