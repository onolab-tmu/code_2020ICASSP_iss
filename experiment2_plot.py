import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Experiment where one source moves in the middle of the experiment"
    )
    parser.add_argument(
        "files", metavar="FILE", nargs="+", type=str, help="Result files"
    )
    parser.add_argument(
        "-o", "--out", type=str, default="figures", help="Directory to save figures"
    )
    args = parser.parse_args()

    # Read in the data
    all_records = []
    room_info = None
    for fn in args.files:
        print(fn)
        with open(fn, "r") as f:
            sim_results = json.load(f)
        all_records += sim_results["data"]

        if room_info is None:
            room_info = sim_results["room_info"]

    # Compute the duration of all signals use
    t_signals = [r["n_samples"] / r["fs"] for r in room_info]
    rt60 = np.array([r["rt60"] for r in room_info])

    # Create output directory if necessary
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    algo_dict = {"auxiva_laplace": "AuxIVA", "mixiva_laplace": "MixIVA"}

    df = pd.DataFrame(
        columns=[
            "Algorithm",
            "Channels",
            "SDR Raw [dB]",
            "SIR Raw [dB]",
            "\u0394SDR [dB]",
            "\u0394SIR [dB]",
            "Runtime [s]",
            "Evaltime [s]",
        ]
    )

    n_sources_perf = set()

    for record in all_records:

        if "sdr_mix" in record:
            sdr_mix = np.array(record["sdr_mix"])
            sdr_out = np.array(record["sdr_out"])
            sir_mix = np.array(record["sir_mix"])
            sir_out = np.array(record["sir_out"])

            sdr_raw = np.mean(sdr_out)
            sir_raw = np.mean(sir_out)
            d_sdr = np.mean(sdr_out - sdr_mix)
            d_sir = np.mean(sir_out - sir_mix)

            evaltime = record["evaltime"]

            # Mark this to be in the performance figure
            n_sources_perf.add(record["n_sources"])

        else:
            sdr_raw = np.nan
            sir_raw = np.nan
            d_sdr = np.nan
            d_sir = np.nan
            evaltime = np.nan

        if "runtime" in record:
            ts = t_signals[record["room_id"]]
            runtime = record["runtime"] / ts
        else:
            runtime = np.nan

        df = df.append(
            [
                {
                    "Algorithm": algo_dict[record["algo"]],
                    "Channels": record["n_sources"],
                    "SDR Raw [dB]": sdr_raw,
                    "SIR Raw [dB]": sir_raw,
                    "\u0394SDR [dB]": d_sdr,
                    "\u0394SIR [dB]": d_sir,
                    "Runtime [s]": runtime,
                    "Evaltime [s]": evaltime,
                }
            ]
        )

    df_melt = df.melt(id_vars=["Algorithm", "Channels"], var_name="Metric")

    # Figure SDR/SIR
    fn = os.path.join(args.out, f"experiment2_metrics.pdf")
    g1 = sns.catplot(
        kind="box",
        data=df_melt,
        x="Channels",
        y="value",
        order=sorted(n_sources_perf),
        col="Metric",
        col_order=["\u0394SDR [dB]", "\u0394SIR [dB]"],
        hue="Algorithm",
    )
    sns.despine(offset=10, trim=False, left=True, bottom=True)
    plt.tight_layout(pad=0.5)
    plt.savefig(fn, bbox_inches="tight")
    plt.close()

    # Figure Runtime
    fn = os.path.join(args.out, f"experiment2_runtime.pdf")
    g1 = sns.catplot(
        kind="point",
        data=df_melt,
        x="# Microphones",
        y="value",
        col="Metric",
        col_order=["Runtime [s]"],
        hue="Algorithm",
    )
    # g1.fig.get_axes()[0].set_yscale('log')
    sns.despine(offset=10, trim=False, left=True, bottom=True)
    plt.tight_layout(pad=0.1)
    plt.savefig(fn, bbox_inches="tight")
    plt.close()

    # Figure for evaluation time (bss_eval)
    fn = os.path.join(args.out, f"experiment2_evaltime.pdf")
    g1 = sns.catplot(
        kind="point",
        data=df_melt,
        x="Channels",
        y="value",
        col="Metric",
        col_order=["Evaltime [s]"],
    )
    sns.despine(offset=10, trim=False, left=True, bottom=True)
    plt.tight_layout(pad=0.1)
    plt.savefig(fn, bbox_inches="tight")
    plt.close()
    
    # Histogram of RT60
    plt.figure(figsize=(3.35, 1.8))
    plt.hist(rt60 * 1000.0)
    plt.xlabel("RT60 [ms]")
    plt.ylabel("Frequency")
    sns.despine(offset=10, trim=False, left=True, bottom=True)
    # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    fig_fn = os.path.join(args.out, f"rt60_hist.pdf")
    plt.savefig(fig_fn, bbox_inches="tight")
    plt.close()

