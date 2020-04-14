Fast and stable blind source separation with rank-1 updates
===========================================================

Abstract
--------

We propose a new algorithm for the blind source separation of acoustic
sources. This algorithm is an alternative to the popular auxiliary function
based independent vector analysis using iterative projection (AuxIVA-IP). It
optimizes the same cost function, but instead of alternate updates of the rows
of the demixing matrix, we propose a sequence of rank-1 updates. Remarkably,
and unlike the previous method, the resulting updates do not require matrix
inversion. Moreover, their computational complexity is quadratic in the
number of microphones, rather than cubic in AuxIVA-IP. In addition, we show
that the new method can be derived as alternate updates of the steering
vectors of sources. Accordingly, we name the method iterative source steering
(AuxIVA-ISS). Finally, we confirm in simulated experiments that the proposed
algorithm separates sources just as well as AuxIVA-IP, at a lower computational
cost.

Authors
-------

* [Robin Scheibler](http://www.robinscheibler.org)
* [Nobutaka Ono](http://www.comp.sd.tmu.ac.jp/onolab/index-e.html)

Install
-------

We use [anaconda](https://www.anaconda.com/distribution/) to setup the Python environment.

    git clone --recursive <url>
    cd piva
    conda env create -f environment.yml
    conda activate piva
    python setup.py build_ext --inplace
    cd ..

Run Experiments
---------------

The two experiments presented in the paper can be run by the following steps.
This produces two files `./experiment_metrics_speed_results.json` `./experiment_speed_11_17_results.json` that are later used to produce the plots.

    conda activate piva

    # Run the simulations
    python ./experiment_metrics_speed.py
    python ./experiment_speed_11_17.py

    # Plot the results
    python ./make_figures.py ./experiment_metrics_speed_results.json ./experiment_speed_11_17_results.json

The two simulation output data files produced for the figures in the paper were kept in the `sim_results` folder.

License
-------

The code in this repository is released under the [MIT license](https://opensource.org/licenses/MIT).
