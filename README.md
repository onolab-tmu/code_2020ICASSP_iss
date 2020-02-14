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

    git clone --recursive <url>
    cd piva
    conda env create -f environment.yml
    conda activate piva
    python setup.py build_ext --inplace
    cd ..

Run Experiments
---------------

    # Run the simulations
    python ./experiment_metrics_speed.py
    python ./experiment_speed_11_17.py

    # Plot the results
    python ./make_figures.py ./experiment_metrics_speed_results.json ./experiment_speed_11_17_results.json
