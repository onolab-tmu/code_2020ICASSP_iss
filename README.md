Code ICASSP 2020: Mixing matrix updates
=======================================

Install
-------

    git clone --recursive <url>
    cd mixiva
    conda env create -f environment.yml
    conda activate mixiva
    python setup.py build_ext --inplace
    cd ..

Run Experiments
---------------

    python ./experiment1_moving_source.py
    python ./experiment2_speed_contest.py
