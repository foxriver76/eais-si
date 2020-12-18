# EAIS 2020 Special Issue

## How To execute the Experiments
You need Python >= 3.6, the latest version of [skmultiflow](https://github.com/scikit-multiflow/scikit-multiflow) installed as well as the Incremental PCA, which you can find below.

### Experiments on Synthetic Streams
- execute `python pca_streams.py` inside the `experiments` folder to reproduce the PCA experiments
- execute `python rp_streams.py` inside the `experiments` folder to reproduce the Random Projection experiments

### Experiments on the NSDQ Dataset
- execute `python pca_nasdaq_skipgram.py` inside the `experiments` folder to reproduce the PCA experiments via Skipgram embedding
- execute `python pca_nasdaq_tf-idf.py` inside the `experiments` folder to reproduce the PCA experiments via tf-idf encoding
- execute `python rp_nasdaq.py` inside the `experiments` folder to reproduce the Random Projection experiments for Skipgram as well as tf-idf

Requirements for Incremental PCA
-----
* C++11 compiler, Python3, Eigen3, Pybind11, Numpy

* Note: Tested on macOS Mojave and Ubuntu 19.0.4 LTS.

******

Setup for Incremental PCA
-----
#### Mac OS with Homebrew
* Install libraries

    `brew install python3`

    `brew install eigen`

    `brew install pybind11`

    `pip3 install numpy`

* Build with cmake

    `cmake .`

    `make`

* This generates a shared library, "inc_pca_cpp.xxxx.so" (e.g., inc_pca_cpp.cpython-37m-darwin.so).

* Install the modules with pip3.

    `pip3 install .`

#### Linux (tested on Ubuntu 19.0.4 LTS)
* Install libraries

    `sudo apt update`

    `sudo apt install libeigen3-dev`

    `sudo apt install python3-pip python3-dev`

    `pip3 install pybind11`

    `pip3 install numpy`

*  Compile with:

    ``c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -I/usr/include/eigen3/ -fPIC `python3 -m pybind11 --includes` inc_pca.cpp inc_pca_wrap.cpp -o inc_pca_cpp`python3-config --extension-suffix` ``

* This generates a shared library, "inc_pca_cpp.xxxx.so" (e.g., inc_pca_cpp.cpython-37m-x86_64-linux-gnu.so).

* Install the modules with pip3.

    `pip3 install .`

******



******

Special Thanks goes to Fujiwara et al. for providing the Incremenatal Streaming PCA 
For more information see:
-----
* Incremental PCA for visualizing streaming multidimensional data from:    
***An Incremental Dimensionality Reduction Method for Visualizing Streaming Multidimensional Data***    
Takanori Fujiwara, Jia-Kai Chou, Shilpika, Panpan Xu, Liu Ren, and Kwan-Liu Ma   
IEEE Transactions on Visualization and Computer Graphics and IEEE VIS 2019 (InfoVis).
DOI: [10.1109/TVCG.2019.2934433](https://doi.org/10.1109/TVCG.2019.2934433)
