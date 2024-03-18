# Install Keras

First [install miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) if you haven't already.

Keras (Python package) is installed by installing Tensorflow. As of 2024, Keras is in transition from version 2 to version 3. To keep everything stable (and working with R), we're going to install version 2 (via Tensorflow version 2.13). It is necessary to use `pip install` within the conda environment, rather than `conda install`, because the conda repositories don't have later versions of the tensorflow packages for Windows.



## R users

For most people, you should be able to install everything following the official installation directions. This amounts to starting R and running this:

```R
install.packages("keras")
keras::install_keras(method = "conda", python_version = "3.10")
```

If you find this doesn't work for whatever reason, start up miniconda, and delete the conda environment (`r-tensorflow`) that was just created during the automated setup:

```bash
conda env remove --name r-tensorflow
```

Then install python, tensorflow, and related packages manually. You need to `pip install` all packages at once, otherwise later package installs could update keras and tensorflow as dependencies, which will break things.

```bash
conda create --name r-tensorflow
conda activate r-tensorflow
conda install python=3.10 -c conda-forge
python -m pip install "tensorflow==2.13.*" tensorflow-hub tensorflow-datasets scipy requests Pillow h5py pandas pydot
```



## Python users

The installation process is substantially the same but you're going to use conda and install a couple of extra packages (matplotlib, plotnine). You need to `pip install` all packages at once, otherwise later package installs could update keras and tensorflow as dependencies, which will break things.

```bash
conda create --name py-tensorflow
conda activate py-tensorflow
conda install python=3.10 -c conda-forge
python -m pip install "tensorflow==2.13.*" tensorflow-hub tensorflow-datasets scipy requests Pillow h5py pandas pydot matplotlib plotnine
```

Then you need to set your IDE to start Python out of this environment.