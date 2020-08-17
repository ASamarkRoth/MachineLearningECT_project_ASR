[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/)

# MachineLearningECT - AT-TPC project - Anton Såmark-Roth


## Running the Jupyter-Notebook

You can run it in the web browser on mybinder (without installing anything) by clicking the link [here](https://mybinder.org/) (ignore the following in that case). 

It is possible to run the notebook on your local computer as follows:

1. Install [miniconda3](https://conda.io/miniconda.html) alternatively the full [anaconda3](https://www.anaconda.com/download) environment on your laptop (the latter is **much** larger).
2. [Download]() this repository.
3. Install and activate the `machine_learning` environment described by the file [`environment.yml`](/environment.yml)  by running the following in a terminal:

```bash
conda env create -f environment.yml
source activate machine_learning
./postBuild
```
4. Run the notebook via `jupyter-lab`

It is preferable to further configure _nbstripout_ for the git repo. If active, this program strips the notebook from the outputs and makes it easier for collaboration and merging. It is performed as follows: 

```bash
nbstripout --install
```

Note: it is however placed in the `postBuild` file.


## Useful links

Lectures on active targets: https://nucleartalent.github.io/MachineLearningECT/doc/pub/Day8/html/Day8.html