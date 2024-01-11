# Set up

1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. If the conda default solver is slow, use the [libmamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) solver (optional but *recommended*)
    - `conda update -n base conda`
    - `conda install -n base conda-libmamba-solver`
    - `conda config --set solver libmamba`
3. Create a virtual environment, which will create a virtual environment with the name called *micasenseGeoreferencing*: `conda env create --file micasenseGeoreferencing.yml`
4. After all the required packages have been installed, `conda activate micasenseGeoreferencing`
5. Open the [TODO: notebook] and try it out! Test images are provided in [TODO]
