![abstract](plots/graphical%20abstract.jpg)
Source: Pak, H. Y., Lin, W., & Law, A. W. K. (2024). Correction of systematic image misalignment in direct georeferencing of UAV multispectral imagery. International Journal of Remote Sensing, 46(3), 930–952. https://doi.org/10.1080/01431161.2024.2440944

# Set up

1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. If the conda default solver is slow, use the [libmamba](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) solver (optional but *recommended*)
    - `conda update -n base conda`
    - `conda install -n base conda-libmamba-solver`
    - `conda config --set solver libmamba`
3. Create a virtual environment, which will create a virtual environment with the name called *micasenseGeoreferencing*: `conda env create --file micasenseGeoreferencing.yml`
4. After all the required packages have been installed, `conda activate micasenseGeoreferencing`

# Python notebooks
The preprocessing of the Micasense UAV imagery is facilitated with the Micasense's image processing library, with some modifications to it, as included in this repository.
1. Open the `preprocessing.ipynb` to preprocess the UAV imagery
2. See `georeferencing.ipynb` for direct georeferencing of UAV imagery
3. See `alignmentError.ipynb` for aligning the systematic image misalignment in UAV imagery

## Folder organisation
Check whether the following folders and data are present after running `preprocessing.ipynb`. See instructions in `preprocessing.ipynb`
- **flight folder** (a folder that contains all the images captured from a flight, as well as meta data, and processed data used the micasense's image processing library).
- **flight_attributes** (a folder that contains the metadata of each capture from the flight).
    - *flight_attributes.csv* a csv file that is automatically generated from the modified micasense preprocessing library
    - *gps_index.txt* a text file that contains the selected gps points selected from the GUI in `algorithms/select_GPS.py`. Run `algorithms/select_GPS.py` to interactively select the start and end points of the flight swaths.
- **RawImg** (a folder that contains all the band images captured from a flight e.g. *IMG_0000_1.tif* to *IMG_xxxx_10.tif*. One can easily create a folder named "RawImg" and shift all the raw images here).
    - *IMG* is the prefix for all image captures
    - *_0000* is the index of the image (each capture should have a unique index)
    - *_1* is the band number of the capture, e.g. for a 10-band image, the band number will range from 1 - 10. So each capture will have 10 images e.g. *IMG_0000_1* to *IMG_0000_10*.
- **stacks** (a folder that contains the processed band images, where band images are all band-aligned to create a multispectral image). e.g. the postfix (as a default) always ends with *_1*, but all images in the stacks folder should have a unique image index.
- **thumbnails** (a folder that contains the thumbnail rgb of each capture, which allows for fast plotting of rgb orthomosaics)

```
flight folder (imagePath)
|
│
└───flight_attributes
│   │   flight_attributes.csv
│   │   gps_index.txt
│   
└───RawImg
|   │   IMG_0000_1.tif
|   │   IMG_0000_2.tif
|   |   ...
└───stacks
|   │   IMG_0000_1.tif
|   │   IMG_0001_1.tif
|   |   ...
└───thumbnails
    │   IMG_0000_1.jpg
    │   IMG_0001_1.jpg
    |   ...
```
