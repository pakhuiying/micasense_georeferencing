[![Lifecycle:Experimental](https://img.shields.io/badge/Lifecycle-Experimental-339999)

# Get training data for sunglint_correction

1. cd to directory where the folder is cloned/downloaded

2. Run `get_training_data.py`

3. Go to directory where raw images are stored (Copy path of the folder)

4. A prompt will appear to ask you to enter the directory of the image: paste the path of the folder

5. Wait for a few seconds to load the images

- The first image loaded should show the calibration panel image

- Images have been band-aligned to produce the reflectance image. However, for images taken on the ground and below 35m AGL (e.g. panel image), the band images will not be aligned properly because rig relatives are used to perform the alignment, and the alignment will only work best for images taken > 35m AGL.

- In this case, do not draw boxes (bboxes) on images that are misaligned

- When you click on the “Next” button, it will take a few seconds to load the next image. Be patient and wait.

## Introducing buttons

Buttons on the right helps in toggling the mode of selection (drawing bboxes) 

- `turbid_glint`: to identify regions with turbid waters AND glint
- `water_glint`: to identify regions with no turbidity AND glint
- `turbid`: to identify regions with turbid waters with NO glint
- `water`: to identify regions with no turbid waters with NO glint
- `shore`: to identify regions of shoreline near the water

- `Reset`: clear all the bboxes drawn

- `Prev`: go to the previous image

- `Next`: go to the next image (bboxes and plots will be automatically saved to *saved_bboxes* and *saved_plots* when `Next` is clicked)

- `Save`: save the current bboxes

- `IMG_index`: Enter the image index number in the text box to jump immediately to that image, instead of having to repeatedly click `Next`


## Example on how to use `get_training_data.py`

The colored boxes each correspond to the different categories. Here is an example on how to label the images:

![gui1](example_images/gui3.png)

![gui2](example_images/gui4.png)

Read `SOP_identify_glint.pdf` for more information on using `get_training_data.py`