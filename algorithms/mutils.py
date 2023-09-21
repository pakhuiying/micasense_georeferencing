import numpy as np
import os
import pickle #This library will maintain the format as well
import micasense.imageutils as imageutils
import micasense.plotutils as plotutils
import micasense.capture as capture
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import matplotlib.patches as patches
import json
import glob
import shutil

panel_radiance_to_irradiance = lambda radiance,albedo: radiance*np.pi/albedo

def order_bands_from_filenames(imageNames):
    """ 
    listing images using glob.glob results in unordered band order (i.e. band 1, 10, 2,3,4,5,6,7,8,9)
    this function ensures that filenames are listed in band order i.e. band 1,2,3,4,5,6,7,8,9,10
    """
    imageNames_ordered = {i+1: None for i in range(10)}
    for fn in imageNames:
        filename = os.path.basename(fn)
        imageNames_ordered[int(filename.split('_')[-1].replace('.tif',''))] = fn
    return list(imageNames_ordered.values())

def load_pickle(fp):
    """
    :param fp (str): absolute filepath of the pickle file
    """
    if fp.endswith('ob'):
        with open(fp, 'rb') as fp:
            data = pickle.load(fp)

        return data
    else:
        print("Not a pickle file")
        return None

def sort_bands_by_wavelength():
    """ import center_wavelengths_by_band.ob and sort"""
    wavelengths = load_pickle(r"saved_data\center_wavelengths_by_band.ob")
    wavelengths = [(i,w) for i,w in enumerate(wavelengths)]
    return sorted(wavelengths,key=lambda x: x[1])

def align_captures(cap,img_type = "reflectance"):
    """ 
    use rig relatives to align band images 
    """
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrices = cap.get_warp_matrices()
    cropped_dimensions,_ = imageutils.find_crop_bounds(cap,warp_matrices)
    im_aligned = imageutils.aligned_capture(cap, warp_matrices, warp_mode, cropped_dimensions, None, img_type=img_type)
    return im_aligned

def get_rgb(im_aligned, normalisation = False, plot=True, save_dir=None):
    """
    get rgb image from multispectral imae
    :param im_aligned (np.ndarray): multispectral image of dims (m,n,c), where c = 10. output from align_captures
    :param normalisation (bool): whether to normalise the rgb image for better contrast or not
    :param plot (bool): whether to plot the images or not
    
    """
    rgb_band_indices = [2,1,0]

    if normalisation is True:
        im_min = np.percentile(im_aligned[:,:,0:2].flatten(),  0.1)  # modify with these percentilse to adjust contrast
        im_max = np.percentile(im_aligned[:,:,0:2].flatten(), 99.9)  # for many images, 0.5 and 99.5 are good values

        im_display = np.zeros((im_aligned.shape[0],im_aligned.shape[1],len(rgb_band_indices)), dtype=np.float32)
        for i,rgb_i in enumerate(rgb_band_indices):
            im_display[:,:,i] = imageutils.normalize(im_aligned[:,:,rgb_i], im_min, im_max)

    else:
        im_display = np.take(im_aligned,rgb_band_indices,axis=2)

    if plot is True:
        fig = plt.figure()
        plt.imshow(im_display)
        plt.axis('off')
        if save_dir is not None:
            fig.savefig('{}.png'.format(save_dir), bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()
    return im_display
    

def import_captures(current_fp):
    """
    :param current_fp (str): filepath of micasense raw image IMG_****_1.tif
    from current_fp, list all band images, and import capture object
    """
    basename = current_fp[:-6]
    fn = glob.glob('{}_*.tif'.format(basename))
    fn = order_bands_from_filenames(fn)
    cap = capture.Capture.from_filelist(fn)
    return cap

def aligned_capture_rgb(capture, warp_matrices, cropped_dimensions, normalisation = True, img_type = 'reflectance',interpolation_mode=cv2.INTER_LANCZOS4):
    """ 
    :param capture (capture object): for 10-bands image
    :param warp_matrices (mxmx3 np.ndarray): in rgb order of [2,1,0] loaded from pickle
    :param cropped_dimensions (tuple): loaded from pickle
    align images using the warp_matrices used for aligning 10-band images and outputs an rgb image
    """

    warp_mode = cv2.MOTION_HOMOGRAPHY
    
    width, height = capture.images[0].size()

    rgb_band_indices = [2,1,0]

    im_aligned = np.zeros((height,width,len(rgb_band_indices)), dtype=np.float32 )

    for i,rgb_i in enumerate(rgb_band_indices):
        if img_type == 'reflectance':
            img = capture.images[rgb_i].undistorted_reflectance()
        else:
            img = capture.images[rgb_i].undistorted_radiance()

        if warp_mode != cv2.MOTION_HOMOGRAPHY:
            im_aligned[:,:,i] = cv2.warpAffine(img,
                                            warp_matrices[rgb_i],
                                            (width,height),
                                            flags=interpolation_mode + cv2.WARP_INVERSE_MAP)
        else:
            im_aligned[:,:,i] = cv2.warpPerspective(img,
                                                warp_matrices[rgb_i],
                                                (width,height),
                                                flags=interpolation_mode + cv2.WARP_INVERSE_MAP)
    (left, top, w, h) = tuple(int(i) for i in cropped_dimensions)
    im_cropped = im_aligned[top:top+h, left:left+w][:]

    if normalisation is True:
        # get normalised rgb image
        im_min = np.percentile(im_cropped.flatten(),  0.1)  # modify with these percentilse to adjust contrast
        im_max = np.percentile(im_cropped.flatten(), 99.9)  # for many images, 0.5 and 99.5 are good values

        im_display = np.zeros((im_cropped.shape[0],im_cropped.shape[1],len(rgb_band_indices)), dtype=np.float32)
        
        for i in range(len(rgb_band_indices)):
            im_display[:,:,i] = imageutils.normalize(im_cropped[:,:,i], im_min, im_max)
    else:
        im_display = im_cropped

    return im_display

def bboxes_to_patches(bboxes):
    if bboxes is not None:
        ((x1,y1),(x2,y2)) = bboxes
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        h = y1 - y2 # negative height as the origin is on the top left
        w = x2 - x1
        return (x1,y2), w, h
    else:
        return None

def sort_bbox(bbox):
    """
    :param (tuple): 4 coordinates describing opposite corners of the bbox coordinates
    """
    ((x1,y1),(x2,y2)) = bbox
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1,y2 = y2, y1

    return ((x1,y1),(x2,y2))

def plot_bboxes(fp):
    """ 
    :param fp (str): filepath of txt file which contains the bboxes of turbid, water, turbid_glint, water_glint, shore
    this function plot the bboxes of each category to validate the selection of bboxes with python GUI (get_training_data.py)
    """
    with open(fp, 'r') as fp:
        data = json.load(fp)

    # initialise categories
    button_names = ['turbid_glint','water_glint','turbid','water','shore']
    
    # intialise colours
    colors = ['orange','cyan','saddlebrown','blue','yellow']

    # mapping of categories and colors
    cat_colors = dict()
    for cat,c in zip(button_names,colors):
        cat_colors[cat] = c
    
    fig,ax = plt.subplots()
    ax.imshow(Image.open(list(data)[0]))
    for v in data.values():
        for k1,v1 in v.items():
            if v1 is not None:
                ((x1,y1),(x2,y2)) = v1
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                h = y1 - y2 # negative height as the origin is on the top left
                w = x2 - x1
                # in patches, x,y coord is the bottom left corner, 
                rect = patches.Rectangle((x1, y2), w, h, linewidth=1, edgecolor=cat_colors[k1], facecolor='none')
                patch = ax.add_patch(rect)
    plt.show()
    return 

def get_all_dir(fp,iter=3):
    """ get all parent sub directories up to iter (int) levels"""
    fp_temp = fp
    sub_dir_list = []
    for i in range(iter):
        base_fn, fn = os.path.split(fp_temp)
        sub_dir_list.append(fn)
        fp_temp = base_fn
    return '_'.join(reversed(sub_dir_list))

def get_bboxes_complement(dir,dir_sub,target_dir=None):
    """
    :param dir (str): directory with the bigger set of data, and the location of where to copy the files from
    :param dir_sub (str): subset of dir
    :param target_dir (str): target_directory to copy files, location of where to copy the files to
    """
    qaqc_plots = [os.path.splitext(i)[0] for i in os.listdir(dir_sub)]
    saved_bboxes = [os.path.splitext(i)[0] for i in os.listdir(dir)]
    complementary_bboxes = set(saved_bboxes).difference(set(qaqc_plots))
    print(len(complementary_bboxes))
    for i in complementary_bboxes:
        fn = '{}.txt'.format(i)
        original_fn = os.path.join(dir,fn)
        target_fn = os.path.join(target_dir,fn)
        shutil.copyfile(original_fn,target_fn)
        
    return list(complementary_bboxes)

def assign_new_parent_dir(panel_fp,parent_dir, split_iter=3):
    """" 
    :param panel_fp (dict): data loaded from rcu.load_panel_fp(r"saved_data\panel_fp.json")
    :param parent_dir (str): new parent_dir to replace with
    :param split_iter (int): iterations to split the os.path to get to the parent_directory
    this function is meant to import panel images from user's local directory even though fp in panel_fp belongs to another platform.
    in that way, the user do not have to search for all the panel images in their local machine as it will be very time consuming
    """
    panel_fp_dict = dict()
    for k,v in panel_fp.items():
        fp_temp = k
        dirs = []
        # split path
        for i in range(split_iter):
            prev_dir, next_dir = os.path.split(fp_temp)
            dirs.append(next_dir)
            fp_temp = prev_dir

        new_parent_dir = os.path.join(parent_dir,*reversed(dirs)) #* is a splat operator to make all items in list as arguments
        capture_list = []
        for capture in v:
            img_fp_list = []
            for img_fp in capture:
                img_fp_list.append(img_fp.replace(k,new_parent_dir))
            capture_list.append(img_fp_list)
        panel_fp_dict[new_parent_dir] = capture_list
    
    return panel_fp_dict

def filepath_from_filename(fn,dir=''):
    """
    :param fn (str): e.g. '13thSur22Sep_F1_RawImg_IMG_0085_1.png'
    reconstruct filepath from filename, only needed for QAQC files
    """
    basename, ext = os.path.splitext(fn)
    name_list = basename.split('_')
    dir = os.path.join(dir,*name_list[:3]) # expand list as arguments
    fn = '_'.join(name_list[3:])
    return dir, fn

def get_all_dir(fp,iter=3):
    """ get all parent sub directories up to 3 levels"""
    fp_temp = fp
    sub_dir_list = []
    for i in range(iter):
        base_fn, fn = os.path.split(fp_temp)
        sub_dir_list.append(fn)
        fp_temp = base_fn
    return '_'.join(reversed(sub_dir_list))

def get_ground_resolution(height,pixel_size=3.75,focal_length=5.4,fov_x=47.2,fov_y=35.4,p_x=1280,p_y=960):
    """returns ground resolution for MicaSense in the x and y direction (in meters)"""
    # s_x = (pixel_size*p_x)/1000
    # s_y = (pixel_size*p_y)/1000
    # fov_x = 2*np.arctan(s_x/(2*focal_length))*180/np.pi
    # fov_y = 2*np.arctan(s_y/(2*focal_length))*180/np.pi
    g_x = 2*height*np.tan(np.pi*fov_x/360)
    g_y = 2*height*np.tan(np.pi*fov_y/360)
    return g_x/p_x, g_y/p_y