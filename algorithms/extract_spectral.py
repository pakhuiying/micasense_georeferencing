# import micasense.imageset as imageset
import micasense.capture as capture
import micasense.imageutils as imageutils
import algorithms.radiometric_calib_utils as rcu
import algorithms.mutils as mutils
# import micasense.plotutils as plotutils
import os, glob
import json
from tqdm import tqdm
import cv2
# import pickle #This library will maintain the format as well

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import numpy as np
from math import ceil

def get_warp_matrices(current_fp):
    """ 
    helper function to get wrap_matrices from an image filename e.g. .../IMG_0041_1..tif
    from current_fp, import captures and output warp_matrices and cropped_dimensions
    """
    cap = mutils.import_captures(current_fp)
    warp_matrices = cap.get_warp_matrices()
    cropped_dimensions,_ = imageutils.find_crop_bounds(cap,warp_matrices)
    return warp_matrices, cropped_dimensions

def query_bboxes(img_bboxes, categories=None, condition= "or"):
    """
    :param img_bboxes (dict): output from store_bboxes
    :param categories (list of str): where each str is a category in 'turbid_glint','water_glint','turbid','water','shore'
    :param condition (str): either "and" or "or", which are operators connecting all the categories listed in categories
    returns a list of dict which fulfills categories queried and condition stipulated
    """
    queried_list = []#{fp: {img_name: dict() for img_name in img_dict.keys()} for fp,img_dict in img_bboxes.items()}
    for fp,img_dict in img_bboxes.items():
        for img_name, category_dict in img_dict.items():
            queried_data = [(cat,bbox,cat in categories) for cat, bbox in category_dict.items()]
            if condition == "or":
                if any([bb[-1] for bb in queried_data]):
                    queried_list.append({fp: {img_name: {bb[0]:bb[1] for bb in queried_data if bb[-1] is True}}})      
            elif condition == "and":
                if all([bb[-1] for bb in queried_data]) is True and (len(queried_data) == len(categories)):
                    queried_list.append({fp: {img_name: {bb[0]:bb[1] for bb in queried_data}}})

    output_fp = {fp: dict() for d in queried_list for fp in d.keys()}
    for d in queried_list:
        for fp, img_dict in d.items():
            for img_name, category_dict in img_dict.items():
                output_fp[fp][img_name] = category_dict
    
    return output_fp


class VerifyBboxes:
    def __init__(self,dir, assign_new_dir = None, split_iter=3):
        """ 
        :param dir (str): directory of where all the bboxes (.txt) are stored i.e. saved_bboxes/
        :param assign_new_dir (str): new parent_dir to replace with
        :param split_iter (int): iterations to split the os.path to get to the parent_directory
            this function is meant to import panel images from user's local directory even though fp in panel_fp belongs to another platform.
            in that way, the user do not have to search for all the panel images in their local machine as it will be very time consuming
        this function is basically a helper class to erify that bboxes are plotted correctly
        this function just checks whether bboxes are plotted correctly. 
        It does not extract the spectral information as radiometric calibration & correction has not been done using the panel images
        """
        self.dir = dir
        self.assign_new_dir = assign_new_dir
        self.split_iter = split_iter
        # initialise categories
        self.button_names = ['turbid_glint','water_glint','turbid','water','shore']
        # intialise colours
        self.colors = ['orange','cyan','saddlebrown','blue','yellow']
        self.color_mapping = {categories:colors for categories,colors in zip(self.button_names,self.colors)}
        # import wavelengths for each band
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}
        self.n_bands = len(wavelengths)
        self.wavelengths_idx = np.array([i[0] for i in wavelengths])
        self.wavelengths = np.array([i[1] for i in wavelengths])
        # aligning band images
        self.warp_mode = cv2.MOTION_HOMOGRAPHY
        self.img_type = "reflectance"

    def assign_new_parent_dir(self,fp):
        """
        :param fp (str): file path of image
        """
        if self.assign_new_dir is not None:
            fp_temp = fp
            dirs = []
            for i in range(self.split_iter):
                prev_dir, next_dir = os.path.split(fp_temp)
                dirs.append(next_dir)
                fp_temp = prev_dir

            new_parent_dir = os.path.join(self.assign_new_dir,*reversed(dirs)) #* is a splat operator to make all items in list as arguments
        else:
            new_parent_dir = fp

        return new_parent_dir

    def store_bboxes(self):
        """ 
        get all the bboxes txt files are store the info in a dictionary with keys:
        parent_directory (e.g. 'F:/surveys_10band/10thSur24Aug/F1/RawImg')
            img_names (e.g. 'IMG_0004_1.tif')
        """
        fp_list = [os.path.join(self.dir,fp) for fp in os.listdir(self.dir) if (fp.endswith(".txt")) and ("last_image" not in fp)]
        
        store_list = []
        for fp in fp_list:
            with open(fp, 'r') as fp:
                data = json.load(fp)
            basename,file_name = os.path.split(list(data)[0])
            if len(file_name) > 14:
                end_folder, file_name = mutils.filepath_from_filename(file_name,dir='')
                basename = os.path.join(basename,end_folder)
                file_name = file_name + '.tif'
            
            if self.assign_new_dir is not None:
                basename = self.assign_new_parent_dir(basename)

            store_list.append({basename: {file_name: data[list(data)[0]]}})

        output_fp = {fp: dict() for d in store_list for fp in d.keys()}
        for d in store_list:
            for fp, img_dict in d.items():
                for img_name, category_dict in img_dict.items():
                    output_fp[fp][img_name] = {cat:bbox for cat,bbox in category_dict.items() if bbox is not None}
                    
        return output_fp

    
    def plot_bboxes(self,show_n = 6,figsize=(8,20),save = False,dpi=200, plot_dir = None):
        """ 
        :param show_n (int): show how many plots. if number of images exceeds show_n, plot only show_n
        :param save (bool): save individual images if save is True. images are saved in folder 'QAQC_plots'
        :param dpi (int): dots per inch, determines the resolution of the image saved
        :param plot_dir (str): directory to save the QAQC_plots
        """
        store_dict = self.store_bboxes()
        # add legends
        colors = ['orange','cyan','saddlebrown','blue','yellow']
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
        labels = ['turbid_glint','water_glint','turbid','water','shore']

        if save is False:
            for flight_fp, img_dict in store_dict.items():
                images_names = list(img_dict)
                n_images = len(images_names)
                if n_images > 0:
                    current_fp = os.path.join(flight_fp,images_names[0])
                    warp_matrices, cropped_dimensions = get_warp_matrices(current_fp)

                    if show_n is None:
                        fig, axes = plt.subplots(ceil(n_images/2),2)

                    elif n_images < show_n:
                        fig, axes = plt.subplots(ceil(n_images/2),2,figsize=figsize)
                    else:
                        fig, axes = plt.subplots(ceil(show_n/2),2,figsize=figsize)
                        img_dict = {i:img_dict[i] for i in list(images_names)[:show_n]}
                    
                    for (image_name,bboxes),ax in tqdm(zip(img_dict.items(),axes.flatten())):
                        current_fp = os.path.join(flight_fp,image_name)
                        cap = mutils.import_captures(current_fp)
                        rgb_image = mutils.aligned_capture_rgb(cap, warp_matrices, cropped_dimensions, normalisation=True)
                        ax.imshow(rgb_image)
                        ax.set_title(image_name)
                        ax.axis('off')
                        for categories, bbox in bboxes.items():
                            coord, w, h = mutils.bboxes_to_patches(bbox)
                            rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor=self.color_mapping[categories], facecolor='none')
                            patch = ax.add_patch(rect)
                    
                    fig.suptitle(flight_fp)
                    plt.legend(lines,labels,loc='center left',bbox_to_anchor=(1.04, 0.5))
                    plt.tight_layout()
                    plt.show()
        
        else:
            if plot_dir is None:
                #create a new dir to store plot images
                plot_dir = os.path.join(os.getcwd(),"QAQC_plots")
                if not os.path.exists(plot_dir):
                    os.mkdir(plot_dir)
            else:
                plot_dir = os.path.join(plot_dir,"QAQC_plots")
                if not os.path.exists(plot_dir):
                    os.mkdir(plot_dir)

            for flight_fp, img_dict in tqdm(store_dict.items(),desc="Flights"):
                images_names = list(img_dict)
                n_images = len(images_names)
                if n_images > 0:
                    current_fp = os.path.join(flight_fp,images_names[0])
                    warp_matrices, cropped_dimensions = get_warp_matrices(current_fp)
                    for image_name,bboxes in tqdm(img_dict.items(),desc="Images"):
                        current_fp = os.path.join(flight_fp,image_name)
                        cap = mutils.import_captures(current_fp)
                        rgb_image = mutils.aligned_capture_rgb(cap, warp_matrices, cropped_dimensions, normalisation=False)
                        # filename
                        parent_dir = mutils.get_all_dir(flight_fp)
                        fn = parent_dir + '_{}'.format(image_name)
                        fn = fn.replace('.tif','')
                        full_fn = os.path.join(plot_dir,fn)
                        # plot
                        fig = plt.figure(figsize=(rgb_image.shape[1]/dpi,rgb_image.shape[0]/dpi))
                        ax = fig.add_axes([0, 0, 1, 1])
                        ax.imshow(rgb_image, interpolation='nearest')
                        # print(rgb_image.shape)
                        for categories, bbox in bboxes.items():
                            coord, w, h = mutils.bboxes_to_patches(bbox)
                            rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor=self.color_mapping[categories], facecolor='none')
                            patch = ax.add_patch(rect)
                            ax.text(coord[0],coord[1]+40,categories,bbox={'facecolor': self.color_mapping[categories], 'alpha': 0.35},fontsize=7)
                        # ax.set_title(f'True RGB ({fn})')
                        ax.axis('off')
                        # ax.legend(lines,labels,loc='center left',bbox_to_anchor=(1.0, 0.5))
                        # plt.tight_layout()
                        # plt.show()
                        #save plots
                        
                        # fig.set_size_inches(rgb_image.shape[1]/dpi,rgb_image.shape[0]/dpi)
                        # plt.imsave(fname='{}.png'.format(full_fn),arr=rgb_image,format='png')
                        fig.savefig('{}.png'.format(full_fn),dpi=dpi)
                        plt.close() # to not show any plots
            
# TODO: refactor the entire code below, by including radiometric correction. to ensure that spectral info from different env conditions are consistent
class ReflectanceImage:
    def __init__(self, cap, warp_matrices=None, cropped_dimensions=None):
        """
        :param cap (capture object):
        :param calibration_curve (dict): dls_panel_irr_calibration loaded from mutils.load_pickle(r"saved_data\dls_panel_irr_calibration.ob")
        :param warp_matrices (list of arrays): for band alignment
        :param cropped_dimensions (tuple): for cropping images after band alignment
        # - undistorted image,
        # - radiometrically calibrated using calibration panel, 
        # - radiometrically corrected by applying the correction factor
        # - band aligned and cropped
        """
        self.cap = cap
        self.warp_matrices = warp_matrices
        self.cropped_dimensions = cropped_dimensions
        self.warp_mode = cv2.MOTION_HOMOGRAPHY
        self.interpolation_mode=cv2.INTER_LANCZOS4
        self.img_type = "reflectance"
        try:
            dls_panel_irr_calibration = mutils.load_pickle(r"saved_data\dls_panel_irr_calibration.ob")
            self.calibration_curve = dls_panel_irr_calibration
        except:
            self.calibration_curve = None
        if warp_matrices is None and cropped_dimensions is None:
            self.warp_matrices = self.cap.get_warp_matrices()
            self.cropped_dimensions,_ = imageutils.find_crop_bounds(self.cap,self.warp_matrices)
        # initialise categories
        self.button_names = ['turbid_glint','water_glint','turbid','water','shore']
        # intialise colours
        self.colors = ['orange','cyan','saddlebrown','blue','yellow']
        self.color_mapping = {categories:colors for categories,colors in zip(self.button_names,self.colors)}
        # import wavelengths for each band
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}
        self.n_bands = len(wavelengths)
        self.wavelengths_idx = np.array([i[0] for i in wavelengths])
        self.wavelengths = np.array([i[1] for i in wavelengths])

    @classmethod
    def get_cap_from_filename(cls, image_names):
        """
        :param image_names (list of str): filenames of images
        """
        image_names = mutils.order_bands_from_filenames(image_names)
        cap = capture.Capture.from_filelist(image_names)
        return cls(cap)
    
    def get_calibrated_corrected_reflectance(self):
        """
        get radiometrically corrected and calibrated reflectance image
        returns list of array of reflectance in band order i.e. band 1,2,3,4,5,6,7,8,9,10
        """
        if self.calibration_curve is not None:
            IC = rcu.ImageCorrection(self.cap,self.calibration_curve)
            return IC.corrected_image()
        else:
            self.cap.undistorted_reflectance(self.cap.dls_irradiance())

    def get_aligned_reflectance(self):
        """
        align band images
        outputs aligned and cropped reflectance image
        """
        reflectance_image = self.get_calibrated_corrected_reflectance() #should be a list of arrays
        height, width = reflectance_image[0].shape[0], reflectance_image[0].shape[1]
        im_aligned = np.zeros((height,width,len(self.warp_matrices)), dtype=np.float32 )

        for i in range(len(self.warp_matrices)):
            img = reflectance_image[i]
            im_aligned[:,:,i] = cv2.warpPerspective(img,
                                                self.warp_matrices[i],
                                                (width,height),
                                                flags=self.interpolation_mode + cv2.WARP_INVERSE_MAP)
        
        (left, top, w, h) = tuple(int(i) for i in self.cropped_dimensions)
        im_cropped = im_aligned[top:top+h, left:left+w][:]
        return im_cropped
    
    def plot_bboxes(self,bboxes,**kwargs):
        """
        :param bboxes (dict): bboxes of corresponding img_fp. 
            keys are categories e.g. turbid_glint, turbid, water_glint, water and shore, and values are the corresponding bboxes
        """
        im_aligned = self.get_aligned_reflectance()

        # number of labelled cateories
        n_cats = len(list(bboxes))
        fig, axes = plt.subplots(n_cats+1,2,figsize=kwargs['figsize'])
        # plot rgb
        rgb_image = mutils.get_rgb(im_aligned, normalisation = False, plot=False)
        # plot normalised rgb
        rgb_image_norm = mutils.get_rgb(im_aligned, normalisation = True, plot=False)
        axes[0,0].imshow(rgb_image)
        axes[0,0].set_title('True RGB')
        axes[0,1].imshow(rgb_image_norm)
        axes[0,1].set_title('Normalised RGB')
        for ax in axes[0,:]:
            ax.axis('off')

        for i, (category,bbox) in enumerate(bboxes.items()):
            bbox = mutils.sort_bbox(bbox)
            ((x1,y1),(x2,y2)) = bbox
            ax_idx = i+1 #axis index
            # crop rgb image based on bboxes drawn
            im_cropped = rgb_image[y1:y2,x1:x2,:]
            # get multispectral reflectances from bboxes
            # flatten the image such that it has shape (m x n,c), where c is the number of bands
            spectral_flatten = im_aligned[y1:y2,x1:x2,:].reshape(-1,im_aligned.shape[-1])
            # sort the image by wavelengths instead of band numbers
            wavelength_flatten = spectral_flatten[:,self.wavelengths_idx]
            wavelength_mean = np.mean(wavelength_flatten,axis=0)
            wavelength_var = np.sqrt(np.var(wavelength_flatten,axis=0)) #std dev
            # add patches to plots
            coord, w, h = mutils.bboxes_to_patches(bbox)
            c = self.color_mapping[category]
            rect = patches.Rectangle(coord, w, h, linewidth=1, edgecolor=c, facecolor='none')
            patch = axes[0,0].add_patch(rect)
            
            axes[ax_idx,0].imshow(im_cropped)
            axes[ax_idx,0].set_title(category)
            axes[ax_idx,0].set_axis_off()
            axes[ax_idx,1].plot(self.wavelengths,wavelength_mean,color=c)
            eb = axes[ax_idx,1].errorbar(self.wavelengths,wavelength_mean,yerr=wavelength_var,color=c)
            eb[-1][0].set_linestyle('--')
            axes[ax_idx,1].set_title(f'Spectra ({category})')
            axes[ax_idx,1].set_xlabel('Wavelength (nm)')
            axes[ax_idx,1].set_ylabel('Reflectance')
        
        plt.tight_layout()
        plt.show()
        return
    
    def plot_multiline(self,bbox,**kwargs):
        """ 
        :param bbox (tuple): bbox of glint area
        plot multispectral reflectance of the image cropped by bbox
        """
        im_aligned = self.get_aligned_reflectance()
        ((x1,y1),(x2,y2)) = mutils.sort_bbox(bbox)

        def multiline(xs, ys, c, ax=None, **kwargs):
            """Plot lines with different colorings

            Parameters
            ----------
            xs : iterable container of x coordinates
            ys : iterable container of y coordinates
            c : iterable container of numbers mapped to colormap
            ax (optional): Axes to plot on.
            kwargs (optional): passed to LineCollection

            Notes:
                len(xs) == len(ys) == len(c) is the number of line segments
                len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

            Returns
            -------
            lc : LineCollection instance.
            """

            # find axes
            ax = plt.gca() if ax is None else ax

            # create LineCollection
            segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
            lc = LineCollection(segments, **kwargs)

            # set coloring of line segments
            #    Note: I get an error if I pass c as a list here... not sure why.
            lc.set_array(np.asarray(c))

            # add lines to axes and rescale 
            #    Note: adding a collection doesn't autoscalee xlim/ylim
            ax.add_collection(lc)
            ax.autoscale()
            return lc

        # m, n, channels = im_aligned.shape
        spectral_flatten = im_aligned[y1:y2,x1:x2,:].reshape(-1,im_aligned.shape[-1])
        wavelength_flatten = spectral_flatten[:,self.wavelengths_idx]
        wavelength_mean = np.mean(wavelength_flatten,axis=0)
        n_lines = wavelength_flatten.shape[0]
        x_array = np.array(self.wavelengths)
        x = np.repeat(x_array[np.newaxis,:],n_lines,axis=0)
        
        assert x.shape == wavelength_flatten.shape, "x and y should have the same shape"
        c = wavelength_flatten[:,-1] # select the last column which corresponds to NIR band
        
        fig, ax = plt.subplots(figsize=kwargs['figsize'])
        lc = multiline(x, wavelength_flatten, c, cmap='bwr',alpha=0.3 ,lw=2)
        ax.plot(self.wavelengths,wavelength_mean,color='yellow',label='Mean reflectance')
        axcb = fig.colorbar(lc)
        axcb.set_label('Reflectance')
        ax.set_title(kwargs['title'])
        ax.set_ylabel('Reflectance')
        ax.set_xlabel('Wavelength (nm)')
        plt.legend()
        plt.show()
        return (np.min(c),np.mean(c),np.max(c)) #where np.min(c) is the background NIR



class ThresholdGlint:
    def __init__(self, im_aligned,bbox):
        """
        :param im_aligned (np.ndarray): band aligned and calibrated & corrected reflectance image
        :param bbox (tuple): bbox of an area e.g. water_glint
        """
        self.im_aligned = im_aligned
        self.bbox = bbox
        # initialise categories
        self.button_names = ['turbid_glint','water_glint','turbid','water','shore']
        # intialise colours
        self.colors = ['orange','cyan','saddlebrown','blue','yellow']
        self.color_mapping = {categories:colors for categories,colors in zip(self.button_names,self.colors)}
        # import wavelengths for each band
        wavelengths = mutils.sort_bands_by_wavelength()
        self.wavelength_dict = {i[0]:i[1] for i in wavelengths}
        self.n_bands = im_aligned.shape[-1]

    def sort_bbox(self):
        ((x1,y1),(x2,y2)) = mutils.sort_bbox(self.bbox)
        
        return ((x1,y1),(x2,y2))
    
    def histogram_threshold(self, n_bins=200,plot=True,**kwargs):
        """
        :param n_bins (int): basically determines how sensitive we want the threshold to be. The larger the n_bins (the more sensitive the threshold for glint cut off)
        :param mode (str): percentile or histogram. this determines the method for glint detection
        """
        ((x1,y1),(x2,y2)) = self.sort_bbox()
        threshold_list = []

        fig, axes = plt.subplots(self.n_bands,3,figsize=kwargs['figsize'])
        for i in range(self.n_bands):
            band_i = self.im_aligned[y1:y2,x1:x2,i]
            band_i_flatten = band_i.flatten()
            axes[i,0].imshow(band_i,vmin=0,vmax=1)
            axes[i,0].axis('off')
            axes[i,0].set_title(f'Band {self.wavelength_dict[i]}')
            count,bins,_ = axes[i,1].hist(band_i_flatten,bins=n_bins)
            threshold_y = np.argmax(count)
            threshold_x = bins[threshold_y+1] #to get the right hand range of the bins
            threshold_list.append(threshold_x)
            axes[i,1].axvline(threshold_x, color='r')
            axes[i,1].text(threshold_x, threshold_y, f'{threshold_x:.3f}')
            axes[i,1].set_title('Histogram')
            axes[i,2].hist(band_i_flatten,n_bins, histtype='step',cumulative=True)
            axes[i,2].set_title('ECDF')
            axes[i,2].axvline(threshold_x, color='r')
            axes[i,2].text(threshold_x, threshold_y, f'{threshold_x:.3f}')

        if plot is True:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
        return threshold_list
    
    def percentile_threshold(self,n_bins=200,percentile_threshold=90,percentile_method='nearest',plot=True,**kwargs):
        """
        :param percentile_threshold (float): value between 0 and 100
        :param normalisation (bool): whether to normalise the image. It can help with increasing the contrast to identify glint. 
            However, non glint areas' contrast may be stretched to appear as glint which is misleading
        :param percentile_method (str): 'linear'(continuous) or 'nearest' (discontinuous method). 
            Note that for numpy version > 1.22, 'interpolation' argument is replaced with 'method', 'interpolation' is deprecated
        """
        ((x1,y1),(x2,y2)) = self.sort_bbox()
        threshold_list = []

        fig, axes = plt.subplots(self.n_bands,3,figsize=kwargs['figsize'])
        for i in range(self.n_bands):
            band_i = self.im_aligned[y1:y2,x1:x2,i]
            band_i_flatten = band_i.flatten()
            
            axes[i,0].imshow(band_i,vmin=0,vmax=1)
            axes[i,0].axis('off')
            axes[i,0].set_title(f'Band {self.wavelength_dict[i]}')
            
            glint_percentile = np.percentile(band_i_flatten,percentile_threshold,interpolation=percentile_method)
            threshold_list.append(glint_percentile)
            axes[i,1].hist(band_i_flatten,bins=n_bins)
            axes[i,1].axvline(glint_percentile, color='r')
            axes[i,1].set_title(f'Histogram ({glint_percentile:.3f})')

            axes[i,2].hist(band_i_flatten,n_bins, histtype='step',cumulative=True)
            axes[i,2].set_title(f'ECDF ({glint_percentile:.3f})')
            axes[i,2].axvline(glint_percentile, color='r')

        if plot is True:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
        return threshold_list
    
    def mask_bands(self,threshold_list,plot=True,**kwargs):
        """
        :param threshold_list (list of float): where each threshold corresponds to the threshold for each band
        threshold image given a threshold_list
        """
        ((x1,y1),(x2,y2)) = self.sort_bbox()

        glint_idxes_list = []
        # extracted_glint_list = []
        # glint_masked_list = []

        fig, axes = plt.subplots(self.n_bands,3,figsize=kwargs['figsize'])
        for i in range(self.n_bands):
            band_i = self.im_aligned[y1:y2,x1:x2,i]
            glint_idxes = np.argwhere(band_i>threshold_list[i])
            glint_idxes_list.append(glint_idxes)
            glint_masked = band_i.copy()
            glint_masked[(glint_idxes[:,0],glint_idxes[:,1])] = 0
            glint_extracted = band_i - glint_masked

            # glint_masked_list.append(glint_masked)
            # extracted_glint_list.append(glint_extracted)

            axes[i,0].imshow(band_i,vmin=0,vmax=1)
            axes[i,0].axis('off')
            axes[i,0].set_title(f'Band {self.wavelength_dict[i]}')

            axes[i,1].imshow(glint_masked,vmin=0,vmax=1)
            axes[i,1].axis('off')
            axes[i,1].set_title('Glint masked')
            
            axes[i,2].imshow(glint_extracted,vmin=0,vmax=1)
            axes[i,2].axis('off')
            axes[i,2].set_title('Glint extracted')
        
        if plot is True:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

        return glint_idxes_list#glint_masked_list,extracted_glint_list
    
    def spectral_unmixing(self, glint_idxes_list, plot=True):

        ((x1,y1),(x2,y2)) = self.sort_bbox()

        masked_im = self.mask_image(glint_idxes_list,plot=False)

        rgb_mean = []
        for i in range(3):
            im = masked_im[:,:,i]
            rgb_mean.append(np.mean(im[im>0]))
        
        rgb_inpaint = np.tile(np.array(rgb_mean),(masked_im.shape[0],masked_im.shape[1],1))
        
        rgb_bands = [2,1,0]
        
        fig = plt.figure(layout="constrained")
        gs = GridSpec(3, 3, figure=fig)

        for i, b in enumerate(rgb_bands):
            rgb_base = rgb_inpaint.copy()
            rgb_base[(glint_idxes_list[i][:,0],glint_idxes_list[i][:,1],b)] = self.im_aligned[y1:y2,x1:x2,i][(glint_idxes_list[i][:,0],glint_idxes_list[i][:,1])]
            ax1 = fig.add_subplot(gs[0, i])
            ax1.set_title(f'Glint for {self.wavelength_dict[i]} band')
            ax1.imshow(rgb_base)
            ax1.axis('off')

        ax2 = fig.add_subplot(gs[1:, :])
        ax2.axis('off')
        rgb_base = rgb_inpaint.copy()
        for i, b in enumerate(rgb_bands):
            rgb_base[(glint_idxes_list[i][:,0],glint_idxes_list[i][:,1],b)] = self.im_aligned[y1:y2,x1:x2,i][(glint_idxes_list[i][:,0],glint_idxes_list[i][:,1])]
        ax2.imshow(rgb_base)
        ax2.set_title('Combined Spectral')
        if plot is True:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

        return

    def mask_image(self,glint_idxes_list, plot=True):
        """
        :param threshold_list (list of float): where each threshold corresponds to the threshold for each band
        threshold image given a threshold_list
        """
        rgb_bands = [2,1,0] #668, 560, 475 nm
        ((x1,y1),(x2,y2)) = self.sort_bbox()
        # concatenate glint_idxes
        # concatenate_glint_idxes = np.unique(np.vstack([glint_idxes_list[i] for i in rgb_bands]),axis=0) #in order of band 2,1,0
        concatenate_glint_idxes = np.unique(np.vstack(glint_idxes_list[:3]),axis=0) #in order of band 2,1,0
        # get rgb image
        rgb_image = np.take(self.im_aligned[y1:y2,x1:x2,:],rgb_bands,axis=2)
        # copy of of rgb
        rgb_image_copy = rgb_image.copy()
        rgb_list = []
        for i in range(len(rgb_bands)):
            im_copy = rgb_image_copy[:,:,i]
            im_copy[(concatenate_glint_idxes[:,0],concatenate_glint_idxes[:,1])] = 0
            rgb_list.append(im_copy)
        
        rgb_masked = np.stack(rgb_list,axis=2)
        glint_extracted = rgb_image - rgb_masked

        fig, axes = plt.subplots(1,3)
        axes[0].imshow(rgb_image)
        axes[0].set_title('True RGB')
        axes[1].imshow(rgb_masked)
        axes[1].set_title('Masked RGB')
        axes[2].imshow(glint_extracted)
        axes[2].set_title('Glint extracted')
        for ax in axes:
            ax.axis('off')
        fig.suptitle('Glint identification')
        
        if plot is True:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

        return rgb_masked
    
    def plot_spectral(self,glint_idxes_list, plot=True):
        """
        plot spectral of masked image and extracted glint
        """
        ((x1,y1),(x2,y2)) = self.sort_bbox()

        extracted_glint_list = []
        glint_masked_list = []

        for i in range(self.n_bands):
            band_i = self.im_aligned[y1:y2,x1:x2,i]
            glint_masked = band_i.copy()
            glint_masked[(glint_idxes_list[i][:,0],glint_idxes_list[i][:,1])] = 0
            glint_extracted = band_i - glint_masked

            glint_masked_list.append(glint_masked)
            extracted_glint_list.append(glint_extracted)

        # store a list of tuple where (mean of reflectance, sd of reflctance)
        extracted_glint_list = [(np.mean(i[i>0]), np.sqrt(np.var(i[i>0],axis=0))) for i in extracted_glint_list]
        glint_masked_list = [(np.mean(i[i>0]), np.sqrt(np.var(i[i>0],axis=0))) for i in glint_masked_list]
        # sort based on wavelengths in ascending manner
        extracted_glint_list = [extracted_glint_list[i] for i in list(self.wavelength_dict)]
        glint_masked_list = [glint_masked_list[i] for i in list(self.wavelength_dict)]

        plot_statistics = {0: glint_masked_list, 1: extracted_glint_list}
        
        fig, axes = plt.subplots(1,2)
        for k, v in plot_statistics.items():
            axes[k].plot(list(self.wavelength_dict.values()), [i[0] for i in v])
            eb = axes[k].errorbar(list(self.wavelength_dict.values()),[i[0] for i in v],yerr=[i[1] for i in v])
            eb[-1][0].set_linestyle('--')

        axes[0].set_title('Reflectance for masked RGB')
        axes[1].set_title('Reflectance for extracted glint')
        for ax in axes:
            ax.set_xlabel('Wavelengths (nm)')
            ax.set_ylabel('Reflectance')

        if plot is True:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

        return

    