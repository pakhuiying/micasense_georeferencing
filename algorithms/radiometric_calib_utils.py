import micasense.capture as capture
import algorithms.mutils as mutils
import os, glob
import json
from tqdm import tqdm
import pickle #This library will maintain the format as well
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np


def list_img_subdir(dir):
    """ 
    given the directory, find all the panel images in the subdirectories
    """
    img_dir = []
    for survey in os.listdir(dir):
        for flight in os.listdir(os.path.join(dir,survey)):
            if len(flight) == 2: # to exclude the folder with missing blue bands
                if os.path.isdir(os.path.join(dir,survey,flight,'RawImg')):
                    img_dir.append(os.path.join(dir,survey,flight,'RawImg'))
                else:
                    print("RawImg folder not found!")
    return img_dir

def save_fp_panels(dir,save_dir=None):
    """ 
    :param dir (str): directory which stores all flight images
    :param save_dir (str): name of the folder where saved data are saved
    given the directory, find all the panel images in the subdirectories and save the filepaths in a dictionary
    >>> save_fp_panels(r"F:\surveys_10band")
    """
    RawImg_list = list_img_subdir(dir)

    RawImg_json = {f: None for f in RawImg_list}
    for k in tqdm(RawImg_json.keys()):
        RawImg_json[k] = get_panels(dir = k,search_n=3)

    if not os.path.exists('saved_data') and save_dir is None:
        dir = 'saved_data'
        os.mkdir('saved_data')
    else:
        dir = save_dir
        os.mkdir(dir)
    with open(os.path.join(dir,'panel_fp.json'), 'w') as fp:
        json.dump(RawImg_json, fp)
    
    return

def get_panels(dir,search_n=5):
    """
    :param dir(str): directory where images are stored
    search for the first 5 and last 5 captures to detect panel
    returns a list of captures in band order
    For panel images, efforts will be made to automatically extract the panel information, 
    panel images are not warped and the QR codes are individually detected for each band
    """
    number_of_files = len(glob.glob(os.path.join(dir,'IMG_*.tif')))//10 #divide by 10 since each capture has 10 bands
    last_file_number = number_of_files - 1 #since index of image starts from 0

    first_few_panels_fp = [glob.glob(os.path.join(dir,'IMG_000{}_*.tif'.format(str(i)))) for i in range(search_n)]
    last_few_panels_fp = [glob.glob(os.path.join(dir,'IMG_{}_*.tif'.format(str(last_file_number-i).zfill(4)))) for i in reversed(range(search_n))]
    panels_fp = first_few_panels_fp + last_few_panels_fp
    panels_list = []

    for f in panels_fp:
        cap_dict = {i+1:None for i in range(10)} # in order to order the files by band order, otherwise IMG_1 and IMG_10 are consecutive
        for cap in f:
            cap_dict[int(cap.split('_')[-1].replace('.tif',''))] = cap
        panels_list.append(list(cap_dict.values()))
    
    panelCaps = [capture.Capture.from_filelist(f) for f in panels_list] # list of captures
    detected_panels = [cap.detect_panels() for cap in panelCaps]

    detected_panels_fp = []
    for panels_n,panel_f in zip(detected_panels,panels_list):
        if panels_n == 10:
            detected_panels_fp.append(panel_f)
    return detected_panels_fp

def load_panel_fp(fp):
    """ 
    load panel_fp into dictionary, where keys are the parent directory, and keys are a list of list of panel images
    """
    with open(fp, 'r') as fp:
        data = json.load(fp)

    return data

def import_panel_reflectance(panelNames):
    """
    :param PanelNames (list of str): full file path of panelNames
    this should only be done once assuming that panel used is the same for all flights
    returns panel_reflectance_by_band
    """
    if panelNames is not None:
        panelCap = capture.Capture.from_filelist(panelNames)
    else:
        panelCap = None
    if panelCap is not None:
        if panelCap.panel_albedo() is not None:
            panel_reflectance_by_band = panelCap.panel_albedo()
        else:
            print("Panel reflectance not detected by serial number")
            panel_reflectance_by_band = None 
    else:
        panel_reflectance_by_band = None 
    
    return panel_reflectance_by_band



def import_panel_irradiance(panelNames,panel_reflectance_by_band=None):
    """
    :param PanelNames (list of str): full file path of panelNames
    :param panel_reflectance_by_band (list of float): reflectance values ranging from 0 to 1. 
    Import this value so we don't have to keep detecting the QR codes repeatedly if the QR panels are the same
    """
    if panelNames is not None:
        panelCap = capture.Capture.from_filelist(panelNames)
        # by calling panel_irradiance, it already accounts for all the radiometric calibration and correcting of vignetting effect and lens distortion
        # if panel_reflectance_by_band is None, internally when panel_irradiance is called,
        # reflectance of panel will be obtained from reflectance_from_panel_serial()
        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)
        # by calling dls_irradiance, it already returns a list of the corrected earth-surface (horizontal) DLS irradiance in W/m^2/nm
        dls_irradiance = panelCap.dls_irradiance()
        return {'dls':dls_irradiance,'panel':panel_irradiance}
    else:
        print("Panels not found")
        return None

def save_dls_panel_irr(panel_fp,panel_albedo=None,save_dir=None):
    """ 
    :param panel_fp (dict): where key is the subdirectory that refers to each flight, and keys is a list of list of all panel captures
    :param panel_albedo (list of float): if none, it will detect the reflectance from the panel's QR code
    save panel and dls irradiance for each band 
    """
    dls_panel_irr = {k:[] for k in panel_fp.keys()}
    for k,list_of_caps in tqdm(panel_fp.items()):
        for cap in list_of_caps:
            dls_panel_dict = import_panel_irradiance(cap,panel_albedo)
            dls_panel_irr[k].append(dls_panel_dict)
    
    if not os.path.exists('saved_data') and save_dir is None:
        os.mkdir('saved_data')
        dir = 'saved_data'
    else:
        dir = save_dir
        if not os.path.exists(dir):
            os.mkdir(dir)
    with open(os.path.join(dir,'dls_panel_irr.ob'), 'wb') as fp:
        pickle.dump(dls_panel_irr,fp)
    
    return dls_panel_irr

class RadiometricCorrection:
    def __init__(self,dls_panel_irr,center_wavelengths,dls_panel_irr_calibration=None):
        """
        :param dls_panel_irr (dict): contains the paired info of panel and dls irradiance for each flight
        :param center_wavelengths (list of float): center wavelengths of the bands in micasense camera
        dls_panel_irr_calibration (dict): where keys are band number, and keys are model coefficients 
        if dls_panel_irr_calibration is None, then it will fit a model to the data again using dls_panel_irr
        """
        self.dls_panel_irr = dls_panel_irr
        self.number_of_bands = 10
        self.center_wavelengths = center_wavelengths
        self.dls_panel_irr_calibration=dls_panel_irr_calibration

    def get_dls_panel_irr_by_band(self):
        """ obtain dls panel irradiance in band order i.e. 1,2,3,4,5,6,7,8,9,10"""
        panel_irr = {i:[] for i in range(self.number_of_bands)}
        dls_irr = {i:[] for i in range(self.number_of_bands)}
        for k, list_of_d in self.dls_panel_irr.items():
            for d in list_of_d:
                for i,dls in enumerate(d['dls']):
                    dls_irr[i].append(dls)
                for i,panel in enumerate(d['panel']):
                    panel_irr[i].append(panel)
        return {'dls':dls_irr,'panel':panel_irr}

    def plot(self):
        """ plot relationship between dls and panel irradiance by band order i.e. 1,2,3,4,5,6,7,8,9,10"""
        dls_panel_irr_by_band = self.get_dls_panel_irr_by_band()
        if self.dls_panel_irr_calibration is None:
            model_coeff = self.fit_curve_by_band()
        else:
            model_coeff = self.dls_panel_irr_calibration
        fig, axes = plt.subplots(self.number_of_bands//2,2,figsize=(8,15))
        for i,ax in zip(range(self.number_of_bands),axes.flatten()):
            x = dls_panel_irr_by_band['dls'][i]
            y = dls_panel_irr_by_band['panel'][i]
            ax.plot(x,y,'o')
            ax.set_title(r'Band {}: {} nm ($R^2:$ {:.3f}, N = {})'.format(i,self.center_wavelengths[i],model_coeff[i]['r2'],len(x)))
            ax.set_xlabel(r'DLS irradiance $W/m^2/nm$')
            ax.set_ylabel(r'Panel irradiance $W/m^2/nm$')
            x_vals = np.linspace(np.min(x),np.max(x),50)
            intercept = model_coeff[i]['intercept']
            slope = model_coeff[i]['coeff']
            y_vals = intercept + slope * x_vals
            ax.plot(x_vals.reshape(-1,1), y_vals.reshape(-1,1), '--')
            ax.text(0.1,ax.get_ylim()[1]*0.8,r"$y = {:.3f}x + {:.3f}$".format(slope,intercept))

        plt.tight_layout()
        return axes

    def fit_curve_by_band(self):
        """ get model coefficient for relationship between dls and panel irradiance by band order i.e. 1,2,3,4,5,6,7,8,9,10 """
        dls_panel_irr_by_band = self.get_dls_panel_irr_by_band()
        model_coeff = dict()#{i: None for i in range(10)}
        for i in range(self.number_of_bands):
            x = np.array(dls_panel_irr_by_band['dls'][i]).reshape(-1, 1)
            y = np.array(dls_panel_irr_by_band['panel'][i]).reshape(-1, 1)
            lm = LinearRegression().fit(x, y)
            r2 = r2_score(y,lm.predict(x))
            model_coeff[i] = {'coeff':lm.coef_[0][0],'intercept':lm.intercept_[0],'r2':r2}
        
        return model_coeff

def get_panel_radiance(fp_list):
    """
    :param fp_list (list of str): list of filepath (in band order i.e. band 1,2,3,4,5,6,7,8,9,10) of panel filepaths
    this function returns the panel_radiance (list of radiance for each band in band order i.e. band 1,2,3,4,5,6,7,8,9,10)
    """
    fp_list = mutils.order_bands_from_filenames(fp_list) # just in case fp_list is not sorted
    cap = capture.Capture.from_filelist(fp_list)
    panel_radiance = cap.panel_radiance()
    return panel_radiance

def get_panel_irradiance(panel_radiance,panel_albedo):
    """
    :param panel_radiance (list of radiance for each band in band order i.e. band 1,2,3,4,5,6,7,8,9,10)
    :param panel_albedo (list of float): list of float ranging from 0 to 1. This parameter is fixed if the same panel is used
    returns panel_irradiance
    """
    panel_irradiance = [mutils.panel_radiance_to_irradiance(radiance,albedo) for radiance, albedo in zip(panel_radiance,panel_albedo)]
    return panel_irradiance
class ImageCorrection:
    def __init__(self,cap,dls_panel_irr_calibration,panel_albedo=None):
        """ 
        :param cap (capture object): from capture.Capture.from_filelist(image_names), an image that is not panel
        :param dls_panel_irr_calibration (dict): where keys (int) are band number (0 to 9), and values are dict, with keys coeff and intercept
            loaded from saved_data
        :param dls_irradiance (list of float): irradiance from dls for each image (image and mission-specific)
        :param panel_albedo (list of float): list of float ranging from 0 to 1. This parameter is fixed if the same panel is used
        returns correction factor for each band (list of float).
        The corection factor is related to the ratio between the DLS and CRP irradiance on calibration images
        This correction ratio is the same for irradiance or radiance, as irradiance is simple radiance * pi
        We have to apply the correction for every image
        """
        self.dls_panel_irr_calibration = dls_panel_irr_calibration
        if panel_albedo is not None:
            self.panel_albedo = panel_albedo #flexibility to use ur own calibration panel, otherwise we will use micasense's panel reflectance values
        else:
            self.panel_albedo = [0.48112499999999997,
                                0.4801333333333333,
                                0.4788733333333333,
                                0.4768433333333333,
                                0.4783016666666666,
                                0.4814866666666666,
                                0.48047166666666663,
                                0.4790833333333333,
                                0.47844166666666665,
                                0.4780333333333333]
        self.cap = cap
        # radiance has already been undistorted, and corrected for vignetting effects etc
        self.radiance = [img.undistorted_radiance() for img in self.cap.images]# gets the same results as [img.undistorted(img.radiance()) for img in self.cap.images] #image radiance (list of arrays)
        dls_irradiance = cap.dls_irradiance() #corrected horizontal irradiance
        self.dls_irradiance = dls_irradiance
        self.dls_radiance = [i/np.pi for i in dls_irradiance]
        self.correction_factor = self.get_correction()
        #TODO: add try and except for load_pickle
        self.wavelength_by_band = mutils.load_pickle('saved_data/center_wavelengths_by_band.ob')
        self.wavelength_dict = {i:w for i,w in enumerate(self.wavelength_by_band)}
    
    @classmethod
    def get_cap_from_filename(cls, image_names,dls_panel_irr_calibration):
        """
        :param image_names (list of str): filenames of images
        """
        image_names = mutils.order_bands_from_filenames(image_names)
        cap = capture.Capture.from_filelist(image_names)
        return cls(cap,dls_panel_irr_calibration)


    def get_correction(self):
        """ 
        outputs a list of float values in band order i.e. band 1,2,3,4,5,6,7,8,9,10
        the correction_factor can then be multiplied to image radiance to get the radiometrically calibrated and corrected reflectance
        using rho_CRP/L_CRP x L instead of rho_CRP/L_DLS x L because if we use L_CRP as the reference, if there are noises at the CRP level then it is hard to know for sure (shadowing of CRP panel)
        """
        assert len(self.panel_albedo) == len(self.dls_irradiance), "panel_albedo bands must equal to panel_radiance bands"
        correction_factor = []
        for band_number,model_calib in self.dls_panel_irr_calibration.items():
            a = model_calib['coeff']
            b = model_calib['intercept']
            # rho_crp = self.panel_albedo[band_number]
            # L_crp = self.panel_radiance[band_number]
            # cf = a/(1-b*rho_crp/(np.pi*L_crp))
            dls_radiance = self.dls_radiance[band_number]
            cf_inverse = a*dls_radiance+b/np.pi #basically computes the estimated L_CRP (crp radiance)
            cf = 1/cf_inverse
            correction_factor.append(cf)
        return correction_factor
    
    def corrected_image(self):
        """
        returns radiometrically calibrated and corrected reflectance image using calibration curve
        """
        return [cf*radiance_im for radiance_im, cf in zip(self.radiance,self.correction_factor)]

    def get_reflectance_image(self,plot = True):
        """
        get cap's reflectance image, using micasense's default process
        by default, if irradiance is not supplied, then it will use the horizontal dls data found in image metadata
        """
        if plot is True:
            fig, ax = plt.subplots(10,1,figsize=(5,20))
            reflectance_imges = self.cap.undistorted_reflectance(self.cap.dls_irradiance())#[i.reflectance() for i in self.cap.images]
            for i,r in enumerate(reflectance_imges):
                ax[i].imshow(r)
                ax[i].set_title(f'Band {self.wavelength_by_band[i]}\nMean reflectance {np.mean(r):.3f}')
                ax[i].axis('off')
            plt.tight_layout()
            plt.show()
            return 
        else:
            return self.cap.undistorted_reflectance(self.cap.dls_irradiance()) #in band order (e.g.bands 0,1,2,3,4,5,6,7,8,9)
        
    def plot_difference_dls_crp_irradiance(self,panel_irradiance):
        """plot the difference in dls and crp irradiance"""
        plt.figure()
        plt.plot(self.wavelength_by_band,panel_irradiance,'o',label="CRP irradiance")
        plt.plot(self.wavelength_by_band,self.dls_irradiance,'o',label="DLS irradiance")
        plt.xlabel('Wavelength (nm)')
        plt.ylabel(r'Irradiance $W/m^2/nm$')
        plt.legend()
        plt.show()

    def plot_different_corrected_image(self,panel_irradiance,plot_image = True):
        """
        plot difference between non-corrected data and radiometrically calibrated and corrected reflectance image,
        where non-corrected data is 
        """
        corrected_img = self.corrected_image() #ordered by band i.e. band 1,2,3,4,5,6,7,8,9,10
        # non_corrected_img_dls = [i.reflectance() for i in self.cap.images] # already undistorted, corrected for vignetting etc
        # TODO: check if it;s the same as self.cap.undistorted_reflectance(self.dls_irradiance)
        non_corrected_img_dls = self.cap.undistorted_reflectance(self.dls_irradiance) 
        non_corrected_img_panel = self.cap.undistorted_reflectance(panel_irradiance)
        assert len(corrected_img) == len(non_corrected_img_dls) == len(non_corrected_img_panel)
        n_images = len(corrected_img)

        titles = ['Calibrated and Corrected reflectance','Corrected with only DLS','Corrected with only CRP']
        
        if plot_image is True:
            fig, axes = plt.subplots(n_images,3,figsize=(15,30))
            for i in range(n_images):
                # axes[i,0].imshow(corrected_img[i],vmin=0,vmax=1)
                # axes[i,1].imshow(non_corrected_img_dls[i],vmin=0,vmax=1)
                # axes[i,2].imshow(non_corrected_img_panel[i],vmin=0,vmax=1)
                ims = [corrected_img[i],non_corrected_img_dls[i],non_corrected_img_panel[i]]
                for ax,title,im in zip(axes[i,:], titles,ims):
                    ax.imshow(im,vmin=0,vmax=1)
                    ax.axis('off')
                    avg_reflectance = np.mean(im)
                    ax.set_title(f'Band {self.wavelength_by_band[i]}\n{title}\nMean reflectance: {avg_reflectance:.3f}')
                
            plt.tight_layout()
            plt.show()
            return
        
        else:
            corrected_img = [np.mean(i) for i in corrected_img]
            non_corrected_img_dls = [np.mean(i) for i in non_corrected_img_dls]
            non_corrected_img_panel = [np.mean(i) for i in non_corrected_img_panel]
            plt.figure()
            mean_reflectances = [corrected_img,non_corrected_img_dls,non_corrected_img_panel]
            for i,r in enumerate(mean_reflectances):
                plt.plot(self.wavelength_by_band,r,'o',label=titles[i])
            plt.title('Comparison of reflectances')
            plt.legend(loc='center left',bbox_to_anchor=(1.04, 0.5))
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Reflectance')
            plt.show()
            return


def radiometric_corrected_aligned_captures(cap,cf,img_type = "reflectance"):
    """
    :param cap (Capture object)
    :param cf (list of float): correction_factor obtained from CorrectionFactor.cf
    This function aligns the band images, then apply the correction factor to all the bands. 
    Note that the correction factor is mission-specific because it depends on the measured CRP radiance on that mission
    returns the reflectance of image in reflectance (default) or "radiance"
    """
    im_aligned = mutils.align_captures(cap,img_type = img_type)
    return np.multiply(im_aligned,np.array(cf))