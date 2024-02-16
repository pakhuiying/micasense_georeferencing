import cameratransform as ct
import numpy as np
import pandas as pd
import os
import rasterio
from osgeo import gdal
from PIL import Image
import glob
import contextlib
import pickle
from math import ceil
import cv2
import matplotlib.pyplot as plt
import algorithms.plot_map as plot_map
import algorithms.flight_attributes as FlightAttributes
import algorithms.select_GPS as SelectGPS
from algorithms.plot_map import PlotMap
from algorithms.alignment_error import correlation_coefficient

def directGeoreferencing_rasterio(imagePath,image_name,focal,lat,lon,alt,flight_angle,original_image_size=(1280,960),cropped_dimensions=(4, 6, 1240, 920)):
    """ 
    :param imagePath (str): directory where raw images are stored
    :param image_name (str): e.g. IMG_0079_1.tif
    :param focal (float): local length of camera
    :param original_image_size (tuple): original image_size = ImageWidth, ImageHeight
    :param lat (float): center coordinates of image
    :param lon (float): center coordinates of image
    :param alt (float): flight altitude
    :param flight_angle (float): in degrees, UAV's yaw converted to degrees
    :param cropped_dimensions (tuple of int): left, top, w, h (this is the cropped dimensions of dual camera system, it may be different for RedEdge cropped dimensions)
    this function georeferences one image
    modified from georeferencing code - UAV-Water-mosaicking-code
    im_cropped = img[top:top+h, left:left+w]
    """
    (left, top, w, h) = cropped_dimensions
    croppedImageWidth, croppedImageHeight = w-1, h-1
    rightExtent = croppedImageWidth + left
    bottomExtent = croppedImageHeight + top
    # heading param is wrt to east, so we add 90 degrees
    cam = ct.Camera(ct.RectilinearProjection(focallength_mm=focal,
                                            image=original_image_size),
                    ct.SpatialOrientation(elevation_m=alt,
                                        tilt_deg=0,
                                        roll_deg=0,
                                    heading_deg=flight_angle+90)) # This parameter should be set as 0º (when the UAV is heading East), 90º (when the UAV is heading North), 180º (when the UAV is heading West), or 270º (when the UAV is heading South). 

    # Latitude and Longitude of the GPS
    cam.setGPSpos(lat, lon, alt)

    # Image corners coordinates
    # UL (upper left), UR (upper right), LR (lower right), LL (lower left)
    coords = np.array([cam.gpsFromImage([left , top]), \
        cam.gpsFromImage([rightExtent-1 , top]), \
        cam.gpsFromImage([rightExtent-1, bottomExtent-1]), \
        cam.gpsFromImage([left , bottomExtent-1])])

    gcp1 = rasterio.control.GroundControlPoint(row=0, col=croppedImageHeight-1, x=coords[0,1], y=coords[0,0], z=coords[0,2], id=None, info=None)
    gcp2 = rasterio.control.GroundControlPoint(row=croppedImageWidth-1, col=croppedImageHeight-1, x=coords[1,1], y=coords[1,0], z=coords[1,2], id=None, info=None)
    gcp3 = rasterio.control.GroundControlPoint(row=croppedImageWidth-1, col=0, x=coords[2,1], y=coords[2,0], z=coords[2,2], id=None, info=None)
    gcp4 = rasterio.control.GroundControlPoint(row=0, col=0, x=coords[3,1], y=coords[3,0], z=coords[3,2], id=None, info=None)

    # Opening the original Image and generating a profile based on flight_stacks file generated before
    with rasterio.open(os.path.join(imagePath,'stacks',image_name), 'r') as src:
        profile = src.profile

        # Transformation
        tsfm = rasterio.transform.from_gcps([gcp1,gcp2,gcp3,gcp4])
        crs = rasterio.crs.CRS({"init": "epsg:4326"})
        profile.update(dtype=rasterio.uint16, transform = tsfm, crs=crs)

        georeferenced_stack_dir = os.path.join(imagePath, 'georeferenced_stacks')
        if not os.path.exists(georeferenced_stack_dir):
            os.mkdir(georeferenced_stack_dir)
            
        with rasterio.open(os.path.join(georeferenced_stack_dir, image_name), 'w', **profile) as dst:
            print(src.read().shape)
            dst.write(src.read().astype(rasterio.uint16)) # We write the coordinates in the image with this line.

def directGeoreferencing_gdal(imagePath,image_name,lat,lon,alt,flight_angle,dirname=""):
    """ 
    :param imagePath (str): directory where raw images are stored
    :param image_name (str): e.g. IMG_0079_1.tif
    :param lat (float): center coordinates of image
    :param lon (float): center coordinates of image
    :param alt (float): flight altitude
    :param flight_angle (float): in degrees, UAV's yaw converted to degrees
    :param dirname (str): directory name
    """
    georeferenced_thumbnails_directory = os.path.join(imagePath,f'georeferenced_thumbnails_{dirname}')
    if not os.path.exists(georeferenced_thumbnails_directory):
        os.mkdir(georeferenced_thumbnails_directory)

    # open image
    image_name = os.path.splitext(image_name)[0]
    try:
        im = np.asarray(Image.open(os.path.join(imagePath,'thumbnails',f'{image_name}.jpg')))
    except:
        im = None
        print(f"Image not found: {image_name}")

    image_fn = os.path.join(georeferenced_thumbnails_directory,image_name)
    GI = FlightAttributes.GeotransformImage(im,lat,lon,
                                altitude = alt,
                                angle = flight_angle)
    GI.georegister(image_fn)

def makeGif(gifParentDir,fileName):
    # gifParentDir = os.path.join(os.path.dirname(imagePath),"images")
    # filepaths
    fp_in = os.path.join(gifParentDir,f'*.png') #"/path/to/image_*.png"
    imgDir = os.path.dirname(gifParentDir)
    gifDir = os.path.join(imgDir,"gif")
    os.mkdir(gifDir) if not os.path.exists(gifDir) else None
    fp_out = os.path.join(gifDir,f'{fileName}.gif')#"/path/to/image.gif"

    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:

            # lazily load images
            imgs = (stack.enter_context(Image.open(f))
                    for f in sorted(glob.glob(fp_in)))

            # extract  first image from iterator
            img = next(imgs)

            # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
            img.save(fp=fp_out, format='GIF', append_images=imgs,
                    save_all=True, duration=200, loop=0)

class BatchCorrect:
    def __init__(self,imagePath,dirName):
        """ 
        :param imagePath (str): directory to image folder
        :param dirName (str): name of directory folder
        """
        self.imagePath = imagePath
        self.rawImagePath = os.path.join(imagePath,'RawImg')
        self.flightAttributesPath = os.path.join(imagePath,'flight_attributes','flight_attributes.csv')
        self.log = pd.read_csv(self.flightAttributesPath)
        self.height_dict = FlightAttributes.get_heights(imagePath,self.log)
        self.flight_points = SelectGPS.readSelectedGPS(self.rawImagePath)
        self.dirName = dirName
    
    def save_data(self,dirPath,fileName, data):
        save_fp = os.path.join(dirPath,f'{fileName}.ob')
        with open(save_fp,'wb') as fp:
            pickle.dump(data,fp)
        return

    def get_distance_interpolation(self,interpolate_distance = 0.01,pad_distance = 1.5):
        IC = FlightAttributes.InterpolateCoordinates(self.log, 
                                interpolate_distance=interpolate_distance, 
                                pad_distance=pad_distance)
        interp_dist_dict = IC.get_interpolated_dict()
        self.interp_dist_dict = interp_dist_dict
        self.IC = IC
        return interp_dist_dict
    
    def get_time_interpolation(self):
        IF = FlightAttributes.InterpolateFlight(self.log, 
                                                interpolate_milliseconds=100)
        df_interpolated = IF.interpolate_flight(plot=False)
        return df_interpolated
    
    def shift_coord(self, shift_n):
        df_interpolated = self.IC.shift_coord(self.interp_dist_dict, shift_n=shift_n)
        # crop df based on selected flight points
        df_cropped = df_interpolated.iloc[self.flight_points,:]
        return df_cropped
    
    def time_delay(self,df_interpolated,timedelta):
        df1 = FlightAttributes.time_delta_correction(df_interpolated, timedelta=timedelta, 
                        columns_to_shift = ['timestamp', 'timedelta', 'latitude', 'longitude'])
        df_cropped = df1.iloc[self.flight_points,:]
        return df_cropped
    
    def get_geotransform(self,df_cropped):
        PG = FlightAttributes.PlotGeoreference(self.imagePath,df_cropped)
        geotransform_list = PG.get_flight_attributes()
        return geotransform_list

    def georeference_UAV(self,ne_point,sw_point,canvas, df_cropped, calculate_correlation=False):
        """ 
        returns the canvas with the UAV images plotted on top of it,
        and correlation coefficient of each image with the canvas
        """
        self.ne = ne_point
        self.sw = sw_point
        geotransform_list = self.get_geotransform(df_cropped)
        GR = GeoreferenceRaster(ne_point,sw_point,canvas,geotransform_list)
        im_display, cc_list = GR.georeference_UAV(calculate_correlation=calculate_correlation)
        return im_display, cc_list
    
    def plot_params(self, PM, ax, ax_idx, title):
        """ 
        :param PM (PlotMap class)
        """
        tickFontSize = 7
        tick_breaks = 4
        NorthFontSize = 7
        triangleFontSize = 5
        scaleBarFontSize = 8
        scaleBarWidth = 3
        n_fig = 8

        PM.plot(ax=ax,add_ticks=False,add_compass=False,add_scale_bar=False)
                
        # set compass
        PM.add_compass(ax=ax,fontsize_N=NorthFontSize,fontsize_triangle=triangleFontSize)
        # set north
        PM.add_scale_bar(ax=ax,fontsize=scaleBarFontSize,lw=scaleBarWidth)
        # set subfigure title
        subfig_title = chr(ord('a') + ax_idx)
        ax.set_title(f'({subfig_title}) {title}')
        
        # add ticks for border plots
        if ax_idx == 0:
            PM.add_ticks(ax=ax,fontsize=tickFontSize,tick_breaks=tick_breaks,add_xticks=False)
        elif ax_idx == n_fig//2:
            PM.add_ticks(ax=ax,fontsize=tickFontSize,tick_breaks=tick_breaks)
        elif ax_idx > n_fig//2:
            PM.add_ticks(ax=ax,fontsize=tickFontSize,tick_breaks=tick_breaks,add_yticks=False)
        else:
            PM.add_ticks(ax=ax,fontsize=tickFontSize,tick_breaks=tick_breaks,add_yticks=False, add_xticks= False)
        return

    def main_timeDelay(self, modify_df,ne_point,sw_point,canvas, calculate_correlation=True):
        """ 
        :param modify_df (func): modify_df is a function to import an external function to modify the df
        """
        height_steps = 6
        est_time_delay1 = 0
        est_time_delay2 = -2
        n_fig = 8
        
        df_interpolated = self.get_time_interpolation()

        height_dict = dict()
        for height in range(height_steps): # try offsets in height of up to 5m by 1meters
            DEM_offset_height = self.height_dict['DEM_offset_height'] - height
            
            timeDelay_dict = dict()

            fig, axes = plt.subplots(2,ceil(n_fig/2), figsize=(16,7))

            for i, (td,ax) in enumerate(zip(np.linspace(est_time_delay2,est_time_delay1, n_fig),axes.flatten())):
                df_cropped = self.time_delay(df_interpolated,timedelta=td)
                df_cropped = modify_df(df_cropped, DEM_offset_height)
                im_display, cc_list = self.georeference_UAV(ne_point, sw_point, canvas, 
                                                            df_cropped, 
                                                            calculate_correlation)
                td_str = f'{abs(td):.3f}'
                timeDelay_dict[td_str] = cc_list
                PM = plot_map.PlotMap(ne_point,sw_point,im_display)
                # add plot params
                self.plot_params(PM, ax, ax_idx=i, title=f'Time: {td_str}')

            # add fig title
            correctedHeight = int(self.height_dict['measuredHeight'] - DEM_offset_height)
            fig.suptitle(f"Height: {self.height_dict['actualHeight']}m, Corrected height: {correctedHeight}m")
            plt.tight_layout()
            plt.show()

            # save cc list
            height_dict[correctedHeight] = timeDelay_dict

            # save fig
            parentDir = os.path.join(self.imagePath,"images",self.dirName)
            os.mkdir(parentDir) if not os.path.exists(parentDir) else None
            fname = os.path.join(parentDir,f'offsetHeight{int(DEM_offset_height)}.png')
            fig.savefig(fname)

        # save cc list on disk
        self.save_data(parentDir,'cc_list',height_dict)
        
        # create gifs
        makeGif(parentDir,self.dirName)

    def main_shiftCoord(self, modify_df,ne_point,sw_point,canvas, calculate_correlation=True):
        """ 
        :param modify_df (func): modify_df is a function to import an external function to modify the df
        """
        height_steps = 6
        dist1 = 0
        dist2 = 70
        n_fig = 8
        
        self.get_distance_interpolation()

        height_dict = dict()
        for height in range(height_steps): # try offsets in height of up to 5m by 1meters
            DEM_offset_height = self.height_dict['DEM_offset_height'] - height
            
            shift_n_dict = dict()

            fig, axes = plt.subplots(2,ceil(n_fig/2), figsize=(16,7))

            for i, (shift_n , ax) in enumerate(zip(np.linspace(dist1,dist2,n_fig,dtype=int),axes.flatten())):
                try:
                    df_cropped = self.shift_coord(shift_n)
                    df_cropped = modify_df(df_cropped, DEM_offset_height)
                    im_display, cc_list = self.georeference_UAV(ne_point, sw_point, canvas, 
                                                                df_cropped, 
                                                                calculate_correlation)
                    shift_n_dict[shift_n] = cc_list
                    PM = plot_map.PlotMap(ne_point,sw_point,im_display)
                    # add plot params
                    self.plot_params(PM, ax, ax_idx=i, title=f'Shift: {shift_n}')
                except:
                    ax.axis('off')
                    pass

            # add fig title
            correctedHeight = int(self.height_dict['measuredHeight'] - DEM_offset_height)
            fig.suptitle(f"Height: {self.height_dict['actualHeight']}m, Corrected height: {correctedHeight}m")
            plt.tight_layout()
            plt.show()

            # save cc list
            height_dict[correctedHeight] = shift_n_dict

            # save fig
            parentDir = os.path.join(self.imagePath,"images",self.dirName)
            os.mkdir(parentDir) if not os.path.exists(parentDir) else None
            fname = os.path.join(parentDir,f'offsetHeight{int(DEM_offset_height)}.png')
            fig.savefig(fname)

        # save cc list on disk
        self.save_data(parentDir,'cc_list',height_dict)
        
        # create gifs
        makeGif(parentDir,self.dirName)

class GeoreferenceRaster:
    def __init__(self, ne, sw, canvas, geotransform_list):
        self.ne = ne
        self.sw = sw
        self.canvas = canvas
        self.geotransform_list = geotransform_list
        self.canvas_height = canvas.shape[0]
        self.canvas_width = canvas.shape[1]
        self.canvas_lat_res = (self.ne[0] - self.sw[0])/self.canvas_height
        self.canvas_lon_res = (self.ne[1] - self.sw[1])/self.canvas_width
        self.upper_lat = self.ne[0]
        self.left_lon = self.sw[1]

    def get_UAV_res(self):
        """ 
        obtain the smallest resolution in UAV imagery 
        latitude and longitude pixel resolution must be the same becus in QGIS both resolution are the same
        """
        pixel_res = 1
        # get the max and min lat and lon values, and the corresponding im idx
        for idx, gt in self.geotransform_list.items():
            if gt['lat_res'] < pixel_res:
                pixel_res = gt['lat_res']
            if gt['lon_res'] < pixel_res:
                pixel_res = gt['lon_res']

        self.pixel_res = pixel_res
        
        return pixel_res
    
    def resize_canvas(self):
        """ bring UAV imagery and base map to the same lat and lon res """
        pixel_res = self.get_UAV_res()
        height_resize = int((self.canvas_lat_res/pixel_res)*self.canvas_height)
        width_resize = int((self.canvas_lon_res/pixel_res)*self.canvas_width)
        return cv2.resize(self.canvas, (width_resize, height_resize))
    
    def get_row_col_index(self, lat, lon, rot_im):
        """ 
        :param lat (float): center coord of rot_im
        :param lon (float): center coord of rot_im
        :param rot_im (np.ndarray): rotated image
        returns the upp/low row and column index when provided center lat and lon values
        """
        nrow, ncol = rot_im.shape[0], rot_im.shape[1]
        row_idx = int((self.upper_lat - lat)/self.pixel_res)
        col_idx = int((lon - self.left_lon)/self.pixel_res)
        #row_idx and col_idx wrt to center coord
        upper_row_idx = row_idx - nrow//2
        upper_row_idx = 0 if upper_row_idx < 0 else upper_row_idx
        lower_row_idx = upper_row_idx + nrow
        left_col_idx = col_idx - ncol//2
        left_col_idx = 0 if left_col_idx < 0 else left_col_idx
        right_col_idx = left_col_idx + ncol
        return upper_row_idx, lower_row_idx, left_col_idx, right_col_idx
    
    def georeference_UAV(self,calculate_correlation=False):
        """
        overlay UAV imagery over basemap
        """
        im_display = self.resize_canvas()
        im_display_copy = im_display.copy()
        cc_list = []
        for idx, gt in self.geotransform_list.items():
            flight_angle = gt['flight_angle']
            fp = gt['image_fp']
            
            if fp.endswith('.tif') or fp.endswith('.jpg'):
                im = np.asarray(Image.open(fp)) if (os.path.exists(fp)) else None
            
            if im is None:
                raise NameError("image is None because filepath d.n.e")
            GI = FlightAttributes.GeotransformImage(im,None,None,None,angle=flight_angle)
            rot_im = GI.affine_transformation(plot=False)
            upper_row_idx, lower_row_idx, left_col_idx, right_col_idx = self.get_row_col_index(gt['lat'],gt['lon'],rot_im) #row/col idx wrt to center coord
            background_im = im_display[upper_row_idx:lower_row_idx,left_col_idx:right_col_idx,:]
            assert rot_im.shape == background_im.shape, f'shapes are diff {rot_im.shape} {background_im.shape}'
            if calculate_correlation is True:
                cc = correlation_coefficient(im_display_copy[upper_row_idx:lower_row_idx,left_col_idx:right_col_idx,0],rot_im[:,:,0])
                cc_list.append((fp,cc))
            overlay_im = np.where(rot_im == 0, background_im,rot_im)
            im_display[upper_row_idx:lower_row_idx,left_col_idx:right_col_idx,:] = overlay_im

        return im_display, cc_list
    
    def plot(self,ax=None,add_ticks=True, add_compass=True, add_scale_bar=True,calculate_correlation=False):
        """ plot the georeferenced raster using the PlotMap plot method """
        im_display, cc_list = self.georeference_UAV(calculate_correlation)
        PM = PlotMap(self.ne,self.sw,im_display)
        PM.plot(ax=ax,
                add_ticks=add_ticks,
                add_compass=add_compass,
                add_scale_bar=add_scale_bar)
        return cc_list

class MosaicSeadron:
    """ MosaicSeadron georeferencing method"""
    def __init__(self,lat,lon,alt,flight_angle,flight_angle_correction = -90,focal=5.4,sensor_size=(4.8,3.6),original_image_size=(1280,960),cropped_dimensions=(4, 6, 1240, 920)):
        """ 
        :param focal (float): local length of camera
        :param original_image_size (tuple): original image_size = ImageWidth, ImageHeight
        :param alt (float): flight altitude
        :param flight_angle (float): in degrees, UAV's yaw converted to degrees
        :param sensor_size (mm per pixel): e.g. sensor_size=(4.8,3.6)
        :param cropped_dimensions (tuple of int): left, top, w, h (this is the cropped dimensions of dual camera system, it may be different for RedEdge cropped dimensions)
        this function georeferences one image
        modified from georeferencing code - MosaicSeadron
        im_cropped = img[top:top+h, left:left+w]
        """
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.flight_angle = flight_angle
        self.flight_angle_correction = flight_angle_correction
        self.focal = focal
        self.sensor_size = sensor_size
        self.original_image_size = original_image_size
        self.cropped_dimensions = cropped_dimensions

    def get_cam(self):
        """ get camera that is oriented towards north"""
        cam = ct.Camera(ct.RectilinearProjection(focallength_mm=self.focal,
                                                sensor = self.sensor_size,
                                                image=self.original_image_size),
                        #0°: the camera faces “north”, 
                        #90°: east, 
                        # 180°: south, 
                        # 270°: west
                        ct.SpatialOrientation(elevation_m=self.alt,
                                            tilt_deg=0,
                                            roll_deg=0,
                                        heading_deg=self.flight_angle+self.flight_angle_correction))
        
        cam.setGPSpos(self.lat, self.lon, self.alt) #cam coord matches with the image's center GPS coordinate

        return cam
    
    def get_gcps(self):
        cam = self.get_cam()
        # Image corners coordinates
        # UL (upper left), UR (upper right), LR (lower right), LL (lower left)

        coords = np.array([cam.gpsFromImage([0, 0]), 
                           cam.gpsFromImage([self.original_image_size[0] - 1, 0]), 
                           cam.gpsFromImage([self.original_image_size[0] - 1, self.original_image_size[1] - 1]), 
                           cam.gpsFromImage([0, self.original_image_size[1] - 1])])
        
        gcp1 = rasterio.control.GroundControlPoint(row = 0, col = 0, x = coords[0, 1], y = coords[0, 0], z = coords[0, 2])
        gcp2 = rasterio.control.GroundControlPoint(row = self.original_image_size[0] - 1, col = 0, x = coords[1, 1], y = coords[1, 0], z = coords[1, 2])
        gcp3 = rasterio.control.GroundControlPoint(row = self.original_image_size[0] - 1, col = self.original_image_size[1] - 1, x = coords[2, 1], y = coords[2, 0], z = coords[2, 2])
        gcp4 = rasterio.control.GroundControlPoint(row = 0, col = self.original_image_size[1] - 1, x = coords[3, 1], y = coords[3, 0], z = coords[3, 2])

        return [gcp1, gcp2, gcp3, gcp4]
    
    def get_geotransform(self, image_fp):
        """ 
        obtain the smallest resolution in UAV imagery 
        latitude and longitude pixel resolution must be the same becus in QGIS both resolution are the same
        """
        cam = self.get_cam()
        UL_coord = cam.gpsFromImage([0, 0])
        LR_coord = cam.gpsFromImage([self.original_image_size[0] - 1, self.original_image_size[1] - 1])
        UL_delta_coord = cam.gpsFromImage([1, 1])

        lat_res = abs(UL_coord[0] - UL_delta_coord[0])#/self.original_image_size[0]
        lon_res = abs(UL_coord[1] - UL_delta_coord[1])#/self.original_image_size[1]
        pix_res = (lat_res + lon_res)/2
        center_coord_lat = (UL_coord[0] + LR_coord[0])/2
        center_coord_lon = (UL_coord[1] + LR_coord[1])/2
        return {'lat': center_coord_lat, 'lon': center_coord_lon, 
                'lat_res': pix_res, 'lon_res': pix_res, 
                'flight_angle': self.flight_angle, 'image_fp': image_fp}

    def get_affine_transform(self):
        return rasterio.transform.from_gcps(self.get_gcps())
    
    def georeference(self, imagePath, image_name, dir_name):
        """ 
        :param imagePath (str): directory to image folder
        :param image_name (str): name of UAV image with .tif extension
        :param dir_name (str): name of directory folder
        """
        tsfm = self.get_affine_transform()

        # Opening the original Image and generating a profile based on flight_stacks file generated before
        with rasterio.open(os.path.join(imagePath,'stacks',image_name), 'r') as src:
            profile = src.profile

            crs = rasterio.crs.CRS({"init": "epsg:4326"})
            profile.update(dtype=rasterio.uint16, transform = tsfm, crs=crs)

            georeferenced_stack_dir = os.path.join(imagePath, dir_name)
            if not os.path.exists(georeferenced_stack_dir):
                os.mkdir(georeferenced_stack_dir)
                
            with rasterio.open(os.path.join(georeferenced_stack_dir, image_name), 'w', **profile) as dst:
                dst.write(np.flip(src.read().astype(rasterio.uint16),axis=1))
        
        return
    
def get_MosaicSeadron_geotransform_list(imagePath,flight_attributes_df, dir_name, 
                                        flight_angle_correction = -90, original_image_size = (1239,919)):
    """ 
    :param imagePath (str): directory path where images are stored
    :param flight_attributes_df (pd.DataFrame):  dataframe with image_name, flight angle
    :param dir_name (str): name of directory folder
    """
    geotransform_list = dict()
    for image_index, rows in flight_attributes_df.iterrows():
        image_name = rows['image_name']
        flight_angle = rows['flight_angle']
        
        # georeference each UAV image
        RG = MosaicSeadron(rows['latitude'],
                                          rows['longitude'],
                                          rows['altitude'],
                                          flight_angle = flight_angle,
                                          flight_angle_correction = flight_angle_correction,
                                          original_image_size = original_image_size)
        RG.georeference(imagePath, image_name, dir_name)
        
        # open georeferenced image
        image_fp = os.path.join(imagePath,dir_name,image_name)
        ds = gdal.Open(image_fp)
        # get attributes
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()
        UL_lon = gt[0]
        UL_lat = gt[3]
        lon_res = gt[1]
        lat_res = gt[5]
        lat = UL_lat + lat_res*height/2
        lon = UL_lon + lon_res*width/2
        thumbnail_fn = f'{os.path.splitext(image_name)[0]}.jpg'
        image_fp = os.path.join(imagePath,'thumbnails',thumbnail_fn)
        geotransform_list[image_index] = {'lat': lat, 'lon': lon, 
                'lat_res': abs(lat_res), 'lon_res': abs(lon_res), 
                'flight_angle': flight_angle, 'image_fp': image_fp}
    return geotransform_list