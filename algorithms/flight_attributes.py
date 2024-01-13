import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pickle
import numpy as np
import pandas as pd
import datetime
import os
import algorithms.mutils as mutils
import cv2
from osgeo import gdal, osr
from scipy.interpolate import griddata
from tqdm import tqdm
from PIL import Image
from math import ceil
from scipy.optimize import curve_fit
import re
import micasense.imageutils as imageutils

# helper function that is only relevant to my file naming
def get_heights(imagePath, flight_attributes_df):
    """ 
    :param imagePath (str): image directory
    :param flight_attributes_df (pd.DataFrame): df that contains POS data, image file name
    """
    measuredHeight = np.median(flight_attributes_df['altitude'])
    # where the folder name contains the flight height information e.g. ...15H...25H...
    actualHeight = int(re.search('[0-9][0-9]H',imagePath).group(0).replace('H',''))
    DEM_offset_height = int(measuredHeight - actualHeight)
    print(f'offset height: {DEM_offset_height}m')
    return {'measuredHeight':measuredHeight, 
    'actualHeight':actualHeight, 
    'height_max':np.max(flight_attributes_df['altitude'].values),
    'height_min':np.min(flight_attributes_df['altitude'].values),
    'DEM_offset_height':DEM_offset_height}

def plot_height(imagePath, flight_attributes_df,height_dict):
    """ 
    plot height with image index
    identify median height and get height info from the folder
    """
    # plot
    plt.figure()
    plt.plot(flight_attributes_df.index.to_list(),flight_attributes_df['altitude'])
    height_max = height_dict['height_max']
    height_min = height_dict['height_min']
    measuredHeight = height_dict['measuredHeight']
    actualHeight = height_dict['actualHeight']
    DEM_offset_height = height_dict['DEM_offset_height']
    plt.axhline(y=measuredHeight, ls='--',c='k',label=r'Median $Height_{GPS}$:' f" {int(measuredHeight)}m\n(offset height = {DEM_offset_height}m)")
    plt.title(r'Max $Height_{GPS}$:' + f' {height_max:.1f}m, ' + r'Min $Height_{GPS}$: ' + f'{height_min:.1f}m\n' + r'$Height_{UAV}$:' +f'{actualHeight}m')
    plt.ylabel(r'$Height_{GPS}$ (m)')
    plt.xlabel('Image index')
    plt.legend()
    # save fig
    parentDir = os.path.join(os.path.dirname(imagePath),"images")
    os.mkdir(parentDir) if not os.path.exists(parentDir) else None
    fname = os.path.join(parentDir,f'height_timeseries.png')
    plt.savefig(fname)
    plt.show()
    return

class FlightAttributes:
    def __init__(self,df):
        """ 
        :param df (pandas DF class): output from:
            data, columns = imgset.as_nested_lists()
            df = pd.DataFrame.from_records(data, index='timestamp', columns=columns)
        """
        self.df = df

    def calculate_angle(self,x0,y0,x1,y1, ref_vec = np.array([1,0])):
        start_v = np.array([x0,y0])
        end_v = np.array([x1,y1])
        vec = end_v - start_v
        vec = vec/np.linalg.norm(vec) # normalised vector
        return np.arccos(np.dot(vec,ref_vec))#/np.pi*180 
    
    def calculate_bends(self,yaw,plot=True):
        idx_list = list(range(len(yaw)))
        yaw_diff = np.diff(yaw,append=0) # to ensure length is same as idx_list and yaw
        # yaw_diff = np.diff(yaw_diff,append=0)
        yaw_diff1 = np.rint(yaw_diff)#yaw_diff.astype(int)
        bends_idx = np.argwhere(yaw_diff1 != 0).flatten()
        bends_yaw = yaw_diff[bends_idx]

        # idx_diff = np.diff(bends_idx,append=bends_idx[-1])
        idx_diff = np.diff(bends_idx,append=0)
        idx2 = np.argwhere(idx_diff>2)
        bends_idx1 = bends_idx[idx2]
        bends_yaw1 = yaw_diff[bends_idx1]
        if plot is True:
            plt.figure()
            plt.plot(idx_list,yaw,label='yaw')
            plt.plot(idx_list,yaw_diff,label='diff')
            plt.scatter(bends_idx,bends_yaw,c='red',s=10,alpha=0.5,label='bends')
            plt.scatter(bends_idx1,bends_yaw1,c='black',s=10,alpha=0.5,label='bends')
            plt.plot()
            plt.legend()
            plt.show()
        return bends_idx1.flatten()
    
    def get_line_idx(self,pad=3):
        """ 
        :param bends_idx (np.ndarray of int): corresponds to the indices where abrupt change of angle detected
        :param pad (int): pad indices [x] after/before start/stop indices
        """
        yaw = self.df['dls-yaw']
        bends_idx = self.calculate_bends(yaw,plot=False)
        idx_list = []
        for i in range(bends_idx.shape[0]-1):
            start_idx = bends_idx[i] + pad
            stop_idx = bends_idx[i+1] -pad
            idx_list.append((start_idx,stop_idx))
        return idx_list
    
    def get_coord_yaw(self):
        """ returns a tuple of (lat_long_array (np.ndarray), yaw (np.ndarray))"""
        idx_list = self.get_line_idx()
        lat_long_list = dict()
        yaw_list = dict()
        for idx in idx_list:
            df = self.df.iloc[idx[0]:idx[1]+1,:]
            #first column is latitude, second column is longitude (y,x)
            lat_long_list[f'{idx[0]}_{idx[1]}'] = df[['latitude','longitude']].to_numpy()
            yaw_list[f'{idx[0]}_{idx[1]}'] = df['dls-yaw'].values
            
        return lat_long_list,yaw_list
    
    def calculate_flight_angle(self, ref_vec = np.array([0,1])):
        # angle with respect to east np.array([0,1]), where latitude, longitude
        coord_yaw = self.get_coord_yaw()
        lat_long_array = coord_yaw[0] #first item of the tuple
        yaw_array = coord_yaw[1] #second item of the tuple
        angle_array_list = dict()
        yaw_array_list = dict()
        # for i in range(len(lat_long_array)):
        for k,v in lat_long_array.items():
            vec = np.diff(v,axis=0)
            vec_mag = np.linalg.norm(vec,axis=1)
            vec_mag = np.tile(vec_mag.reshape(-1,1),(1,2))
            vec = vec/vec_mag #normalised vector
            angle_array = np.arccos(np.dot(vec,ref_vec))
            angle_array_list[k] = angle_array
            yaw_array_list[k] = yaw_array[k][:-1]
            # angle_array_list.append(angle_array)
        return angle_array_list, yaw_array_list
    
    def plot_flight_angle(self):
        angle_array_list, yaw_array_list = self.calculate_flight_angle()
        plt.figure()
        for k in angle_array_list.keys():
            plt.scatter(np.mean(yaw_array_list[k]),np.mean(angle_array_list[k]),s=2,alpha=0.5,label=k)
        plt.xlabel('yaw')
        plt.ylabel('Flight angle')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        return
    
    def save_coord_yaw(self,fp=None):
        angle_array_list, yaw_array_list = self.calculate_flight_angle()
        angle = np.concatenate(list(angle_array_list.values()),axis=0)
        yaw = np.concatenate(list(yaw_array_list.values()),axis=0)
        coord_yaw = np.column_stack([angle,yaw])
        if fp is not None:
            dir = os.path.join(os.getcwd(),f'{fp}.ob')
            with open(dir,'wb') as fp:
                pickle.dump(coord_yaw,fp)
        return coord_yaw
    
def get_flight_angle(coords, ref_vec = np.array([1,0])):
    """ 
    :param coords (np.ndarray): where 1st column is latitude, 2nd column is longitude
        1st row is startpt of vector, 2nd row is endpt of vector
    :param ref_vec (np.ndarray): angle with respect to north np.array([1,0]), where latitude, longitude
    returns angle in degrees wrt to north based on vector between coord0 and coord1
    """
    vec = np.diff(coords,axis=0)
    vec_mag = np.linalg.norm(vec,axis=1)
    vec_mag = np.tile(vec_mag.reshape(-1,1),(1,2))
    vec = vec/vec_mag #normalised vector
    angle_array = np.arccos(np.dot(vec,ref_vec))
    return angle_array[0]/np.pi*180

def get_flight_direction(coords):
    """ returns dot product wrt to north vector and east vector"""
    vec = np.diff(coords,axis=0).flatten() #lat, lon
    vec_north = np.array([1,0])
    vec_east = np.array([0,1])
    return np.sign(np.dot(vec,vec_north)),np.sign(np.dot(vec,vec_east))

def get_flight_angle_fn():
    """
    returns function and params
    """
    t0 = (-np.pi,1.5)
    t1 = (-1.5,np.pi)
    t2 = (1.5,0)
    t3 = (3,1.5) #np.pi,1.5

    k0 = (t1[1]-t0[1])/(t1[0]-t0[0])
    k1 = (t2[1]-t1[1])/(t2[0]-t1[0])
    k2 = (t3[1]-t2[1])/(t3[0]-t2[0])

    c0 = t0[1] - k0*t0[0]
    c1 = t1[1] - k1*t1[0]
    c2 = t2[1] - k2*t2[0]

    x0 = -1.5
    x1 = 1.5

    fn = lambda x: k0*x+c0 if x<x0 else (k1*x+c1 if (x<x1) else k2*x+c2)

    return fn, [k0,k1,k2,c0,c1,c2]

class GeotransformImage:
    def __init__(self,im,lat,long,altitude,angle):
        """"
        :param im (np.ndarray): original raw image
        :param lat (float): latitude
        :param long (float): longitude
        :param altitude (float): height above ground
        :param angle (float): angle in degrees relative to east (x=1,y=0)
        """
        self.im = im
        self.lat = lat
        self.long = long
        self.altitude = altitude
        self.angle = angle
    
    def get_ground_resolution(self):
        """ returns ground resolution in meters in x, y direction"""
        return mutils.get_ground_resolution(height=self.altitude)
    
    def get_degrees_per_meter(self):
        """ 
        use the quick and dirty method that:
        111,111 meters (111.111 km) in the y direction is 1 degree (of latitude) and 
        111,111 * cos(latitude in degrees) meters in the x direction is 1 degree (of longitude).
        x_degrees: degree per meters in the longitude direction
        y_degrees: degree per meters in the latitude direction
        returns (y_degrees, x_degrees) which corresponds lat, long degrees resolution per meter
        """
        y_degrees = 1/111111
        x_degrees = 1/(111111*np.cos(self.lat/180*np.pi))
        return y_degrees, x_degrees 
    
    def get_degrees_per_pixel(self):
        """ 
        returns (y_degrees, x_degrees) which corresponds lat, long degrees resolution per pixel
        """
        x_meters, y_meters = self.get_ground_resolution() # resolution per pixel in meters
        y_degrees, x_degrees = self.get_degrees_per_meter() # degree resolution per meters
        avg_meter_res = (x_meters+y_meters)/2
        return avg_meter_res*y_degrees, avg_meter_res*x_degrees
    
    def geotransform(self):
        """
        # GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
        # GT(1) w-e pixel resolution / pixel width.
        # GT(2) row rotation (typically zero).
        # GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
        # GT(4) column rotation (typically zero).
        # GT(5) n-s pixel resolution / pixel height (negative value for a north-up image).
        geotransform = (left_extent, -lon_res_per_pixel, 0, top_extent, 0, -lat_res_per_pixel) #xmin=longitude, ymax=lat
        returns geotransformed image, and geotransform params
        """
        rot_im = self.affine_transformation(plot=False) #a north-up image
        rows, cols = rot_im.shape[0],rot_im.shape[1] 
        lat_res, lon_res = self.get_degrees_per_pixel()
        y_extent = self.lat+lat_res*int(0.5*rows)
        x_extent = self.long+lon_res*int(0.5*cols) # may need to find the upper right extent instead
        UL = (y_extent, x_extent)#lat, long
        geotransform = (UL[1],-lon_res,0,UL[0],0,-lat_res)
        # print(f'Geotransform: {geotransform}')
    
        return rot_im, geotransform
    
    def georegister(self, fn=None):
        """ 
        :param fn (str): file name without file extension
        assign coordinates to geotransformed image
        """
        fp = f'{fn}.tif'
        rot_im, geotransform = self.geotransform()
        flipped_transformed_img = np.fliplr(rot_im) #flip images because QGIS
        rows, cols = flipped_transformed_img.shape[0],flipped_transformed_img.shape[1]
        n_bands = self.im.shape[-1] if (len(self.im.shape) == 3) else 1

        if fn is not None:
            dst_ds = gdal.GetDriverByName('GTiff').Create(fp,cols, rows, n_bands, gdal.GDT_Float32)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference()            # establish encoding
            srs.ImportFromEPSG(4326)                # WGS84 lat/long
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            
            # flipped_transformed_img = arr
            if len(self.im.shape) == 3: #then it has 3-bands
                for i in range(3):
                    dst_ds.GetRasterBand(i+1).WriteArray(flipped_transformed_img[:,:,i])
        
            else: #greyscale image with 1 bands
                dst_ds.GetRasterBand(1).WriteArray(flipped_transformed_img)
                # write 1-band to the raster
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None
        return flipped_transformed_img

    def affine_transformation(self, plot = True):
        """angle in degrees. angle is positive for anti-clockwise and negative for clockwise in cv2.getRotationMatrix2D"""
        angle = self.angle

        rows, cols = self.im.shape[0],self.im.shape[1] 
        center = (cols//2,rows//2)
        # using cv2.getRotationMatrix2D() to get the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center,angle,1) #center, angle, scale
        # rotate the image using cv2.warpAffine
        # rotated_image = cv2.warpAffine(src=img, M=rotation_matrix, dsize=(width, height))
        cosofRotationMatrix = np.abs(rotation_matrix[0][0]) #scale*cos(angle)
        sinofRotationMatrix = np.abs(rotation_matrix[0][1]) #scale*sin(angle)
        # Now will compute new height & width of
        # an image so that we can use it in
        # warpAffine function to prevent cropping of image sides
        newImageHeight = int((cols * sinofRotationMatrix) +
                            (rows * cosofRotationMatrix))
        newImageWidth = int((cols * cosofRotationMatrix) +
                            (rows * sinofRotationMatrix))
        # After computing the new height & width of an image
        # we also need to update the values of rotation matrix
        rotation_matrix[0][2] += (newImageWidth/2) - center[0]
        rotation_matrix[1][2] += (newImageHeight/2) - center[1]

        # Now, we will perform actual image rotation WITHOUT MASK
        rotatingimage = cv2.warpAffine(self.im, rotation_matrix, (newImageWidth, newImageHeight))

        if plot is True:
            fig, axes = plt.subplots(1,2,figsize=(10,5))
            axes[0].imshow(self.im)
            axes[1].imshow(rotatingimage)
            axes[1].set_title(f'\nangle: {angle:.2f}')
            plt.show()
        return rotatingimage

def interpolate_timestamp(df,column_names = ['latitude','longitude','altitude','flight_angle','north_vec','east_vec'], milliseconds=100):
    """
    :param df (pd.DataFrame): column names must include timestamp, latitude, longitude, altitude, flight_angle
    :param column_names (list of str): list of columns to interpolate
    :param milliseconds (float): timedelta in milliseconds, interpolate between two timestamps at 100ms frequency
    linear interpolation using timedelta
    returns a tuple of (original dataframe, interpolated dataframe)
    """
    if not all([i in df.columns for i in column_names]):
        raise NameError(f'at least one name does not exist: {column_names}')
    
    middle_idx = len(df.index)//2
    freq = datetime.timedelta(milliseconds=milliseconds)#duration//10 # timedelta object
    timedelta = freq.total_seconds()*1000
    date_range_list = []
    for i, rows in df.iterrows():
        if i == len(df.index) -1:
            pass
        else:
            daterange = pd.date_range(start=rows['timestamp'], end=df['timestamp'][i+1], freq=freq).to_series() # date arange,,closed='left'
            daterange = daterange.reset_index()
            date_range_list.append(daterange.iloc[:,0])
    
    interpolated_timestamp = pd.concat(date_range_list)
    # print(interpolated_timestamp)
    timedelta_series = interpolated_timestamp - df['timestamp'][0]
    timedelta_series = timedelta_series.dt.total_seconds() #returns time delta in total_seconds

    og_timestamp = df['timestamp']
    og_timedelta_series = og_timestamp - df['timestamp'][0]
    og_timedelta_series = og_timedelta_series.dt.total_seconds()
    og_dict = {'timestamp': og_timestamp, 'timedelta': og_timedelta_series}
    for names in column_names:
        og_dict[names] = df[names]
    og_df = pd.DataFrame(og_dict)
    
    interpolated_dict = {'timestamp': interpolated_timestamp,'timedelta':timedelta_series}
    for names in column_names:
        interpolated_dict[names] = griddata(og_timedelta_series, df[names], timedelta_series, method='linear')
    interpolated_df = pd.DataFrame(interpolated_dict)
    
    return og_df, interpolated_df

class InterpolateCoordinates:
    def __init__(self, df,method='linear', interpolate_distance = 0.1, pad_distance = 1, estimate_angle="yaw"):
        """ 
        :param method (str): linear, nearest, cubic
        :param pad_distance (float): extend values backwards and forwards by pad_distance
        :param estimate_angle (str): yaw or coord
        sampling equidistantly based on fixed time intervals between 2 points should be avoided 
        because where the distance is larger for 2 points, the correction in GPS will be greater too
        the image will be shifted for pictures with smaller distance between 2 points
        """
        self.df = df
        self.method = method
        self.interpolate_distance = interpolate_distance
        self.pad_distance = pad_distance
        self.lat = self.df['latitude'].to_numpy()#.reshape(-1,1)
        self.lon = self.df['longitude'].to_numpy()#.reshape(-1,1)
        self.estimate_angle = estimate_angle
    
    def get_cum_distance(self):
        coord_stack = np.hstack([self.lat.reshape(-1,1),self.lon.reshape(-1,1)])
        distance = np.sum(np.diff(coord_stack,axis=0)**2,axis=1) #distance between two adjacent points
        distance = distance**(1/2)
        distance_median = np.median(distance) # median distance since stationary GPS points will skew when calculating avg
        distance_normalised = distance/distance_median #normalise the distance so the median distance between images is 1
        distance_cum_sum = np.cumsum(distance_normalised)
        # append 0 to cum sum array since it has one less data point than lat/lon since distance is calculated between the coords
        distance_cum_sum = np.append([0],distance_cum_sum)
        return distance_cum_sum
    
    def interpolate_coordinates(self):
        """ 
        :param pad_distance (float): for the points with only one adjacent values (e.g. first and last points), pad those with pad_distance for interpolation between 2 points
        """
        cum_distance = self.get_cum_distance()
        n = cum_distance.shape[0]
        # initialise dict
        interp_dist_dict = dict() #where keys correspond to the image index (may not necessarily be image name if some images are missing)
        # interp_dist_dict[0] = {'latitude': self.lat[0], 'longitude':self.lon[0]}
        # interp_dist_dict[n-1] = {'latitude': self.lat[n-1], 'longitude':self.lon[n-1]}

        def interpolate_lat(x):
            return np.interp(x, cum_distance, self.lat)
        
        def interpolate_lon(x):
            return np.interp(x, cum_distance, self.lon)
        
        for i in range(n):
            # prev_dist = cum_distance[i-1]
            current_dist = cum_distance[i]
            # next_dist = cum_distance[i+1]
            # interpolation between adjacent coordinates may not be ideal 
            # if two points are very close together because the drone is hovering
            # so the interpolation may not be much, we cant shift the images much forward/backwards
            
            # interpolate the distance backward
            backward_interpolated_x = np.arange(current_dist-self.pad_distance,current_dist, self.interpolate_distance)
            # interpolate distance forward
            forward_interpolated_x = np.arange(current_dist,current_dist+self.pad_distance,self.interpolate_distance)
            # keep track of index that maps to the original cum distance, this helps us to easily shift the coordinates wrt to the original index
            # indexOfCurrentCoord = backward_interpolated_x.shape[0] #since pad_distance is the same, the original index is always in the middle of the array
            # concat the interpolated values so interpolation can be done at one go
            interpolated_x = np.hstack([backward_interpolated_x,forward_interpolated_x])
            
            # dist_x = cum_distance[i-1:i+2]
            # lat_y = self.lat[i-1:i+2]
            # lon_y = self.lon[i-1:i+2]
            interpolated_lat = interpolate_lat(interpolated_x)
            interpolated_lon = interpolate_lon(interpolated_x)
            interp_dist_dict[i] = {'latitude': interpolated_lat, 'longitude': interpolated_lon}

        return interp_dist_dict
    
    def append_flight_angle(self):
        """calculate flight angle (in degrees) and append to the df"""
        if self.estimate_angle == "yaw":
            self.df['flight_angle'] = -self.df['dls-yaw']/np.pi*180
        else:
            column_idx = [i for i,c in enumerate(self.df.columns.to_list()) if c in ['latitude','longitude']]
            angle_coord_list = []
            for i,rows in tqdm(self.df.iterrows()):
                if (i == 0) or (i == len(self.df.index)-1):
                    # since the first and last points do not have adjacent points, use yaw instead
                    angle_coord_list.append(-rows['dls-yaw']/np.pi*180)
                else:
                    # estimate flight angle from 2 adjacent coordinates
                    flight_att_diff = self.df.iloc[[i-1,i+1],column_idx]
                    flight_att_diff = flight_att_diff.iloc[:,:2].values
                    flight_angle_coord = get_flight_angle(flight_att_diff)
                    north_vec, east_vec = get_flight_direction(flight_att_diff)
                    if east_vec < 0:
                        flight_angle = flight_angle_coord # means rotation of image is positive (acw)
                    else:
                        flight_angle = -flight_angle_coord # means rotation of image is negative (cw)
                    angle_coord_list.append(flight_angle)
            self.df['flight_angle'] = angle_coord_list
    
    def get_interpolated_dict(self):
        """ returns a dict of interpolated coordinates per image index"""
        # get flight_angle
        self.append_flight_angle()
        return self.interpolate_coordinates()
    
    def shift_coord(self,interp_dist_dict,shift_n=0):
        """ 
        :param interp_dist_dict (dict): output from
        :param shift_n (int): if positive, interpolate forward, if negative, interpolate backwards
        """
        originalIndex = interp_dist_dict[0]['latitude'].shape[0]//2
        
        assert abs(shift_n) < originalIndex, f'shift_n: {shift_n} is > interpolated length: {originalIndex}. Select shift_n to be < {originalIndex} or reduce interpolate_distance or increase pad_distance'
        
        lat_list = [coord_dict['latitude'][originalIndex+shift_n] for coord_dict in interp_dist_dict.values()]
        lon_list = [coord_dict['longitude'][originalIndex+shift_n] for coord_dict in interp_dist_dict.values()]

        df_copy = self.df.copy()
        df_copy['latitude'] = lat_list
        df_copy['longitude'] = lon_list
        
        return df_copy
    
class InterpolateFlight:
    def __init__(self, df, interpolate_milliseconds=100, estimate_angle = "yaw",column_names = ['latitude','longitude','altitude','flight_angle','north_vec','east_vec']):
        """ 
        :param df (pd.DataFrame): imported from the folder 'flight_attributes'
        :param interpolate_milliseconds (int): interpolate timeseries every [x] milliseconds
        :param estimate_angle (str): "yaw" or "coord". either use yaw directly or calculate flight angle from coords
        :param column_names (list of str): column names in df
        """
        self.df = df
        self.interpolate_milliseconds = interpolate_milliseconds
        self.estimate_angle = estimate_angle
        self.column_names = column_names
    
    def append_flight_angle(self,df):
        """calculate flight angle (in degrees) and append to the df"""
        if self.estimate_angle == "yaw":
            df['flight_angle'] = -df['dls-yaw']/np.pi*180
        else:
            column_idx = [i for i,c in enumerate(df.columns.to_list()) if c in ['latitude','longitude']]
            angle_coord_list = []
            for i,rows in tqdm(df.iterrows()):
                if (i == 0) or (i == len(df.index)-1):
                    angle_coord_list.append(np.NaN)
                    pass
                else:
                    # estimate flight angle from 2 adjacent coordinates
                    flight_att_diff = df.iloc[[i-1,i+1],column_idx]
                    flight_att_diff = flight_att_diff.iloc[:,:2].values
                    flight_angle_coord = get_flight_angle(flight_att_diff)
                    north_vec, east_vec = get_flight_direction(flight_att_diff)
                    if east_vec < 0:
                        flight_angle = flight_angle_coord # means rotation of image is positive (acw)
                    else:
                        flight_angle = -flight_angle_coord # means rotation of image is negative (cw)
                    angle_coord_list.append(flight_angle)
            # angle_coord_list.append(np.NaN)
            df['flight_angle'] = angle_coord_list
        
        df = df.ffill(axis=0).bfill(axis=0) #fill forward and fill backward
        return df
    
    def calculate_flight_direction(self, df):
        """ append north and east vec to df"""
        
        column_idx = [i for i,c in enumerate(df.columns.to_list()) if c in ['latitude','longitude']]
        north_vec_list = []
        east_vec_list = []
        for i,rows in tqdm(df.iterrows()):
            if (i == 0) or (i == len(df.index)-1):
                north_vec_list.append(np.NaN)
                east_vec_list.append(np.NaN)
                pass
            else:
                # estimate flight angle from 2 adjacent coordinates
                flight_att_diff = df.iloc[[i-1,i+1],column_idx]
                flight_att_diff = flight_att_diff.iloc[:,:2].values
                flight_dir = get_flight_direction(flight_att_diff)
                north_vec_list.append(flight_dir[0])
                east_vec_list.append(flight_dir[1])
        
        df['north_vec'] = north_vec_list
        df['east_vec'] = east_vec_list
        df = df.ffill(axis=0).bfill(axis=0) #fill forward and fill backward
        return df

    def interpolate_flight(self, plot=True):
        """returns an interpolated df"""
        flight_attributes_df = self.append_flight_angle(self.df)
        flight_attributes_df = self.calculate_flight_direction(flight_attributes_df)
        # convert to datetime format
        flight_attributes_df['timestamp'] = pd.to_datetime(flight_attributes_df['timestamp'])
        # interpolate df
        df, df_interpolated = interpolate_timestamp(flight_attributes_df,milliseconds=self.interpolate_milliseconds,
                                                    column_names = ['latitude','longitude','altitude','flight_angle','north_vec','east_vec'])
        # df_interpolated = self.append_flight_angle(df_interpolated)
        # common_columns = set(df_interpolated.columns.to_list()).intersection(set(flight_attributes_df.columns.to_list()))
        # common_columns = list(sorted(common_columns))
        df_interpolated = df_interpolated.merge(flight_attributes_df,how='outer',on=['timestamp']+self.column_names).ffill(axis=0)
        # df_interpolated = self.calculate_flight_direction(df_interpolated)
        # assign unique index for image name
        df_interpolated['index'] = df_interpolated['image_name'].str.split('_').str[1].astype(int)
        
        if plot is True:
            fig, axes = plt.subplots(1,2,figsize=(10,5))
            axes[0].scatter(df['longitude'],df['latitude'],c=df['flight_angle'],alpha=0.5)
            im = axes[1].scatter(df_interpolated['longitude'],df_interpolated['latitude'],c=df_interpolated['flight_angle'],alpha=0.5)
            cax = plt.colorbar(im,ax=axes[1])
            cax.set_label('Flight angle (deg)')
            axes[0].set_title(f'Flight angle ({len(df.index)})')
            axes[1].set_title(f'Interpolated flight angle ({len(df_interpolated.index)})')
            for ax in axes:
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
            plt.tight_layout()
            plt.show()
        
        return df_interpolated
    
class PlotGeoreference:
    def __init__(self, imagePath, flight_attributes_df, wql_dict = None):
        """ 
        :param imagePath (str): directory path where images are stored
        :param flight_attributes_df (pd.DataFrame):  dataframe with image_name, flight angle, north_vec, east_vec e.g. df_interpolated
        :param fp_list (list of fp): filepath of the thumbnail
        :param wql_dict (dict): where keys are: lat, lon, measurements
        :param geotransform_list (dict): 
            where key is the int index of the image
            where each value is a dict of keys:'lat','lon','lat_res','lon_res'
        :param im_list (dict): 
            where key is the int index of the image
            where each value is an image
        returns an np.ndarray
        """
        self.imagePath = imagePath
        self.flight_attributes_df = flight_attributes_df
        self.wql_dict = wql_dict
        self.rgb_bands = [2,1,0]

    def normalise_image(self,im_rgb,p_min=0.1,p_max=95):
        """ 
        :param im_rgb (np.ndarray): rgb image array
        """
        # get normalised rgb image
        im_min = np.percentile(im_rgb.flatten(), p_min)  # modify with these percentilse to adjust contrast
        im_max = np.percentile(im_rgb.flatten(), p_max)  # for many images, 0.5 and 99.5 are good values
        
        im_display = np.zeros(im_rgb.shape)
        for i in range(im_rgb.shape[2]):
            im_display[:,:,i] = imageutils.normalize(im_rgb[:,:,i], im_min, im_max)
        
        return im_display

    def get_flight_attributes(self):
        """ 
        returns a dict, where keys are image_index that corresponds to image_name,
            values are lat, lon, lat_res, lon_res, flight_angle, image_fp
        """
        # column_idx = [i for i,c in enumerate(self.flight_attributes_df.columns.to_list()) if c in ['image_name','latitude','longitude','altitude','flight_angle']]

        # im_list = dict()
        geotransform_list = dict()

        for i,rows in self.flight_attributes_df.iterrows():
            
            lat = rows['latitude']
            lon = rows['longitude']
            altitude = rows['altitude']
            flight_angle = rows['flight_angle']

            image_name = os.path.splitext(rows['image_name'])[0] #image name without extension
            image_fp = os.path.join(self.imagePath,'thumbnails',f'{image_name}.jpg')
            img_idx = int(image_name.split('_')[1])

            GI = GeotransformImage(None, lat, lon, altitude = altitude, angle = flight_angle)
            
            lat_res, lon_res = GI.get_degrees_per_pixel()
            
            geotransform_list[img_idx] = {'lat':lat,'lon':lon,'lat_res':lat_res,'lon_res':lon_res,
                                    'flight_angle': flight_angle, 'image_fp':image_fp}
            
        return geotransform_list

    def get_canvas(self,scale_factor=1.05):
        geotransform_list = self.get_flight_attributes() # returns a dict on attributes of each im
        idx = 0
        lat_max = [idx,0]
        lat_min = [idx,180]
        lon_max = [idx,0]
        lon_min = [idx,180]
        pixel_res = 1
        # get the max and min lat and lon values, and the corresponding im idx
        for idx, gt in geotransform_list.items():
            if gt['lat'] > lat_max[1]:
                lat_max = [idx,gt['lat']]
            if gt['lat'] < lat_min[1]:
                lat_min = [idx,gt['lat']]
            if gt['lon'] > lon_max[1]:
                lon_max = [idx,gt['lon']]
            if gt['lon'] < lon_min[1]:
                lon_min = [idx, gt['lon']]
            if gt['lat_res'] < pixel_res:
                pixel_res = gt['lat_res']
            if gt['lon_res'] < pixel_res:
                pixel_res = gt['lon_res']
        
        self.pixel_res = pixel_res
        
        im_list = dict()
        for im_type, coord_type in zip(['upper_lat','lower_lat','left_lon','right_lon'],[lat_max,lat_min,lon_min,lon_max]):
            # fp = self.fp_list[coord_type[0]]
            fp = geotransform_list[coord_type[0]]['image_fp']
            if fp.endswith('.tif') or fp.endswith('.jpg'):
                im = np.asarray(Image.open(fp))
            elif fp.endswith('.ob'):
                im = mutils.load_pickle(fp)
                im = np.take(im,self.rgb_bands,axis=2)
            GI = GeotransformImage(im,None,None,None,angle=geotransform_list[coord_type[0]]['flight_angle'])
            rot_im = GI.affine_transformation(plot=False)
            im_list[im_type] = rot_im

        upper_lat = lat_max[1] + ceil(im_list['upper_lat'].shape[0]/2)*pixel_res
        lower_lat = lat_min[1] - ceil(im_list['lower_lat'].shape[0]/2)*pixel_res
        left_lon = lon_min[1] - ceil(im_list['left_lon'].shape[1]/2)*pixel_res
        right_lon = lon_max[1] + ceil(im_list['right_lon'].shape[1]/2)*pixel_res

        self.upper_lat = upper_lat
        self.lower_lat = lower_lat
        self.left_lon = left_lon
        self.right_lon = right_lon

        nrow = ceil(scale_factor*(upper_lat - lower_lat)/pixel_res)
        ncol = ceil(scale_factor*(right_lon - left_lon)/pixel_res)
        if fp.endswith('.tif') or fp.endswith('.jpg'):
            im_display = np.zeros((nrow,ncol,3),dtype=np.uint8) #includes alpha channel
        elif fp.endswith('.ob'):
            im_display = np.zeros((nrow,ncol,3), dtype=np.float64)
        print(f'shape of canvas{im_display.shape}')
        return im_display
    
    def get_row_col_index(self, lat, lon, lat_res, lon_res, rot_im):
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

    def plot_wql(self, axis = None,**kwargs):
        if self.wql_dict is None:
            return None
        else:
            if axis is None:
                im_display = self.get_canvas()
            tss_lat = self.wql_dict['lat']
            tss_lon = self.wql_dict['lon']
            tss_measurements = self.wql_dict['measurements']
            rows_idx = []
            cols_idx = []
            tss_idx = []
            for i in range(len(tss_lat)):
                lat = tss_lat[i]
                lon = tss_lon[i]
                if lat > self.upper_lat or lat < self.lower_lat:
                    continue
                if lon > self.right_lon or lon < self.left_lon:
                    continue
                row_idx = int((self.upper_lat - lat)/self.pixel_res)
                col_idx = int((lon - self.left_lon)/self.pixel_res)
                rows_idx.append(row_idx)
                cols_idx.append(col_idx)
                tss_idx.append(tss_measurements[i])

            if axis is None:
                fig,axis = plt.subplots(figsize=(7,10))
                if im_display.dtype == 'uint8':
                    im_display[im_display == 0] = 255
                else:
                    im_display[im_display == 0] = 1.0
                axis.imshow(im_display)
            im = axis.scatter(cols_idx,rows_idx,c=tss_idx,alpha=0.5,label='in-situ sampling',**kwargs)
            axis.legend(loc='lower center',bbox_to_anchor=(0.5,-0.2),prop={'size': 16})
            axcb = plt.colorbar(im,ax=axis)
            axcb.set_label('Turbidity (NTU)')
            if axis is None:
                plt.show()

            return

    def plot_georeference(self, reduction_factor = 5, plot = True, add_wql=False, normalise=True,p_min=0.1,p_max=95,axis=None,**kwargs):
        """
        :param reduction_factor (int): factor to reduce image by to speed up display of pic
        :param add_wql (bool): whether or not to add wql points
        :param axis (Axes class): plot on a supplied axis instead, if None, create a new fig
        :param normalise (bool): whether to normalise image or not
        :param p_min (float): min percentile value for contrast stretching
        :param p_max (float): max percentile value for contrast stretching
        """
        im_display = self.get_canvas()
        geotransform_list = self.get_flight_attributes()
        # column_idx = [i for i,c in enumerate(self.flight_attributes_df.columns.to_list()) if c in ['latitude','longitude']]
        for i, rows in self.flight_attributes_df.iterrows():
            flight_angle = rows['flight_angle']

            image_name = os.path.splitext(rows['image_name'])[0] #image name without extension
            fp = os.path.join(self.imagePath,'thumbnails',f'{image_name}.jpg')
            img_idx = int(image_name.split('_')[1])

            # print(rows['image_name'], fp)
            if fp.endswith('.tif') or fp.endswith('.jpg'):
                im = np.asarray(Image.open(fp)) if (image_name in fp) else None
            elif fp.endswith('.ob'):
                im = mutils.load_pickle(fp)
                im = np.take(im,self.rgb_bands,axis=2)
            if im is None:
                raise NameError("image is None because filepath does not match image name")
            GI = GeotransformImage(im,None,None,None,angle=flight_angle)
            rot_im = GI.affine_transformation(plot=False)
            # rot_im = np.fliplr(np.flipud(rot_im))
            
            # print(rot_im.shape)
            att = geotransform_list[img_idx]
            upper_row_idx, lower_row_idx, left_col_idx, right_col_idx = self.get_row_col_index(att['lat'],att['lon'],att['lat_res'],att['lon_res'],rot_im) #row/col idx wrt to center coord
            # print(upper_row_idx, lower_row_idx, left_col_idx, right_col_idx)
            background_im = im_display[upper_row_idx:lower_row_idx,left_col_idx:right_col_idx,:]
            assert rot_im.shape == background_im.shape, f'shapes are diff {rot_im.shape} {background_im.shape}'
            overlay_im = np.where(rot_im == 0, background_im,rot_im)
            im_display[upper_row_idx:lower_row_idx,left_col_idx:right_col_idx,:] = overlay_im

        nrow, ncol = im_display.shape[0], im_display.shape[1]
        # resize image by specifying custom width and height
        
        if plot is True:
            if im_display.dtype == 'uint8':
                im_display[im_display == 0] = 255
            else:
                im_display[im_display == 0] = 1.0

            if normalise is True:
                im_display = self.normalise_image(im_display, p_min,p_max)

            resized = cv2.resize(im_display, (ncol//reduction_factor, nrow//reduction_factor))

            if axis is None:
                fig, axis = plt.subplots(figsize=(15,10))

            if add_wql is False:
                axis.imshow(resized)
            else:
                if self.wql_dict is not None:
                    axis.imshow(im_display)
                    self.plot_wql(axis = axis,**kwargs)
                    axis.axis('off')

            if axis is None:
                plt.show()
            return resized
        else:
            return im_display

def time_delta_correction(df_interpolated,columns_to_shift = ['timestamp','timedelta','latitude','longitude','altitude','flight_angle','north_vec','east_vec'], timedelta = 0.1):
    """ 
    :param timedelta (float): timedelta in multiples of 0.1 seconds (100 millisecond), can be negative/positive
    :param columns_to_shift (list of str): columns to time shift
    """
    df = df_interpolated.copy()
    rows_shift = int(timedelta/0.1)
    print(f'rows shifted: {rows_shift}')
    
    df[columns_to_shift] = df[columns_to_shift].shift(rows_shift)

    return df.groupby('image_name').nth(0).reset_index()

class ExtractInsituSpectral:
    def __init__(self, flight_attributes_df, fp_list, wql_dict = None,DEM_offset_height = 15, radius = 1):
        """ 
        :param flight_attributes_df (pd.DataFrame):  dataframe with flight angle, north_vec, east_vec e.g. df_interpolated
        :param fp_list (list of fp): filepath of the thumbnail
        :param wql_dict (dict): where keys are: lat, lon, measurements
        :param geotransform_list (dict): 
            where key is the int index of the image
            where each value is a dict of keys:'lat','lon','lat_res','lon_res'
        :param im_list (dict): 
            where key is the int index of the image
            where each value is an image
        returns an np.ndarray
        """
        self.flight_attributes_df = flight_attributes_df
        self.wql_dict = wql_dict
        # where keys are image index extracted from the image_name from fp_list
        self.fp_list = {int(os.path.splitext(os.path.split(fp)[-1])[0].split('_')[1]):fp for fp in fp_list}
        self.DEM_offset_height = DEM_offset_height
        PG = PlotGeoreference(flight_attributes_df, fp_list,wql_dict,DEM_offset_height)
        self.geotransform_list = PG.get_flight_attributes()
        self.radius = int(radius*10)
    
    def get_row_col_index(self, row_idx, col_idx, nrow, ncol):
        pad = self.radius
        upper_row_idx = row_idx - pad
        lower_row_idx = row_idx + pad
        left_col_idx = col_idx - pad
        right_col_idx = col_idx + pad
        
        upper_row_idx = 0 if upper_row_idx < 0 else upper_row_idx
        lower_row_idx = nrow if lower_row_idx > nrow else lower_row_idx
        left_col_idx = 0 if left_col_idx < 0 else left_col_idx
        right_col_idx = ncol if right_col_idx > ncol else right_col_idx

        return upper_row_idx, lower_row_idx, left_col_idx, right_col_idx
    
    def check_within_bounding_box(self, att, rot_im):
        """ 
        :param att (dictionary): with keys: lat, lon, lat_res, lon_res
        """
        nrow, ncol = rot_im.shape[0], rot_im.shape[1]
        
        upper_lat = att['lat'] + ceil(nrow/2)*att['lat_res']
        lower_lat = att['lat'] - ceil(nrow/2)*att['lat_res']
        left_lon = att['lon'] - ceil(ncol/2)*att['lon_res']
        right_lon = att['lon'] + ceil(ncol/2)*att['lon_res']

        # print(upper_lat,lower_lat,left_lon,right_lon)
        tss_lat = self.wql_dict['lat']
        tss_lon = self.wql_dict['lon']
        tss_measurements = self.wql_dict['measurements']

        rows_idx = []
        cols_idx = []
        # tss_idx = []
        # extracted_spectral_list = []
        TSS_df_dict = dict()
        for i in range(len(tss_lat)):
            lat = tss_lat[i]
            lon = tss_lon[i]
            
            if lat > upper_lat or lat < lower_lat:
                continue
            if lon > right_lon or lon < left_lon:
                continue

            row_idx = int((upper_lat - lat)/att['lat_res'])
            col_idx = int((lon - left_lon)/att['lon_res'])
            # print(f'row_idx: {row_idx}, col_idx: {col_idx}')
            upper_row_idx, lower_row_idx, left_col_idx, right_col_idx = self.get_row_col_index(row_idx, col_idx, nrow, ncol)
            ROI = rot_im[upper_row_idx:lower_row_idx,left_col_idx:right_col_idx,:]
            ROI_list = [ROI[:,:,i] for i in range(ROI.shape[2])]
            band_reflectance = [np.mean(i[i!=0]) for i in ROI_list] #remove 0s, then calculate the mean of the ROI for each layer
            TSS_df_dict[i] = [i, tss_measurements[i], lat, lon] + band_reflectance
            # extracted_spectral_list.append(extracted_spectral_mean)
            rows_idx.append(row_idx)
            cols_idx.append(col_idx)
            # tss_idx.append(tss_measurements[i])
        df_columns = ['observation_number','tss_conc','tss_lat','tss_lon'] + ['band_{}'.format(i) for i in range(rot_im.shape[2])]
        tss_df = pd.DataFrame.from_dict(TSS_df_dict,orient='index',columns=df_columns)

        return rows_idx, cols_idx, tss_df#rows_idx, cols_idx, tss_idx, extracted_spectral_list

    def get_reflectance_from_GPS(self, reflectance_fp, plot = True):
        """ 
        :param reflectance_fp (str): filepath of stacked reflectance image
        :param radius (int): take the average reflectance over a square radius of radius*10
        """
        img_idx = os.path.basename(reflectance_fp)
        img_idx = int(os.path.splitext(img_idx)[0].split('_')[1])
        
        assert img_idx in list(self.fp_list), 'reflectance_fp must correspond to fp_list variable'

        if reflectance_fp.endswith('.ob'):
            reflectance = mutils.load_pickle(reflectance_fp)
        else:
            reflectance = None
        
        assert reflectance is not None, 'reflectance image cannot be opened!'
        GI = GeotransformImage(reflectance,None,None,None,angle=self.geotransform_list[img_idx]['flight_angle'])
        rot_im = GI.affine_transformation(plot=False)
        # rot_im = np.fliplr(rot_im)
        att = self.geotransform_list[img_idx]

        rows_idx, cols_idx, tss_df = self.check_within_bounding_box(att, rot_im)
        # rows_idx, cols_idx, tss_idx, extracted_spectral_list = self.check_within_bounding_box(att, rot_im)
        if plot is True:
            rgb_indices = [2,1,0]
            plt.figure(figsize=(10,10))
            plt.imshow(np.take(rot_im,rgb_indices,axis=2))
            plt.scatter(cols_idx,rows_idx,c=tss_df['tss_conc'],alpha=0.5,label='in-situ sampling')
            plt.axis('off')
            plt.show()
        
        return tss_df
    
    def extract_spectral(self,save_fp=None):
        df_list = []
        for img_idx, fp in self.fp_list.items():
            tss_df = self.get_reflectance_from_GPS(fp, plot = False)
            tss_df.insert(1,'image_index',img_idx)#['image_index'] = img_idx
            df_list.append(tss_df)
        df = pd.concat(df_list)
        df.dropna(how='any',inplace=True)
        if save_fp is not None:
            df.to_csv(save_fp,index=False)
        return df

class CompareInsituSpectral:
    def __init__(self,image_indices, dir_list,df_interpolated,wql_dict,titles,timedelta=-1.5,DEM_offset_height=15):
        """ 
        :param image_indices (list of int): where each element in the list represents the image index that is in image name (IMG_0123)
        :param dir_list (list of str): full filepath of the directories where reflectance/corrected reflectance is saved
        :param df_interpolated (pd DataFrame): the interpolated df from InterpolateFlight.interpolate_flight()
        :param wql_dict (dict): dictionary of wql details where keys are lat, lon, measurements
        :param titles (list of str): titles for each subplot that corresponds to the algorithm
        :param timedelta (float): time shift for time delay correction for image alignment
        :param DEM_offset_height (float): offset height to measure altitude
        """
        assert len(titles) == len(dir_list), 'length of title must == length of directories'
        self.image_indices = image_indices
        self.dir_list = dir_list
        self.df_interpolated = df_interpolated
        self.wql_dict = wql_dict
        self.titles = titles
        self.timedelta = timedelta
        self.DEM_offset_height = DEM_offset_height
        df_cropped = time_delta_correction(self.df_interpolated, timedelta=self.timedelta)
        self.df_cropped = df_cropped.iloc[image_indices,:]
    
    def extract_spectral(self, save_dir = None):
        """
        :param save_dir (directory) to save extracted spectral_information
        """
        if (save_dir is not None) and (os.path.exists(os.path.join(save_dir,"Extracted_Spectral_Information")) is False):
            extracted_spectral_dir = os.path.join(save_dir,"Extracted_Spectral_Information")
            os.mkdir(extracted_spectral_dir)
        else:
            extracted_spectral_dir = None
            df_list = []
        for i, (reflectance_directory,fn) in enumerate(zip(self.dir_list, self.titles)):
            img_list = [os.path.join(reflectance_directory,f'IMG_{str(i).zfill(4)}_1.ob') for i in self.image_indices]
            EIS = ExtractInsituSpectral(self.df_cropped, img_list, self.wql_dict ,DEM_offset_height = self.DEM_offset_height)
            if extracted_spectral_dir is not None:
                _ = EIS.extract_spectral(os.path.join(extracted_spectral_dir,f'{i+1}_{fn}.csv'))
            else:
                df = EIS.extract_spectral()
                df_list.append(df)
        
        if extracted_spectral_dir is None:
            return df_list
        else:
            print(f'extracted spectral data saved in {extracted_spectral_dir}')
            return None

    def compare_rgb(self):
        
        fig, axes = plt.subplots(1,len(self.dir_list),figsize=(10,10))
        
        for d,dir in enumerate(self.dir_list):
            img_list = [os.path.join(dir,f'IMG_{str(i).zfill(4)}_1.ob') for i in self.image_indices]
            PG = PlotGeoreference(self.df_cropped,img_list,None,
                                                DEM_offset_height=self.DEM_offset_height)
            _ = PG.plot_georeference(reduction_factor = 5, plot = True, add_wql=False, axis=axes[d])

        for i,ax in enumerate(axes.flatten()):
            ax.axis('off')
            ax.set_title(self.titles[i])
        plt.show()
        return