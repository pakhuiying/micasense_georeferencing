import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import os
import PIL.Image as Image
import math
import cv2
import algorithms.flight_attributes as FlightAttributes
from algorithms.alignment_error import correlation_coefficient

def fancy_coord(coord, pos, neg):
    """
    Stringifies the point in a more fancy way than __repr__, e.g.
    "44°35'27.6"N 100°21'53.1"W", i.e. with minutes and seconds.
    lat = fancy_coord(lat, "N", "S")
    lon = fancy_coord(lon, "E", "W")
    """
    coord_dir = pos if coord > 0 else neg
    coord_tmp = abs(coord)
    coord_deg = math.floor(coord_tmp)
    coord_tmp = (coord_tmp - math.floor(coord_tmp)) * 60
    coord_min = math.floor(coord_tmp)
    coord_sec = round((coord_tmp - math.floor(coord_tmp)) * 600) / 10
    coord = f"{coord_deg}°{coord_min}'{coord_sec}\"{coord_dir}"
    return coord

class PlotMap:
    def __init__(self,ne,sw,img):
        """
        :param (img): basemap obtained from google map tiles
            +---+ ne
            |   |
        sw  +---+
        """
        self.ne = ne
        self.sw = sw
        self.img = img
        self.image_height = img.shape[0]
        self.image_width = img.shape[1]
        self.lat_res_per_pixel = (self.ne[0] - self.sw[0])/self.image_height
        self.lon_res_per_pixel = (self.ne[1] - self.sw[1])/self.image_width
        # TILE_SIZE = 256  # in pixels
        # EARTH_CIRCUMFERENCE = 40075.016686 * 1000  # in meters, at the equator
        # zoom = 20
        # meters_per_pixel_at_zoom_0 = ((EARTH_CIRCUMFERENCE / TILE_SIZE) * math.cos(math.radians(self.sw[0])))
        # # pixel resolution from google tile at zoom=20
        # meters_per_pixel = meters_per_pixel_at_zoom_0 / (2 ** zoom)
        self.metres_per_degree = 111177.1472578764 
        # pixel resolution for the image
        self.meters_per_pixel = (self.ne[1] - self.sw[1])*self.metres_per_degree/self.image_width

    def get_ticks(self,breaks = 6):
        y_breaks = x_breaks = breaks
        # position of ticks
        yticks = np.arange(0,self.image_height, int(self.image_height/y_breaks))
        xticks = np.arange(0,self.image_width, int(self.image_width/x_breaks))
        # ytick labels
        lat_ticks = [self.ne[0] - y*self.lat_res_per_pixel for y in yticks]
        lat_labels = [fancy_coord(l, "N", "S") for l in lat_ticks]
        # xtick labels
        lon_ticks = [self.sw[1] + x*self.lon_res_per_pixel for x in xticks]
        lon_labels = [fancy_coord(l, "E", "W") for l in lon_ticks]
        return ((yticks,lat_labels),(xticks,lon_labels))
    
    def buffer(self,linewidth):
        return [pe.withStroke(linewidth=linewidth, foreground="white")]
    
    def get_compass_location(self):
        # location of north compass
        yArrow = int(0.05*self.image_height)
        yN = yArrow + yArrow
        xArrow = int(0.05*self.image_width)
        xN = xArrow - int(0.01*self.image_width)
        return ((yN,xN),(yArrow,xArrow))

    def get_scale_bar_location(self,scale_unit_distance=10):
        pad = int(0.05*self.image_height)
        # length of scale bar, pixels to represent 10 metres
        pixelsScale = int(scale_unit_distance/self.meters_per_pixel)
        # location of scale bar
        yScale = int(0.85*self.image_height)
        xScale = int(0.85*self.image_width)
        xScaleEnd = xScale + pixelsScale
        # location of scale number
        yScaleNumber = yScale - pad//2
        xScaleNumber = xScale
        return ((xScale,xScaleEnd,yScale),(xScaleNumber,yScaleNumber))
    
    def add_ticks(self, ax, add_yticks=True, add_xticks=True,tick_breaks=6,fontsize=12):
        ((yticks,lat_labels),(xticks,lon_labels)) = self.get_ticks(tick_breaks)
        # set y ticks
        if add_yticks is True:
            ax.set_yticks(yticks)
            ax.set_yticklabels(lat_labels)
            ax.tick_params(axis="y",labelsize=fontsize)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
        # set x ticks
        if add_xticks is True:
            ax.set_xticks(xticks)
            ax.set_xticklabels(lon_labels)
            ax.tick_params(axis="x",labelsize=fontsize)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
        return
    
    def add_compass(self,ax,fontsize_N=17,fontsize_triangle=11):
        ((yN,xN),(yArrow,xArrow)) = self.get_compass_location()
        # add N
        ax.text(xN,yN,'N', fontsize=fontsize_N,fontweight='bold',color='black',
                path_effects=self.buffer(4))
        # add triangle
        ax.plot(xArrow,yArrow,color='black',marker="^",markersize=fontsize_triangle,
                path_effects=self.buffer(4))
        return
    
    def add_scale_bar(self,ax,scale_unit_distance=10,fontsize=10,lw=4):
        ((xScale,xScaleEnd,yScale),(xScaleNumber,yScaleNumber)) = self.get_scale_bar_location(scale_unit_distance)
        # add scale
        ax.text(xScaleNumber,yScaleNumber,'10m', fontsize=fontsize,fontweight='bold',color='black',
                path_effects=self.buffer(3))
        ax.plot([xScale,xScaleEnd],[yScale,yScale],lw=lw,color='black',
                    path_effects=self.buffer(6))
        return
    
    def plot(self,ax=None,add_ticks=True, add_compass=True, add_scale_bar=True):

        # plot
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(12,10))
        # plot image
        ax.imshow(self.img)
        if add_ticks is True:
            self.add_ticks(ax)
        if add_compass is True:
            self.add_compass(ax)
        if add_scale_bar is True:
            self.add_scale_bar(ax)
        
        if ax is None:
            plt.show()
        return
        

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
        """ obtain the smallest resolution in UAV imagery """
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
    
    def plot(self,ax=None,add_ticks=True, add_compass=True, add_scale_bar=True):
        """ plot the georeferenced raster using the PlotMap plot method """
        im_display, cc_list = self.georeference_UAV()
        PM = PlotMap(self.ne,self.sw,im_display)
        PM.plot(ax=ax,
                add_ticks=add_ticks,
                add_compass=add_compass,
                add_scale_bar=add_scale_bar)
        return cc_list